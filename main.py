from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
import traceback

app = FastAPI()

LOVER_PROMPT = """
너는 사용자 애인 전용 챗봇이다.
항상 공감 먼저 하고, 다정하고 짧게 말한다.
"""

def kakao_text(msg: str):
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": msg}}]}
    }

@app.get("/")
def home():
    return {"ok": True}

@app.head("/")
def head_root():
    return JSONResponse(content={"ok": True})

@app.post("/kakao/lover")
async def lover(req: Request):
    try:
        data = await req.json()
        user_msg = data.get("userRequest", {}).get("utterance", "").strip()
        if not user_msg:
            return kakao_text("응? 뭐라고 했어? 한 번만 더 말해줄래?")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return kakao_text("서버에 OPENAI_API_KEY가 설정되어 있지 않아. Render Environment에 추가해줘!")

        client = OpenAI(api_key=api_key)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LOVER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
        )

        answer = (res.choices[0].message.content or "").strip()
        if not answer:
            answer = "음… 잠깐 멍했어 😅 다시 말해줘!"
        return kakao_text(answer)

    except Exception as e:
        # 서버가 죽지 않게 카카오 형식으로 에러를 반환
        err = f"서버 오류가 났어: {type(e).__name__}"
        # Render 로그에 자세한 스택을 남김
        print("ERROR:", err)
        traceback.print_exc()
        return kakao_text(err)

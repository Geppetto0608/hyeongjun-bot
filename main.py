import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# ==============================
# 카카오 응답 포맷 함수
# ==============================
def kakao_text(msg: str):
    return JSONResponse({
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": msg}}
            ]
        }
    })

# ==============================
# 친구 챗봇 프롬프트
# ==============================
FRIEND_PROMPT = """
너는 내 친구 전용 챗봇이다.
편한 반말로 공감하면서 말해라.

규칙:
- 1~3문장
- 친구처럼 편하게
- 가끔 ㅋㅋ, ㅇㅋ, ㄱㄱ 같은 표현 써도 됨
"""

# ==============================
# 카카오 친구용 엔드포인트
# ==============================
@app.post("/kakao/lover")
async def lover(req: Request):
    try:
        data = await req.json()
        user_msg = data.get("userRequest", {}).get("utterance", "").strip()

        if not user_msg:
            return kakao_text("ㅇ?")

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FRIEND_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=80,
            timeout=8,
        )

        answer = res.choices[0].message.content.strip()
        return kakao_text(answer)

    except Exception as e:
        print("ERROR:", e)
        return kakao_text("야 잠깐 오류남 다시말해봐 ㅋㅋ")


# ==============================
# 서버 루트 테스트용
# ==============================
@app.get("/")
def root():
    return {"status": "friend chatbot running"}

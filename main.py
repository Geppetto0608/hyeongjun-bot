import os
import re
import asyncio
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

app = FastAPI()

# --- 1. 유틸리티 ---
def kakao_text(msg: str) -> dict:
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": msg}}]},
    }

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+", flags=re.UNICODE)

def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text)

def collapse_lines(text: str, max_lines: int = 3) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines]).strip()

def detect_politeness(user_text: str) -> str:
    t = (user_text or "").strip()
    polite_markers = ["요", "니다", "까요", "드립니다", "했어요", "되나요", "주세요", "죄송", "감사"]
    if any(m in t for m in polite_markers): return "polite"
    return "casual"

# --- 2. 페르소나 설정 ---
FRIEND_SYSTEM = """
너는 사용자의 '친구 전용' 챗봇이다. 따뜻하거나 친절한 톤 금지. 툭툭 던지는 친구 말투로.

규칙:
- 이모티콘/느낌표/감탄사 금지
- 한 답변 1~3줄. 길게 설명 금지
- 기본 반말. 사용자가 존댓말이면 너도 존댓말(딱딱하게)로만 맞춰
- 공감은 선택. 꼭 해야 할 때만 한 단어로: "ㅇㅇ", "그럴만함", "알겠음"
- 질문은 최대 1개. 캐묻지 마
- 리액션 짧게: "ㅇㅇ", "ㅇㅋ", "왜", "와이", "?", "ㄱㄱ", "ㄴㄴ", "ㅋㅋ"
- 해결은 A/B 한줄 정리 또는 다음 액션 한줄만 제시
""".strip()

FRIEND_PROFILE = """
[내 프로필(친구용)]
- 생일: 1999/06/08
- 소속: 한양대 대학원(석박통합) / (UNICON 랩)
- 말투: 이모티콘 안씀, 짧게 말함, 현실적으로 정리함
""".strip()

FRIEND_FEWSHOT = [
    {"role": "user", "content": "뭐하냐"},
    {"role": "assistant", "content": "그냥 있음. 와이"},
    {"role": "user", "content": "나 요즘 너무 바빠서 뭐부터 해야할지 모르겠음"},
    {"role": "assistant", "content": "급한거부터. 마감 뭐임"},
]

def build_messages(user_text: str) -> list[dict]:
    mode = detect_politeness(user_text)
    style_addon = "사용자가 존댓말이면 너도 존댓말로." if mode == "polite" else "사용자가 반말이면 너도 반말로."
    
    return [
        {"role": "system", "content": f"{FRIEND_SYSTEM}\n{style_addon}\n\n{FRIEND_PROFILE}"},
        *FRIEND_FEWSHOT,
        {"role": "user", "content": user_text},
    ]

# --- 3. 백그라운드 작업 (핵심: 콜백 보내기) ---
async def background_process(callback_url: str, user_text: str):
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        # OpenAI 호출 (이제 시간 제한 걱정 없음. 30초 걸려도 됨)
        res = await client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=build_messages(user_text),
            max_tokens=80,
            temperature=0.6,
        )
        answer = res.choices[0].message.content.strip()
        answer = strip_emojis(answer)
        answer = collapse_lines(answer, max_lines=3)

        # ★ 카카오 서버로 답변 전송 (POST)
        async with httpx.AsyncClient() as http_client:
            await http_client.post(
                callback_url,
                json=kakao_text(answer),
                timeout=10.0
            )
            print(f"[Callback Success] Sent: {answer}")

    except Exception as e:
        print(f"[Callback Error] {e}")

# --- 4. 메인 엔드포인트 ---
@app.post("/kakao/lover")
async def kakao_friend(req: Request, background_tasks: BackgroundTasks):
    try:
        data = await req.json()
        user_request = data.get("userRequest", {})
        user_text = user_request.get("utterance", "").strip()
        callback_url = user_request.get("callbackUrl")  # ★ 카카오가 준 '답장 주소'

        if not user_text:
            return JSONResponse(kakao_text("?"))

        # 1. 콜백 URL이 있으면 -> "useCallback: true" 먼저 뱉고 뒤에서 처리
        if callback_url:
            print(f"[Async] Background Task Started for: {user_text}")
            background_tasks.add_task(background_process, callback_url, user_text)
            
            # ★ 카카오에게: "알겠어, 곧 보낼게" (즉시 응답)
            return JSONResponse({
                "version": "2.0",
                "useCallback": True
            })

        # 2. 콜백 URL이 없으면 (테스트 환경 등) -> 그냥 기다려서 답함
        else:
            print("[Sync] No Callback URL. Processing directly.")
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            res = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=build_messages(user_text),
                max_tokens=80,
                temperature=0.6,
                timeout=4.0 
            )
            answer = strip_emojis(res.choices[0].message.content.strip())
            return JSONResponse(kakao_text(answer))

    except Exception as e:
        print(f"[Error] {e}")
        # 콜백 모드일 땐 여기서 에러 리턴해도 사용자한텐 안 보임 (이미 useCallback 나감)
        return JSONResponse(kakao_text("오류."))

@app.post("/kakao/lover/")
async def kakao_friend_slash(req: Request, background_tasks: BackgroundTasks):
    return await kakao_friend(req, background_tasks)

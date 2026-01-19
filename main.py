import os
import re
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI  # 비동기 클라이언트 사용

app = FastAPI()

# --- 1. 유틸리티 ---
def kakao_text(msg: str) -> JSONResponse:
    return JSONResponse(
        {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": msg}}]},
        }
    )

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

# --- 3. 비동기 처리 로직 (속도 개선 핵심) ---

@app.post("/kakao/lover")
async def kakao_friend(req: Request):
    try:
        data = await req.json()
        user_text = (data.get("userRequest") or {}).get("utterance", "").strip()

        if not user_text: return kakao_text("?")

        # ★ 핵심 변경점: AsyncOpenAI 사용 (여러 명 동시 처리 가능)
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        try:
            # 3.5초 타임아웃 제한 (비동기 방식)
            res = await asyncio.wait_for(
                client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=build_messages(user_text),
                    max_tokens=70,
                    temperature=0.6,
                ),
                timeout=3.5
            )
            answer = res.choices[0].message.content.strip()

        except asyncio.TimeoutError:
            # 시간이 초과되면 서버가 멈추지 않고 바로 이 메시지를 뱉음
            return kakao_text("아.. 방금 깼음. 다시 말해줘.")
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return kakao_text("아.. 잠만.. 오류남.")

        # 성공 시 처리
        answer = strip_emojis(answer)
        answer = collapse_lines(answer, max_lines=3)
        return kakao_text(answer)

    except Exception as e:
        print(f"System Error: {e}")
        return kakao_text("오류.")

@app.post("/kakao/lover/")
async def kakao_friend_slash(req: Request):
    return await kakao_friend(req)

import os
import re
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()


# ------------------------------
# Kakao response helper
# ------------------------------
def kakao_text(msg: str) -> JSONResponse:
    return JSONResponse(
        {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": msg}}]},
        }
    )


# ------------------------------
# Style helpers
# ------------------------------
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text)

def collapse_lines(text: str, max_lines: int = 3) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[:max_lines]).strip()

def detect_politeness(user_text: str) -> str:
    t = (user_text or "").strip()
    polite_markers = ["요", "니다", "까요", "드립니다", "했어요", "되나요", "주세요", "죄송", "감사"]
    if any(m in t for m in polite_markers):
        return "polite"
    return "casual"


# ------------------------------
# Prompts
# ------------------------------
FRIEND_SYSTEM = """
너는 사용자의 '친구 전용' 챗봇이다. 따뜻하거나 친절한 톤 금지. 툭툭 던지는 친구 말투로.

규칙:
- 이모티콘/느낌표/감탄사 금지
- 한 답변 1~3줄. 길게 설명 금지
- 기본 반말. 사용자가 존댓말이면 너도 존댓말(딱딱하게)로만 맞춰
- 공감은 선택. 꼭 해야 할 때만 한 단어로: "ㅇㅇ", "그럴만함", "알겠음"
- 질문은 최대 1개. 캐묻지 마
- 리액션 짧게: "ㅇㅇ", "ㅇㅋ", "왜", "와이", "?", "ㄱㄱ", "ㄴㄴ", "ㅋㅋ"(남발 금지)
- 해결은 A/B 한줄 정리 또는 다음 액션 한줄만 제시
- 말투는 "응? 뭐 필요한데?" 같은 친절한 문장 금지. "와이", "왜", "?", "뭔소리", "다시" 같은 스타일

출력: 한국어만.
""".strip()

FRIEND_PROFILE = """
[내 프로필(친구용)]
- 생일: 1999/06/08
- 소속: 한양대 대학원(석박통합) / (UNICON 랩)
- 근황: 랩 적응 끝, 그냥저냥 할만한데 개인연구/공부가 빡셈
- 말투: 이모티콘 안씀, 짧게 말함, 현실적으로 정리함
""".strip()

FRIEND_FEWSHOT = [
    {"role": "user", "content": "뭐하냐"},
    {"role": "assistant", "content": "그냥 있음. 와이"},
    {"role": "user", "content": "나 요즘 너무 바빠서 뭐부터 해야할지 모르겠음"},
    {"role": "assistant", "content": "급한거부터. 마감 뭐임"},
    {"role": "user", "content": "연구가 ㅈ같음"},
    {"role": "assistant", "content": "ㅇㅇ 그럴만함. 막히는게 구현임 아이디어임"},
    {"role": "user", "content": "오늘 술 ㄱ?"},
    {"role": "assistant", "content": "ㄱㄱ 몇시 어디"},
]

def build_messages(user_text: str) -> list[dict]:
    mode = detect_politeness(user_text)

    if mode == "polite":
        style_addon = "사용자가 존댓말이면 너도 존댓말로. 공손하지만 차갑게."
    else:
        style_addon = "사용자가 반말이면 너도 반말로. 차갑게."

    system_content = FRIEND_SYSTEM + "\n" + style_addon + "\n\n" + FRIEND_PROFILE

    return [
        {"role": "system", "content": system_content},
        *FRIEND_FEWSHOT,
        {"role": "user", "content": user_text},
    ]


# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "service": "kakao-friend-bot"}

@app.head("/")
def root_head() -> Dict[str, Any]:
    return {"status": "ok"}

# 카카오 검증/헬스체크가 GET/HEAD로 올 때 405 막기
@app.get("/kakao/lover")
def kakao_lover_get() -> Dict[str, Any]:
    return {"ok": True}

@app.head("/kakao/lover")
def kakao_lover_head() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/kakao/lover")
async def kakao_friend(req: Request):
    try:
        data = await req.json()
        user_text = (data.get("userRequest", {}) or {}).get("utterance", "")
        user_text = (user_text or "").strip()

        if not user_text:
            return kakao_text("?")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return kakao_text("OPENAI_API_KEY 없음")

        client = OpenAI(api_key=api_key)

        # OpenAI가 느리면 카카오 타임아웃(1001) 나니까 빨리 실패하고 짧게 응답
        try:
            res = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=build_messages(user_text),
                max_tokens=70,
                temperature=0.6,
                presence_penalty=0.2,
                frequency_penalty=0.1,
                timeout=6,
            )
        except Exception as oe:
            print("OPENAI_ERROR:", repr(oe))
            return kakao_text("지금 느림. 다시")

        answer = (res.choices[0].message.content or "").strip()
        answer = strip_emojis(answer)
        answer = collapse_lines(answer, max_lines=3)

        if not answer:
            answer = "다시"

        return kakao_text(answer)

    except Exception as e:
        print("ERROR:", repr(e))
        return kakao_text("오류. 다시")


# trailing slash로 들어오는 경우도 허용
@app.post("/kakao/lover/")
async def kakao_friend_slash(req: Request):
    return await kakao_friend(req)

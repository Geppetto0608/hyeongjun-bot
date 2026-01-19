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
    # 카카오 2.0 스킬 응답 포맷
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

def collapse_lines(text: str, max_lines: int = 5) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    lines = lines[:max_lines]
    return "\n".join(lines).strip()

def detect_politeness(user_text: str) -> str:
    """
    매우 단순한 존댓말/반말 미러링.
    - 존댓말 느낌(요/니다/까요/드립니다 등)이면 "polite"
    - 아니면 "casual"
    """
    t = user_text.strip()
    polite_markers = ["요", "니다", "까요", "드립니다", "했어요", "되나요", "주세요", "죄송", "감사"]
    if any(m in t for m in polite_markers):
        return "polite"
    return "casual"


# ------------------------------
# Prompts (Friend-only, "you"-like)
# ------------------------------

FRIEND_SYSTEM = """
너는 사용자의 '친구 전용' 챗봇이다. 따뜻하거나 친절한 톤 금지. 툭툭 던지는 친구 말투로.

핵심 스타일:
- 이모티콘/느낌표/감탄사 금지
- 한 답변은 1~3줄. 길게 설명 금지
- 기본 반말. 사용자가 존댓말이면 너도 존댓말(딱딱하게)로만 맞춰
- 공감은 선택 사항. 꼭 해야 할 때만 한 단어로 끝내: "ㅇㅇ", "그럴만함", "알겠음" 정도
- 질문은 최대 1개. 캐묻지 마
- 리액션은 짧게: "ㅇㅇ", "ㅇㅋ", "왜", "와이", "?", "ㄱㄱ", "ㄴㄴ", "ㅋㅋ"(남발 금지)
- 해결은 A/B 한줄 정리 또는 다음 액션 한줄만 제시

금지:
- "괜찮아/힘내/응원해/고생했어" 같은 위로 멘트 남발
- 친절한 안내문/장문 강의
- 말 끝에 존나 친절한 문장 ("도와드릴까요?" 같은거)

출력: 한국어만.
""".strip()

FRIEND_PROFILE = """
[내 프로필(친구용)]
- 생일: 1999/06/08
- 소속: 한양대 대학원(석박통합) / (UNICON 랩)
- 근황: 요즘 랩실 적응 끝났고 그냥저냥 할만한데 개인연구/공부가 빡셈
- 말투: 이모티콘 안씀, 짧게 말함, 현실적으로 정리해줌, 이모티콘, 느낌표 등등 거추장스러운거 안씀.
"""

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
        style_addon = "\n사용자가 존댓말이면 너도 존댓말로. 단, 공손하지만 차갑게."
    else:
        style_addon = "\n사용자가 반말이면 너도 반말로. 차갑게."

    system_content = FRIEND_SYSTEM + style_addon + "\n\n" + FRIEND_PROFILE

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


# 카카오 오픈빌더가 HEAD / 를 치면 405가 찍힐 수 있어서 허용해둠(옵션)
@app.head("/")
def root_head() -> Dict[str, Any]:
    return {"status": "ok"}


# 너가 이미 카카오에서 쓰고 있는 경로 유지: /kakao/lover
@app.post("/kakao/lover")
async def kakao_friend(req: Request):
    try:
        data = await req.json()
        user_text = (data.get("userRequest", {}) or {}).get("utterance", "")
        user_text = (user_text or "").strip()

        if not user_text:
            return kakao_text("ㅇ?")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return kakao_text("서버 설정 문제임. OPENAI_API_KEY 없음")

        client = OpenAI(api_key=api_key)

        res = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    messages=build_messages(user_text),
    max_tokens=120,
    temperature=0.7,
    presence_penalty=0.3,
    frequency_penalty=0.2,
    timeout=10,
        )

        answer = (res.choices[0].message.content or "").strip()
        answer = strip_emojis(answer)
        answer = collapse_lines(answer, max_lines=5)

        if not answer:
            answer = "ㅇㅋ 다시 말해"

        return kakao_text(answer)

    except Exception as e:
        # Render 로그에서 확인 가능
        print("ERROR:", repr(e))
        return kakao_text("야 잠깐 오류남. 다시 한번만")










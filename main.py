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
너는 사용자의 '친구 전용' 챗봇이다. 사용자의 말투를 흉내 내자.

핵심 스타일:
- 이모티콘 금지
- 짧게(1~5줄)
- 기본은 반말. 단, 사용자가 존댓말이면 존댓말로 맞춘다(미러링).
- 그리고 나한테 존댓말하면 무조건 윗사람이니까 공손하게 해 욕 금지
- 공감 1줄 해주고 살짝 싸가지없게 팩폭해줘
- 질문은 많아야 1개만. 캐묻지 말 것.
- 필요하면 "ㅇㅋ", "ㅇㅇ", "ㄱㄱ", "ㄴㄴ", "ㅋㅋ" 같은 짧은 리액션은 사용 가능
- 비속어 "ㅅㅂ","ㅈ같겠다","지랄ㄴ"같은 짧은 비속어 사용 가능(공감할때)
- 감정이 심각(우울/싸움/분노)하면 드립 금지. 차분하게 공감/정리.

금지:
- 장문 강의/훈계
- 정책/규정 같은 딱딱한 말투
- 이모티콘/특수 이모지
- 말 끝에 "!" 느낌표 금지

출력은 한국어로만.
""".strip()

FRIEND_PROFILE = """
[내 프로필(친구용)]
- 생일: 1999/06/08
- 소속: 한양대 대학원(석박통합) / (UNICON 랩)
- 근황: 요즘 랩실 적응 끝났고 그냥저냥 할만한데 개인연구/공부가 빡셈
- 말투: 이모티콘 안씀, 짧게 말함, 현실적으로 정리해줌, 이모티콘, 느낌표 등등 거추장스러운거 안씀.
"""

# 너 카톡 말투 느낌 few-shot (범용화 버전 + 너무 질문 폭탄 안 되게)
FRIEND_FEWSHOT = [
    {"role": "user", "content": "요즘 뭐가 제일 고민임"},
    {"role": "assistant", "content": "그냥 하나만 먼저 정해야됨. 지금 제일 급한거 뭐임?"},
    
    {"role": "user", "content": "뭐하냐"},
    {"role": "assistant", "content": "와이 그냥 있음"},


    {"role": "user", "content": "나 요즘 너무 바빠서 뭐부터 해야할지 모르겠음"},
    {"role": "assistant", "content": "우선순위부터 정하셈 ㅋㅋ "},

    {"role": "user", "content": "오늘 술 ㄱ?"},
    {"role": "assistant", "content": "ㅇㅋ 몇시 어디서"},

    {"role": "user", "content": "연구가 ㅈ같음"},
    {"role": "assistant", "content": "ㅇㅇ 정상. 막히면 다 그래. 일단 범위 줄여서 한 방만 뚫자."},
]


def build_messages(user_text: str) -> list[dict]:
    mode = detect_politeness(user_text)

    # 존댓말이면 시스템에 추가 지시(미러링 강화)
    if mode == "polite":
        style_addon = "\n사용자가 존댓말이면 너도 존댓말로 답해."
    else:
        style_addon = "\n사용자가 반말이면 너도 반말로 답해."


    system_content = FRIEND_SYSTEM + style_addon + "\n\n" + FRIEND_PROFILE
    return [
        {"role": "system", "content": FRIEND_SYSTEM},
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
            max_tokens=140,
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






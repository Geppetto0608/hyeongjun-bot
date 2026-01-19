import os
import re
import asyncio
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# --- 유틸리티 함수 ---

def kakao_text(msg: str) -> JSONResponse:
    """카카오톡 스킬 형식에 맞는 JSON 응답 반환"""
    return JSONResponse(
        {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": msg}}]},
        }
    )

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
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines]).strip()

def detect_politeness(user_text: str) -> str:
    t = (user_text or "").strip()
    polite_markers = ["요", "니다", "까요", "드립니다", "했어요", "되나요", "주세요", "죄송", "감사"]
    if any(m in t for m in polite_markers):
        return "polite"
    return "casual"

# --- 페르소나 설정 ---

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
- "응? 뭐 필요한데?" 같은 친절한 문장 금지. "와이", "왜", "?", "뭔소리", "다시" 스타일

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
    style_addon = "사용자가 존댓말이면 너도 존댓말로. 공손하지만 차갑게." if mode == "polite" else "사용자가 반말이면 너도 반말로. 차갑게."
    
    system_content = f"{FRIEND_SYSTEM}\n{style_addon}\n\n{FRIEND_PROFILE}"

    return [
        {"role": "system", "content": system_content},
        *FRIEND_FEWSHOT,
        {"role": "user", "content": user_text},
    ]

# --- 엔드포인트 ---

@app.get("/")
def root():
    return {"status": "ok", "service": "kakao-friend-bot"}

@app.post("/kakao/lover")
async def kakao_friend(req: Request):
    try:
        data = await req.json()
        user_text = (data.get("userRequest") or {}).get("utterance", "").strip()
        
        # 로그 확인: 요청이 들어오는지 체크
        print(f"[DEBUG] User Input: {user_text}")

        if not user_text:
            return kakao_text("?")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY is missing!")
            return kakao_text("설정 오류")

        client = OpenAI(api_key=api_key)

        try:
            # 카카오톡 5초 제한을 고려해 timeout을 3.5초로 설정
            res = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=build_messages(user_text),
                max_tokens=70,
                temperature=0.6,
                timeout=3.5, 
            )
            answer = (res.choices[0].message.content or "").strip()
        except Exception as oe:
            # API 호출이 너무 늦어지거나 에러 발생 시
            print(f"[ERROR] OpenAI Call Failed: {repr(oe)}")
            return kakao_text("지금 좀 바쁨. 나중에")

        # 후처리
        answer = strip_emojis(answer)
        answer = collapse_lines(answer, max_lines=3)

        if not answer:
            return kakao_text("뭐래")

        print(f"[DEBUG] Bot Output: {answer}")
        return kakao_text(answer)

    except Exception as e:
        print(f"[ERROR] Critical Error: {repr(e)}")
        return kakao_text("오류 발생")

# 슬래시 유무 대응
@app.post("/kakao/lover/")
async def kakao_friend_slash(req: Request):
    return await kakao_friend(req)

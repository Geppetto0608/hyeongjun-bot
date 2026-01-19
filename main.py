from fastapi import FastAPI, Request
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

LOVER_PROMPT = "너는 다정한 애인 챗봇이다."

@app.get("/")
def home():
    return {"ok": True}

@app.post("/kakao/lover")
async def lover(req: Request):
    data = await req.json()
    user_msg = data["userRequest"]["utterance"]

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":LOVER_PROMPT},
            {"role":"user","content":user_msg}
        ]
    )

    answer = res.choices[0].message.content.strip()

    return {
        "version":"2.0",
        "template":{
            "outputs":[{"simpleText":{"text":answer}}]
        }
    }


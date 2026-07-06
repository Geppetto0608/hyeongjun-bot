# Hyeongjun Bot

Personal Python bot deployment experiment using FastAPI, Render, and an OpenAI-compatible chat API endpoint.

## What It Does

- Exposes Kakao-style webhook endpoints with FastAPI.
- Returns short text responses formatted for Kakao chatbot payloads.
- Supports callback-based background processing for longer model calls.
- Reads `OPENAI_API_KEY` from the deployment environment.

## Repository Status

This is a small deployment experiment, not a production chatbot.

Validation status:

- FastAPI app source is present.
- Render deployment config is present.
- Production uptime, latency, and chatbot quality metrics: TBD.

## Files

```text
main.py          FastAPI app and Kakao webhook handlers
requirements.txt Python dependencies
render.yaml      Render deployment configuration
```

## Local Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uvicorn main:app --reload
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
uvicorn main:app --reload
```

## Notes For Portfolio Review

- This repo demonstrates lightweight API deployment and webhook handling.
- Do not commit real API keys or private Kakao channel configuration.
- Add screenshots or request/response examples before treating it as a polished portfolio item.


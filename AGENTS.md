# AGENTS.md

## Project Purpose

This repository is a small FastAPI chatbot deployment experiment for Kakao-style webhook responses.

## Rules

- Do not commit API keys, tokens, Kakao secrets, or private deployment credentials.
- Keep `OPENAI_API_KEY` in environment variables or Render secrets only.
- Preserve the Kakao response payload format unless the user asks to change integrations.
- Keep responses short because the current bot persona and channel format expect concise messages.
- Do not claim production reliability, uptime, or user metrics unless evidence is added.

## Validation

- Run `python -m py_compile main.py` when Python is available.
- Run `uvicorn main:app --reload` for local smoke testing when dependencies are installed.
- If testing requires external Kakao callback infrastructure, mark it `TBD`.


Read PLAN.md for the full project plan.
# Project 3: Agentic RAG over Live Data & Memory

## About me
I'm a beginner learning to become an AI engineer. Explain in simple words,
step by step, with a small example. Explain WHY, not just WHAT. Keep it short.

## Tech stack
- Windows + PowerShell, Python 3.12
- Ollama (llama3.2, nomic-embed-text), ChromaDB, rank-bm25, SQLite,
  Streamlit, requests, pydantic, pytest
- Free APIs: Frankfurter (currency), Open-Meteo (weather)
- All settings live in config.py

## Plan (ONE step at a time)
0 setup + health check | 1 currency tool | 2 weather tool | 3 docs tool
4 multi-tool agent | 5 corrective RAG | 6 long-term memory | 7 freshness + cache
8 Streamlit UI | 9 evaluation | 10 README + notes

## Rules
- Never jump ahead to a later step.
- Make a plan and wait for my OK before writing code.
- Every API call: timeout= and raise_for_status().
- After writing code, run it and show me the output.

## After each step
1. Update LEARNING.md (what we built + key concepts in simple words)
2. git commit with a clear message
3. git push
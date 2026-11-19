# Project plan: Agentic RAG over Live Data & Memory

## Goal
Assistant that answers from company docs + live data (currency, weather)
+ long-term memory. Example: "Flying to Tokyo. Hotel limit in yen + weather now?"
→ docs ($220/night) + live JPY rate + live weather + memory (user is vegetarian).

## Steps (each: goal → files → done when)
0 Setup: config.py, check_setup.py → all 5 checks ✅, pushed to GitHub
1 Currency: tools/currency.py get_exchange_rate(from, to) → rate, date, source, as_of;
  bad code/timeout = friendly error; pytest tests
2 Weather: tools/weather.py get_weather(city) → city→coords→weather in words;
  "city not found" handled; tests
3 Docs: tools/docs.py search_documents(query, role) → reuse Project 1
  (chunking, hybrid search BM25+embeddings, role filter in DB query)
4 Agent: agent.py → tool registry, loop MAX_STEPS=4, trace of tool calls;
  Tokyo question uses 3 tools
5 Corrective RAG: grader.py → grade chunks (JSON+Pydantic), rewrite + retry once,
  else "I don't know"
6 Memory: memory.py → save_memory, recall_memory, forget_me (SQLite + ChromaDB)
7 Cache: cache.py → TTL (currency 60 min, weather 15 min), "as of" times,
  re-ingest changed docs via file hash
8 UI: app.py Streamlit → chat, role picker, "Tools used" panel, memory sidebar
9 Eval: eval/ → 30 questions, mocked APIs, tool/answer/memory/refusal scores
10 README + notes

## Project 1 code
Reuse from C:\Users\annma\enterprise-knowledge-agent
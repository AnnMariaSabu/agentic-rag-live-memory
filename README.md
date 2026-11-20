# Agentic RAG over Live Data & Memory

A local AI helpdesk assistant for the employees of a (fictional) company, **Acme Corp**.
It answers questions from **company documents**, **live exchange rates**, **live weather**
and **long-term memory about the user**. An agent decides which of these each question needs.

Everything runs on your own computer: a local LLM through [Ollama](https://ollama.com),
plus two free public APIs that need no key.

> **You:** I'm flying to Tokyo next week. What's my hotel limit per night in yen, and what's the weather there now?
>
> **Assistant:** The hotel limit per night in Tokyo is 34,900.8 yen. The current weather in Tokyo is clear with a temperature of 6.5 °C.
> 🕒 Exchange rate USD→JPY: ECB rate of 2026-11-20, fetched 2026-11-20 17:30 UTC (just fetched)
>
> 🔧 Tools used: `search_documents` (travel policy: $220/night) → `get_weather("Tokyo")` → `get_exchange_rate(USD→JPY, amount=220)`

This is Project 3 of a learning path. It reuses the document search from Project 1
(*enterprise-knowledge-agent*) and builds an agent, self-checking retrieval, memory, caching,
a UI and an evaluation around it.

---

## What it can do

| Feature | What it means |
|---|---|
| 📄 **Document search** | Hybrid search: keyword (BM25) + meaning (embeddings), merged with RRF, over 8 policy files (.txt, .pdf, .docx) |
| 🔒 **Role-based access** | Employee / HR / IT / Admin. Restricted files are filtered **inside the database query**, so they are never fetched for the wrong role |
| 💱 **Live currency** | [Frankfurter](https://frankfurter.dev) (European Central Bank rates); the code does the maths (`amount` → `converted`) |
| 🌤️ **Live weather** | [Open-Meteo](https://open-meteo.com): city → coordinates → current weather in words |
| 🤖 **Agent** | Tool registry, router (skip tools for small talk), loop capped at 4 steps, full trace of every tool call |
| ✅ **Corrective RAG** | The LLM grades each retrieved chunk (JSON checked by Pydantic), rewrites the query and retries once, else says *"I don't know"* |
| 🧠 **Long-term memory** | Per-user facts in SQLite + ChromaDB; recalled automatically; **Forget me** deletes everything |
| 🕒 **Freshness + cache** | TTL cache (currency 60 min, weather 15 min), "as of" times on every live fact, re-ingest only changed files (SHA-256) |
| 🖥️ **Streamlit UI** | Chat, role picker, "Tools used" panel with sources, memory sidebar |
| 📊 **Evaluation** | 30 questions with mocked APIs, scored on tools, answer, refusal and memory |

## How it works

```
                          ┌──────────────── agent.py ────────────────┐
 question ──► recall memories ──► router: TOOLS or CHAT?             │
  + role                          │                                  │
  + user id                       ▼                                  │
                     loop (max 4 steps): LLM picks tools ──► run ───┤
                                  ▲                          │       │
                                  └──── tool results ◄───────┘       │
                                                                     │
   tools:  search_documents ─► grader.py (grade → rewrite → retry)   │
           get_exchange_rate ─► cache ─► Frankfurter                 │
           get_weather       ─► cache ─► Open-Meteo                  │
           save_memory       ─► SQLite + ChromaDB                    │
                          └──────────────────────────────────────────┘
                                  │
          answer + trace + freshness notes + memories used ──► app.py (Streamlit)
```

**Design rules that made the small model (llama3.2, 3B) reliable:**

- **Code decides what must be safe or exact:** the role and user ID are injected by code (the
  LLM can't choose them), money conversion is computed by code, "I don't know" is forced by code
  when every search came back empty, and freshness notes are built by code.
- **When a prompt can't fix it, add structure:** a router stops tool calls for "Hi!". A dependency
  rule postpones a currency conversion that was requested before the policy amount was known, then
  nudges the model to convert the real amount. If the model *writes* about a tool ("I'll call
  get_exchange_rate…") instead of calling it, code nudges it to call the tool. A live-data error is
  shown as that error, never as "I don't know based on the company documents".
- **Destructive actions are for humans only:** `forget_me` is a button in the UI, never an LLM tool.

## Setup (Windows + PowerShell)

Requirements: Python 3.12, [Ollama](https://ollama.com), internet access for the two free APIs.

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe check_setup.py   # 5 health checks, expect 5/5 ✅
.\.venv\Scripts\python.exe ingest.py        # build the document search database
```

## Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py         # web UI → http://localhost:8501
.\.venv\Scripts\python.exe agent.py               # 5-question demo in the terminal
.\.venv\Scripts\python.exe agent.py --chat        # chat in the terminal
.\.venv\Scripts\python.exe ingest.py --rebuild    # rebuild the document database from scratch
```

## Tests and evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not live"   # fast offline tests (fake LLM, fake APIs)
.\.venv\Scripts\python.exe -m pytest                 # + live tests (needs Ollama + internet)
.\.venv\Scripts\python.exe -m eval.run_eval          # 30-question evaluation → eval/report.md
```

### Evaluation results

30 questions, real `llama3.2` + real document search, grader and memory; currency and weather
APIs **mocked** with fixed numbers so the expected answers never change
([full report](eval/report.md)).

| Score | What it checks | Result |
|---|---|---|
| tools | called exactly the expected tools | 30/30 |
| answer | answer contains the expected facts, and no forbidden ones (e.g. salaries for an Employee) | 30/30 |
| refusal | says "I don't know" when the documents don't cover it, and only then | 30/30 |
| memory | the fact was saved / recalled | 2/2 |

| Category | Passed | Category | Passed |
|---|---|---|---|
| docs (incl. role security) | 10/10 | small talk | 2/2 |
| currency | 4/4 | refusal | 4/4 |
| weather | 4/4 | memory | 2/2 |
| mixed (2–3 tools) | 4/4 | **all checks** | **30/30** |

The first run scored **27/30** and found two bugs that the unit tests had missed: a live-data
error was answered with the "no company documents" phrase, and the model *described* a currency
conversion instead of calling the tool. Both were fixed in code (with tests), then the full
evaluation was run again. The scores check facts, not style, and come from one run at
temperature 0.

## Project structure

```
config.py            all settings (models, URLs, timeouts, TTLs, chunk sizes)
check_setup.py       Step 0: health check of LLM, embeddings, both APIs, tool calling
tools/currency.py    Step 1: get_exchange_rate (Frankfurter) + amount conversion + cache
tools/weather.py     Step 2: get_weather (Open-Meteo geocoding + forecast) + cache
tools/docs.py        Step 3: search_documents (hybrid BM25 + embeddings, role filter)
rag/                 loaders, chunker, permissions (reused from Project 1), embeddings
ingest.py            read → chunk → embed → ChromaDB; skips unchanged files (file hash)
agent.py             Step 4: registry, router, loop, trace, memory recall, freshness notes
grader.py            Step 5: corrective RAG (grade → rewrite → retry once → "I don't know")
memory.py            Step 6: save / recall / list / forget_me (SQLite + ChromaDB)
cache.py             Step 7: TTL cache in SQLite
app.py               Step 8: Streamlit UI
eval/                Step 9: questions.json, run_eval.py, report.md
tests/               pytest suite (offline by default, `live` marker for real services)
data/documents/      the 8 Acme policy documents (hr, it, travel; some restricted)
LEARNING.md          what each step taught, in simple words
```

## Limitations

- **Small local model.** llama3.2 (3B) is fast and free but sometimes phrases numbers loosely
  (e.g. "34900.8 yen") and needs the guard rails described above. A larger model would need fewer.
- **One question at a time.** The agent doesn't use the chat history for follow-ups like
  "and for domestic trips?". Long-term memory gives continuity across chats, not within one.
- **Demo identity.** The role and user ID are chosen in the UI. A real deployment would take them
  from a login, never from the user.
- **Grader cost.** Each document search makes up to 7 extra LLM calls (3 grades, 1 rewrite,
  3 grades), so document questions take longer than live-data questions.
- **Fictional data.** The company, policies and salary bands are made up for learning.

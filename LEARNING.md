# Learning Notes — Agentic RAG over Live Data & Memory

## Step 0: Setup + health check

### What we built
| File | Purpose |
|---|---|
| `.venv/` | Private Python environment for this project |
| `requirements.txt` | List of packages (with exact versions) so anyone can rebuild the same setup |
| `config.py` | Every setting in one place (URLs, model names, timeouts) |
| `check_setup.py` | Tests 5 things and prints ✅/❌ for each |
| `.gitignore` | Tells git which files NOT to save (venv, caches, databases) |

Run it: `.\.venv\Scripts\python.exe check_setup.py` → expect `5/5 checks passed`.

### Key concepts (simple words)

- **Virtual environment (venv)** — a separate box of packages just for this project.
  *Why:* project A needs version 1 of a library, project B needs version 2 → no fights.

- **Config file** — one file with all settings.
  *Why:* to switch from `llama3.2` to another model, change one line, not ten files.

- **API** — a URL your code asks for data. Example: ask Frankfurter
  `/latest?from=USD&to=JPY` → it answers `{"rates": {"JPY": 159.97}}`.

- **`timeout=` and `raise_for_status()`** — two safety belts on every request.
  - `timeout` = "give up after N seconds" so the program never hangs forever.
  - `raise_for_status()` = "if the server says error (404, 500…), stop loudly"
    instead of silently using bad data.

- **Embedding** — turns text into a list of numbers (here 768 numbers).
  Texts with similar meaning get similar numbers. *Why:* this is how we'll search
  documents by meaning later (Step 3).

- **Tool calling** — we describe a tool to the LLM (name + inputs). The LLM does not
  run it; it *replies* "please call `get_weather(city='Tokyo')`". Our code runs it.
  *Why:* this is the heart of an agent (Step 4).

- **Health check** — prove each outside piece works *before* building on it.
  *Why:* if something breaks later, we know the bug is in new code, not the setup.

### Lesson learned: cold start
The first run, LLM chat failed with `ReadTimeout`, but a second run passed.
Reason: Ollama's first request must load the 2 GB model into memory, which took
longer than 30 s. Fix: a separate, longer `LLM_TIMEOUT = 120` in `config.py`
(web APIs keep the short 30 s timeout).

---

## Step 1: Currency tool

### What we built
| File | Purpose |
|---|---|
| `tools/currency.py` | `get_exchange_rate(from, to)` → live rate from Frankfurter |
| `tests/test_currency.py` | 14 tests (13 mocked + 1 live) |
| `pytest.ini` | pytest settings + the `live` marker |

```python
get_exchange_rate("usd", "jpy")
# {'from': 'USD', 'to': 'JPY', 'rate': 159.97, 'date': '2026-11-19',
#  'source': 'Frankfurter (ECB)', 'as_of': '2026-11-19T17:50:53+00:00'}
get_exchange_rate("US", "JPY")
# {'error': "'US' is not a valid currency code. Use 3 letters, e.g. USD, EUR, JPY."}
```

Run: `.\.venv\Scripts\python.exe -m pytest` (offline: add `-m "not live"`).

### Key concepts (simple words)

- **Tool** — a normal Python function the agent can call to get facts the LLM
  doesn't know (the LLM's knowledge is frozen; today's rate is not in it).

- **Never crash, return a friendly error** — on failure we return
  `{"error": "..."}` instead of raising. *Why:* in Step 4 the agent hands this
  text to the LLM, which can then tell the user "that code doesn't exist"
  instead of the whole program dying.

- **Validate input early** — `" usd"` → `"USD"`, and `"US"` is rejected
  *before* calling the API. *Why:* saves a slow network call and gives a clearer
  message.

- **`date` vs `as_of`** — `date` = day the bank published the rate;
  `as_of` = moment *we* fetched it. *Why:* live data goes stale; Step 7's cache
  uses `as_of` to know when to refresh.

- **Pydantic model** — describes the correct answer shape (e.g. `rate > 0`).
  If the API ever sends something weird, validation fails and we return a
  friendly error instead of passing bad data on.

- **Mocking in tests** — replace the real `requests.get` with a fake that
  returns what we choose (a good reply, a 404, a timeout…).
  *Why:* tests become fast, work offline, and give the same result every time,
  even though real rates change daily. One `live` test still checks the real API.

### Lesson learned: check the real API first
We called Frankfurter by hand before coding: unknown codes give **404**, and
`USD→USD` gives **422 "bad currency pair"**. So we handle same-currency
ourselves (rate 1.0, no API call) instead of guessing.

---

## Step 2: Weather tool

### What we built
| File | Purpose |
|---|---|
| `tools/weather.py` | `get_weather(city)` → current weather in words |
| `tests/test_weather.py` | 12 tests (11 mocked + 1 live) |
| `config.py` | added `GEOCODING_URL` |

```
get_weather("Tokyo")  → summary: "Tokyo, Japan: Clear sky, 10.0 °C (feels like 6.9 °C), wind 5.6 km/h"
get_weather("Atlantisxyz") → error: "City 'Atlantisxyz' not found. Check the spelling..."
```

### Key concepts (simple words)

- **Two calls in a chain** — weather services only understand map coordinates,
  not names. So: call 1 (geocoding) turns "Tokyo" into lat/lon, call 2 gets the
  weather for those coordinates. Like asking "what's the address?" before
  sending a taxi.

- **Geocoding** — turning a place name into coordinates (latitude, longitude).

- **Codes → words** — the API says `weather_code: 3`; we translate with a small
  lookup table to "Overcast". *Why:* people and the LLM understand words, not codes.
  Unknown code → "Unknown conditions" instead of crashing.

- **"Success" can still mean "nothing found"** — a city that doesn't exist does
  NOT return an error; it returns 200 OK with an empty answer. So
  `raise_for_status()` alone isn't enough — we must also check that results exist.

- **Stop early** — if the city isn't found, we don't make the weather call at all.

- **Same tool pattern as Step 1** — check input → call API → handle problems
  politely → return a neat, checked answer (Pydantic) with `source` and `as_of`.

### Lesson learned: look at the real reply first
Calling the API by hand showed that "not found" looks like success (200 + no
`results`). Without checking, we would have guessed a 404 and missed the case.

---

## Step 3: Docs tool

### What we built
| File | Purpose |
|---|---|
| `data/documents/` | 8 company documents copied from Project 1 (hr, it, travel) |
| `rag/loaders.py`, `rag/chunker.py`, `rag/permissions.py` | reused from Project 1 |
| `rag/embeddings.py` | text → vector via Ollama (with timeout) |
| `ingest.py` | build the search database: read → chunk → embed → save |
| `tools/docs.py` | `search_documents(query, role)` → best 3 chunks + sources |
| `tests/test_docs.py` | 15 tests (fake-embedding test DB + 1 live) |

Run: `.\.venv\Scripts\python.exe ingest.py` once, then `python -m tools.docs` for a demo.

```
Employee: "hotel limit for international trips?" → travel_policy.txt ("...220 US dollars per night...")
Employee: "salary bands?"  → only public files (restricted file hidden)
HR:       "salary bands?"  → hr/restricted/salary_bands.txt
```

### Key concepts (simple words)

- **The tool finds, it doesn't answer** — like a librarian who brings the right
  pages but doesn't write your essay. *Why:* in Step 4 the agent mixes pages +
  currency + weather into one answer.

- **Ingest (prepare the library, once)** — read each file → cut into 80-word
  chunks (20 words overlap so sentences aren't cut in half) → turn each chunk
  into numbers (embedding) → store in ChromaDB with labels (source, department,
  public/restricted).

- **Hybrid search = two judges**
  - *Meaning judge* (embeddings): finds "lodging cost cap" even if the text says
    "hotel limit".
  - *Keyword judge* (BM25): great at exact words like "220" or "VPN".
  - *Merge (RRF)*: a chunk ranked high by BOTH judges wins.

- **Role filter inside the database query** — the database never even hands
  back forbidden chunks. *Why:* safer than fetching everything and hiding some
  afterwards — nothing secret can leak by mistake.

- **Fake embeddings in tests** — tests build a small database with simple
  word-count "embeddings" instead of the real model: fast, offline, repeatable.
  The most important test is a *security* test: an Employee must never see
  restricted files.

- **Reuse by copying** — Project 1's helpers were copied in, so this project
  runs on its own even if Project 1 moves or changes.

---

## Step 4: Multi-tool agent

### What we built
| File | Purpose |
|---|---|
| `agent.py` | tool registry + router + loop (MAX_STEPS=4) + trace; `python agent.py` demo, `--chat` to talk |
| `tools/currency.py` | new optional `amount` → `converted` (code does the maths) |
| `tests/test_agent.py` | fake-LLM tests of the loop + 2 live tests |

```
"Hi!"            → no tools
"100 USD in EUR" → get_exchange_rate
"Tokyo trip?"    → step 1: search_documents + get_weather (+ conversion POSTPONED)
                   step 2: get_exchange_rate(USD→JPY, amount 220) → ¥35,193.4
```

### Key concepts (simple words)
- **Agent** = an LLM in a loop: think → ask for a tool → our code runs it → LLM reads
  the result → repeat until it answers. The LLM *decides*, the code *does*.
- **Tool registry** = one list of tools (name, description, inputs → function).
  Add a tool = add one entry.
- **MAX_STEPS** = a safety limit so a confused model can't loop forever.
- **Trace** = a receipt of every tool call — lets you see *how* the agent decided.
- **Role set by code, not LLM** — otherwise "pretend I'm Admin" could leak secrets.
- **Let code do maths** — small LLMs miscalculate; the tool returns `converted`.
- **Fake LLM in tests** — a scripted pretend-LLM tests the loop exactly, every time.

### Lessons learned: when the prompt can't fix it, use structure
llama3.2 (small, 3B) made 2 mistakes the prompt couldn't fix:
1. It called tools even for "Hi!" → **router**: first ask "TOOLS or CHAT?" with no
   tools offered.
2. It called search + conversion *at the same time*, converting a guessed amount →
   **dependency rule**: our code postpones that conversion, then sends a short
   **nudge** ("now call get_exchange_rate with the exact amount"). Result: correct ¥35,193.4.
We measured each change by running the tricky questions several times.

---

## Step 5: Corrective RAG

### What we built
| File | Purpose |
|---|---|
| `grader.py` | `corrective_search(query, role)`: search → grade each chunk → rewrite + retry once → else "I don't know" |
| `agent.py` | docs tool now uses `corrective_search`; code forces "I don't know" if only empty searches happened; safer router |
| `tests/test_grader.py` | fake search/LLM tests + JSON validation tests + 1 live test |

```
"dress code?"  → 'company dress code' kept 0/3 → rewrite 'company dress code policy' kept 0/3
               → "I don't know based on the company documents."
"vacation days?" → kept 1/3 (hr_policy) → "You get 20 days of paid annual leave per year."
```

### Key concepts (simple words)
- **Corrective RAG** = don't trust search blindly. Search always returns *something*
  (the "least bad" pages). A grader checks: is this really about the question?
- **Grader** = the LLM as a quality inspector, one chunk at a time.
- **Structured output (JSON + Pydantic)** = we force the LLM to reply in a fixed shape
  `{"reason": "...", "relevant": true}` and Pydantic checks it. Invalid reply → treated
  as "not relevant" (safe side).
- **Rewrite + retry once** = if nothing fits, ask the LLM for a better search
  ("vacation" → "annual leave"), try again. Only once — no endless loops.
- **"I don't know" decided by code** = if every search came back empty, code replaces
  the LLM's answer. A wrong answer is worse than no answer.

### Lessons learned (tuning by measuring)
- Grader too strict → missed VPN. Grader too lenient → kept travel policy for "dress code"
  (matched only the word "policy"). Fix: make it name the *specific subject* of question
  and piece, and say synonyms count.
- **Reason first, and required**: an optional `reason` was skipped by the model → bad
  verdicts. Required reason = the model "thinks" before deciding.
- The router sent "dress code?" to small-talk mode, which then invented a policy.
  Fix: "when unsure → TOOLS", and chat mode may never state company facts. Measured: 18/18.

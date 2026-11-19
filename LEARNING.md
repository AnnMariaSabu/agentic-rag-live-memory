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

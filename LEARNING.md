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

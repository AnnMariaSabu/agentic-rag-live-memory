"""All project settings live here. Change a value once, it changes everywhere."""

# --- Ollama (local LLM) ---
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

# --- Free public APIs (no key needed) ---
FRANKFURTER_URL = "https://api.frankfurter.app"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# --- Network ---
TIMEOUT = 30  # seconds to wait for web APIs
LLM_TIMEOUT = 120  # Ollama's first call must load the model into memory (slow "cold start")

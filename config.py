"""All project settings live here. Change a value once, it changes everywhere."""

from pathlib import Path

# --- Folders ---
PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "data" / "documents"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# --- Document search (reused from Project 1) ---
COLLECTION_NAME = "company_docs"
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
CHUNK_SIZE = 80  # words per chunk
CHUNK_OVERLAP = 20  # words shared between neighbouring chunks
TOP_K = 3  # how many chunks the search returns
RRF_K = 60  # standard constant for merging the two rankings

# --- Long-term memory (Step 6) ---
MEMORY_DB = PROJECT_DIR / "memory.db"  # SQLite: the list of memories (source of truth)
MEMORY_COLLECTION = "memories"  # ChromaDB: memories searchable by meaning
MEMORY_TOP_K = 3  # memories given to the LLM per question

# --- Ollama (local LLM) ---
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

# --- Free public APIs (no key needed) ---
FRANKFURTER_URL = "https://api.frankfurter.app"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"  # city name -> coordinates

# --- Network ---
TIMEOUT = 30  # seconds to wait for web APIs
LLM_TIMEOUT = 120  # Ollama's first call must load the model into memory (slow "cold start")

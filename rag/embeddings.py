"""Turn texts into vectors with Ollama (nomic-embed-text).

nomic-embed-text works best with a label in front of the text:
"search_document: " when storing, "search_query: " when searching.
"""

import requests

import config


def embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": texts},
        timeout=config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["embeddings"]

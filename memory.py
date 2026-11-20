"""Long-term memory: facts about a user that last across chats.

SQLite  = the list of memories (easy to show, count and delete) — source of truth.
ChromaDB = the same memories as vectors, so we can find them BY MEANING
           ("Tokyo trip" can recall "I am vegetarian" ... if it's among the closest).

Every memory belongs to one user_id. Users never see each other's memories.
"""

import sqlite3
import uuid
from datetime import datetime, timezone

import chromadb
import requests

import config
from rag import embeddings


def _db():
    conn = sqlite3.connect(config.MEMORY_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                        text TEXT NOT NULL, created_at TEXT NOT NULL)""")
    return conn


def _collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(config.MEMORY_COLLECTION, metadata={"hnsw:space": "cosine"})


def save_memory(user_id: str, text: str) -> dict:
    """Remember one fact about the user. Never raises."""
    if not isinstance(text, str) or not text.strip():
        return {"error": "Nothing to remember: the memory text is empty."}
    text = text.strip()
    try:
        with _db() as conn:
            same = conn.execute("SELECT id FROM memories WHERE user_id = ? AND lower(text) = lower(?)",
                                (user_id, text)).fetchone()
            if same:
                return {"saved": False, "note": "Already remembered.", "text": text}
            vector = embeddings.embed(["search_document: " + text])[0]
            memory_id = str(uuid.uuid4())
            created = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _collection().add(ids=[memory_id], documents=[text], embeddings=[vector],
                              metadatas=[{"user_id": user_id}])
            conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?)", (memory_id, user_id, text, created))
        return {"saved": True, "id": memory_id, "text": text}
    except requests.RequestException:
        return {"error": "Could not reach the embedding model, so nothing was saved."}


def recall_memory(user_id: str, query: str, k: int = config.MEMORY_TOP_K) -> list[str]:
    """The user's k memories closest in meaning to the query. Never raises (returns [])."""
    try:
        collection = _collection()
        count = len(collection.get(where={"user_id": user_id})["ids"])
        if count == 0:
            return []
        vector = embeddings.embed(["search_query: " + query])[0]
        result = collection.query(query_embeddings=[vector], n_results=min(k, count),
                                  where={"user_id": user_id})  # only THIS user's memories
        return result["documents"][0]
    except requests.RequestException:
        return []


def list_memories(user_id: str) -> list[dict]:
    with _db() as conn:
        rows = conn.execute("SELECT id, text, created_at FROM memories WHERE user_id = ? "
                            "ORDER BY created_at", (user_id,)).fetchall()
    return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]


def forget_me(user_id: str) -> dict:
    """Delete EVERYTHING remembered about this user, in both stores."""
    with _db() as conn:
        deleted = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,)).rowcount
    _collection().delete(where={"user_id": user_id})
    return {"deleted": deleted}


if __name__ == "__main__":
    # Demo: python memory.py
    user = "demo-user"
    forget_me(user)
    print(save_memory(user, "I am vegetarian."))
    print(save_memory(user, "My home office is in Berlin."))
    print(save_memory(user, "I am vegetarian."))  # duplicate
    print("recall for 'dinner on my trip':", recall_memory(user, "dinner on my trip", k=1))
    print("other user sees:", recall_memory("someone-else", "dinner"))
    print("list:", [m["text"] for m in list_memories(user)])
    print(forget_me(user), "-> left:", list_memories(user))

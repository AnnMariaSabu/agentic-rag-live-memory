"""Docs tool: hybrid search over company documents, filtered by role.

Returns the best matching chunks (text + source). It does NOT write the answer —
the agent (Step 4) combines chunks with other tools' results into one reply.
Like the other tools, it NEVER raises: failures return {"error": "..."}.
"""

import re

import chromadb
import requests
from rank_bm25 import BM25Okapi

import config
from rag import embeddings
from rag.permissions import ROLES, search_filter

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "be", "of", "to", "in", "on",
    "at", "for", "and", "or", "with", "about", "do", "does", "did", "i", "my",
    "me", "you", "your", "we", "what", "who", "which", "when", "where", "how",
    "why", "can", "could", "should", "would", "will", "this", "that", "there",
    "many", "much", "get",
}


def tokenize(text):
    """'What is the Hotel limit?' -> ['hotel', 'limit'] (for keyword search)."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


def _get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def _hybrid_search(collection, query, where, k):
    """Meaning search + keyword search, merged with Reciprocal Rank Fusion."""
    data = collection.get(where=where)  # role filter applied IN the database
    if not data["ids"]:
        return []

    chunks = {
        cid: {"text": text, "source": meta["source"], "distance": None,
              "keyword_score": 0.0, "rrf": 0.0}
        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    }

    # 1. Meaning search: rank every allowed chunk by vector distance
    query_vector = embeddings.embed(["search_query: " + query])[0]
    results = collection.query(
        query_embeddings=[query_vector], n_results=len(chunks), where=where
    )
    for rank, (cid, dist) in enumerate(zip(results["ids"][0], results["distances"][0]), 1):
        chunks[cid]["distance"] = round(dist, 4)
        chunks[cid]["rrf"] += 1 / (config.RRF_K + rank)

    # 2. Keyword search: rank chunks by BM25 score
    ids = list(chunks)
    bm25 = BM25Okapi([tokenize(chunks[cid]["text"]) for cid in ids])
    scores = bm25.get_scores(tokenize(query))
    keyword_ranking = sorted(zip(scores, ids), reverse=True)
    for rank, (score, cid) in enumerate(keyword_ranking, 1):
        if score <= 0:
            break
        chunks[cid]["keyword_score"] = round(float(score), 4)
        chunks[cid]["rrf"] += 1 / (config.RRF_K + rank)

    # 3. Merge: chunks ranked high by BOTH searches win
    best = sorted(chunks.values(), key=lambda c: c["rrf"], reverse=True)[:k]
    for c in best:
        c["rrf"] = round(c["rrf"], 5)
    return best


def search_documents(query: str, role: str, k: int = config.TOP_K) -> dict:
    """Find the k best document chunks this role is allowed to see."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "Please give a question to search for."}
    if role not in ROLES:
        return {"error": f"Unknown role '{role}'. Use one of: {', '.join(ROLES)}."}
    query = query.strip()

    try:
        collection = _get_collection()
    except Exception:  # missing folder/collection
        return {"error": "The document database is not built yet. Run ingest.py first."}

    try:
        results = _hybrid_search(collection, query, search_filter(role), k)
        return {"query": query, "role": role, "results": results}
    except requests.Timeout:
        return {"error": "The embedding model took too long to answer. Please try again."}
    except requests.ConnectionError:
        return {"error": "Could not reach the embedding model. Is Ollama running?"}
    except requests.HTTPError as e:
        return {"error": f"The embedding model returned an error ({e})."}


if __name__ == "__main__":
    # Quick demo: python -m tools.docs   (run ingest.py first)
    for q, role in [
        ("What is the hotel limit for international trips?", "Employee"),
        ("What are the salary bands?", "Employee"),
        ("What are the salary bands?", "HR"),
    ]:
        print(f"\n=== {role}: {q}")
        out = search_documents(q, role)
        if "error" in out:
            print("ERROR:", out["error"])
            continue
        for i, c in enumerate(out["results"], 1):
            print(f"  {i}. [{c['source']}] {c['text'][:90]}...")

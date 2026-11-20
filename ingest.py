"""Build the document search database: read -> chunk -> embed -> save in ChromaDB.

Run:  .\\.venv\\Scripts\\python.exe ingest.py             (only changed files)
      .\\.venv\\Scripts\\python.exe ingest.py --rebuild   (everything from scratch)

Step 7: every file gets a fingerprint (SHA-256 hash). If the fingerprint is the
same as last time, the file is skipped — no slow re-embedding. Changed files are
re-done, deleted files are removed from the database.
"""

import hashlib
import sys

import chromadb

import config
from rag import embeddings
from rag.chunker import split_text
from rag.loaders import load_text


def file_hash(path) -> str:
    """Fingerprint of the file's bytes AND the chunk settings:
    changing CHUNK_SIZE in config.py also counts as 'changed'."""
    h = hashlib.sha256(path.read_bytes())
    h.update(f"{config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}".encode())
    return h.hexdigest()


def get_metadata(path):
    """Department + access level come from the folder: hr/restricted/x.txt -> hr, restricted."""
    parts = path.relative_to(config.DOCS_DIR).parts
    return {
        "source": path.relative_to(config.DOCS_DIR).as_posix(),
        "department": parts[0],
        "access": "restricted" if "restricted" in parts else "public",
        "file_hash": file_hash(path),
    }


def stored_hashes(collection) -> dict:
    """{source: file_hash} for every file already in the database."""
    data = collection.get(include=["metadatas"])
    return {m["source"]: m.get("file_hash") for m in data["metadatas"]}


def run_ingest(rebuild: bool = False) -> dict:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if rebuild and config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    known = stored_hashes(collection)
    counts = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0}
    seen = set()

    for path in sorted(config.DOCS_DIR.rglob("*")):
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        meta = get_metadata(path)
        source = meta["source"]
        seen.add(source)

        if known.get(source) == meta["file_hash"]:  # same fingerprint -> nothing to do
            counts["unchanged"] += 1
            print(f"{source:45} unchanged, skipped")
            continue

        status = "updated" if source in known else "added"
        if source in known:
            collection.delete(where={"source": source})  # remove the old chunks first
        chunks = split_text(load_text(path), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if chunks:
            vectors = embeddings.embed(["search_document: " + c for c in chunks])
            collection.add(
                ids=[f"{source}::{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=vectors,
                metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
            )
        counts[status] += 1
        print(f"{source:45} {status}: {len(chunks)} chunks ({meta['access']})")

    for source in known:
        if source not in seen:  # the file was deleted from the folder
            collection.delete(where={"source": source})
            counts["removed"] += 1
            print(f"{source:45} removed (file no longer exists)")

    print(f"Done: {counts}. Total chunks in database: {collection.count()}")
    return counts


if __name__ == "__main__":
    run_ingest(rebuild="--rebuild" in sys.argv)

"""Build the document search database: read -> chunk -> embed -> save in ChromaDB.

Run:  .\\.venv\\Scripts\\python.exe ingest.py
Simple version: rebuilds everything each time.
(Step 7 will skip unchanged files using a file hash.)
"""

import chromadb

import config
from rag import embeddings
from rag.chunker import split_text
from rag.loaders import load_text


def get_metadata(path):
    """Department + access level come from the folder: hr/restricted/x.txt -> hr, restricted."""
    parts = path.relative_to(config.DOCS_DIR).parts
    return {
        "source": path.relative_to(config.DOCS_DIR).as_posix(),
        "department": parts[0],
        "access": "restricted" if "restricted" in parts else "public",
    }


def run_ingest():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)  # start fresh
    collection = client.create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for path in sorted(config.DOCS_DIR.rglob("*")):
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        meta = get_metadata(path)
        chunks = split_text(load_text(path), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            continue
        vectors = embeddings.embed(["search_document: " + c for c in chunks])
        collection.add(
            ids=[f"{meta['source']}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
        )
        print(f"{meta['source']:45} -> {len(chunks)} chunks ({meta['access']})")

    print("Total chunks in database:", collection.count())


if __name__ == "__main__":
    run_ingest()

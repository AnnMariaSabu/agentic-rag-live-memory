"""Shared test helpers (pytest loads this file automatically)."""

import hashlib
import re

import pytest

import config


def fake_embed(texts):
    """Each word adds 1 to one of 64 slots -> similar words, similar vectors.
    Fast, offline stand-in for the real embedding model."""
    vectors = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        vectors.append(v)
    return vectors


@pytest.fixture(autouse=True)
def isolated_memory(request, tmp_path, monkeypatch):
    """Every non-live test gets its own empty memory store, and the agent's automatic
    memory recall is switched off (memory has its own tests)."""
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setattr(config, "MEMORY_DB", tmp_path / "memory.db")
    import agent
    monkeypatch.setattr(agent, "recall_memory", lambda user_id, query: [])

"""Tests for the docs tool and its helpers.

Most tests build a small test database from the real documents, but with
FAKE embeddings (word-count vectors) so they are fast and run offline.
The 'live' test uses real Ollama embeddings.
"""

import hashlib
import re
from unittest.mock import patch

import pytest
import requests

import config
import ingest
from rag.chunker import split_text
from rag.permissions import search_filter
from tools.docs import search_documents


def fake_embed(texts):
    """Each word adds 1 to one of 64 slots -> similar words, similar vectors."""
    vectors = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        vectors.append(v)
    return vectors


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Build a fresh test database in a temp folder with fake embeddings."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    with patch("rag.embeddings.embed", side_effect=fake_embed):
        ingest.run_ingest()
        yield


def sources(result):
    return [c["source"] for c in result["results"]]


# --- helpers reused from Project 1 ---

def test_chunker_overlap():
    words = " ".join(str(i) for i in range(10))
    assert split_text(words, size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("a b c", size=3, overlap=3)


def test_permission_filters():
    assert search_filter("Employee") == {"access": "public"}
    assert search_filter("Admin") is None
    with pytest.raises(ValueError):
        search_filter("Boss")


# --- the tool ---

def test_finds_travel_policy(test_db):
    result = search_documents("hotel limit international trips", "Employee")
    assert sources(result)[0] == "travel/travel_policy.txt"
    assert len(result["results"]) == config.TOP_K


def test_employee_never_sees_restricted(test_db):
    result = search_documents("salary bands", "Employee")
    assert not any("restricted" in s for s in sources(result))


def test_hr_sees_hr_restricted(test_db):
    result = search_documents("salary bands", "HR")
    assert "hr/restricted/salary_bands.txt" in sources(result)


def test_hr_never_sees_it_restricted(test_db):
    result = search_documents("server admin guide root password", "HR")
    assert "it/restricted/server_admin_guide.txt" not in sources(result)


def test_admin_sees_everything(test_db):
    result = search_documents("server admin guide", "Admin")
    assert "it/restricted/server_admin_guide.txt" in sources(result)


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_query(bad):
    assert "give a question" in search_documents(bad, "Employee")["error"]


def test_unknown_role():
    assert "Unknown role" in search_documents("hotel", "Boss")["error"]


def test_database_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "empty")
    assert "Run ingest.py first" in search_documents("hotel", "Employee")["error"]


def test_ollama_down(test_db):
    with patch("rag.embeddings.embed", side_effect=requests.ConnectionError):
        assert "Is Ollama running" in search_documents("hotel", "Employee")["error"]


@pytest.mark.live
def test_live_real_embeddings(tmp_path, monkeypatch):
    """Real Ollama embeddings. Skip offline with: pytest -m "not live" """
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    ingest.run_ingest()
    result = search_documents("What is the hotel limit for international trips?", "Employee")
    assert "error" not in result, result
    assert "travel/travel_policy.txt" in sources(result)[:2]

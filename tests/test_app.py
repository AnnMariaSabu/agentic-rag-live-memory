"""Tests for the Streamlit UI, using Streamlit's built-in AppTest (no browser needed).
The agent is faked, so these tests are fast and offline."""

import sqlite3
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

import agent
import config
import memory

FAKE_RESULT = {
    "answer": "Your limit is ¥34,900.8 per night.",
    "trace": [
        {"step": 1, "tool": "search_documents", "arguments": {"query": "hotel limit", "role": "Employee"},
         "result": {"results": [{"source": "travel/travel_policy.txt", "text": "The hotel limit is 220..."}],
                    "attempts": [{"query": "hotel limit", "kept": 1, "checked": 3}]}},
        {"step": 2, "tool": "get_exchange_rate",
         "arguments": {"from_currency": "USD", "to_currency": "JPY", "amount": 220},
         "result": {"from": "USD", "to": "JPY", "rate": 158.64, "amount": 220.0, "converted": 34900.8}},
    ],
    "freshness": ["Exchange rate USD→JPY: ECB rate of 2026-11-20, fetched 2026-11-20 17:30 UTC (just fetched)"],
    "memories_used": ["The user is vegetarian."],
}


@pytest.fixture
def fake_agent(monkeypatch):
    run = MagicMock(return_value=FAKE_RESULT)
    monkeypatch.setattr(agent, "run_agent", run)
    return run


def app():
    return AppTest.from_file(str(config.PROJECT_DIR / "app.py"), default_timeout=30).run()


def test_page_loads(fake_agent):
    at = app()
    assert not at.exception
    assert at.title[0].value == "🧭 Acme Assistant"
    assert at.selectbox[0].options == ["Employee", "HR", "IT", "Admin"]


def test_chat_shows_answer_tools_and_freshness(fake_agent):
    at = app()
    at.selectbox[0].set_value("HR")
    at.chat_input[0].set_value("Hotel limit in yen?").run()
    assert not at.exception
    fake_agent.assert_called_once_with("Hotel limit in yen?", "HR", user_id=agent.DEFAULT_USER)
    texts = [m.value for m in at.markdown]
    assert "Your limit is ¥34,900.8 per night." in texts
    assert at.expander[0].label == "🔧 Tools used (2)"
    captions = [c.value for c in at.caption]
    assert any("🕒 Exchange rate USD→JPY" in c for c in captions)
    assert any("📄 travel/travel_policy.txt" in c for c in captions)


def test_memory_sidebar_and_forget_me(fake_agent):
    # Real memory functions on the temp DB from conftest. Insert directly into SQLite,
    # so no embedding model is needed for this test.
    with sqlite3.connect(config.MEMORY_DB) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                     "text TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO memories VALUES ('1', ?, 'The user is vegetarian.', '2026-11-20')",
                     (agent.DEFAULT_USER,))
    at = app()
    assert "- The user is vegetarian." in [m.value for m in at.sidebar.markdown]

    forget = [b for b in at.sidebar.button if "Forget me" in b.label][0]
    assert forget.disabled  # must tick the confirmation box first
    at.sidebar.checkbox[0].check().run()
    [b for b in at.sidebar.button if "Forget me" in b.label][0].click().run()
    assert memory.list_memories(agent.DEFAULT_USER) == []

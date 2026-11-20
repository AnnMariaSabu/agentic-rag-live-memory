"""Tests for memory.py and how the agent uses it (fake embeddings, temp folders)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import agent
import config
import memory
from tests.conftest import fake_embed


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "MEMORY_DB", tmp_path / "memory.db")
    with patch("rag.embeddings.embed", side_effect=fake_embed):
        yield memory


def test_save_and_recall(mem):
    mem.save_memory("ann", "The user is vegetarian.")
    mem.save_memory("ann", "The user lives in Berlin.")
    assert mem.recall_memory("ann", "is the user vegetarian", k=1) == ["The user is vegetarian."]
    assert len(mem.list_memories("ann")) == 2


def test_users_never_see_each_other(mem):
    mem.save_memory("ann", "The user is vegetarian.")
    assert mem.recall_memory("bob", "vegetarian") == []
    assert mem.list_memories("bob") == []


def test_duplicate_not_saved_twice(mem):
    mem.save_memory("ann", "The user is vegetarian.")
    again = mem.save_memory("ann", "the user is VEGETARIAN.")
    assert again["saved"] is False and len(mem.list_memories("ann")) == 1


def test_empty_text(mem):
    assert "empty" in mem.save_memory("ann", "  ")["error"]


def test_forget_me_deletes_both_stores(mem):
    mem.save_memory("ann", "The user is vegetarian.")
    mem.save_memory("bob", "The user likes tea.")
    assert mem.forget_me("ann") == {"deleted": 1}
    assert mem.list_memories("ann") == [] and mem.recall_memory("ann", "vegetarian") == []
    assert mem.list_memories("bob") != []  # other users untouched


def test_embedding_down_saves_nothing(mem):
    with patch("rag.embeddings.embed", side_effect=requests.ConnectionError):
        assert "nothing was saved" in mem.save_memory("ann", "x is y")["error"]
    assert mem.list_memories("ann") == []


def test_memory_persists_across_restarts(mem):
    mem.save_memory("ann", "The user is vegetarian.")
    # A "restart" = new connections; data lives in files, not in Python variables
    assert [m["text"] for m in memory.list_memories("ann")] == ["The user is vegetarian."]


# --- agent + memory ---

def tool_call(name, **arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def test_agent_puts_memories_in_prompt(monkeypatch):
    monkeypatch.setattr(agent, "recall_memory", lambda user_id, query: ["The user is vegetarian."])
    monkeypatch.setattr(agent, "needs_tools", lambda q: True)
    with patch("agent.chat", return_value={"role": "assistant", "content": "ok"}) as fake_llm:
        result = agent.run_agent("Dinner ideas for my trip?", "Employee", user_id="ann")
    system_prompt = fake_llm.call_args.args[0][0]["content"]
    assert "The user is vegetarian." in system_prompt
    assert result["memories_used"] == ["The user is vegetarian."]


def test_save_memory_tool_gets_user_id_from_code(monkeypatch):
    monkeypatch.setattr(agent, "needs_tools", lambda q: True)
    fake_save = MagicMock(return_value={"saved": True, "text": "The user is vegetarian."})
    monkeypatch.setitem(agent.TOOLS["save_memory"], "function", fake_save)
    script = [tool_call("save_memory", text="The user is vegetarian.", user_id="someone-else"),
              {"role": "assistant", "content": "Noted!"}]
    with patch("agent.chat", side_effect=script):
        agent.run_agent("Remember I'm vegetarian", "Employee", user_id="ann")
    fake_save.assert_called_once_with(text="The user is vegetarian.", user_id="ann")


def test_forget_me_is_not_an_llm_tool():
    assert "forget_me" not in agent.TOOLS


@pytest.mark.live
def test_live_remember_then_use(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "MEMORY_DB", tmp_path / "memory.db")
    agent.run_agent("Please remember that I am vegetarian.", "Employee", user_id="live-test")
    assert any("vegetarian" in m["text"].lower() for m in memory.list_memories("live-test"))
    result = agent.run_agent("What's my meal allowance per day on an international trip?",
                             "Employee", user_id="live-test")
    assert any("vegetarian" in m.lower() for m in result["memories_used"])

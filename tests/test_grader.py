"""Tests for grader.py (corrective RAG). The search and the LLM are faked."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import agent
import grader
from grader import Grade, Rewrite, ask_json, corrective_search


def chunk(text, source="x.txt"):
    return {"text": text, "source": source}


def found(*chunks):
    return {"query": "q", "role": "Employee", "results": list(chunks)}


def fake_grader(relevant_words):
    """Grade a chunk relevant if it contains one of the given words."""
    def grade(question, c):
        return Grade(relevant=any(w in c["text"] for w in relevant_words), reason="test")
    return grade


def test_relevant_on_first_try_no_rewrite():
    with patch("grader.search_documents", return_value=found(chunk("hotel 220"), chunk("parking"))), \
         patch("grader.grade_chunk", side_effect=fake_grader(["hotel"])), \
         patch("grader.ask_json") as rewrite:
        out = corrective_search("hotel limit", "Employee")
    assert [c["text"] for c in out["results"]] == ["hotel 220"]  # irrelevant chunk dropped
    assert out["attempts"] == [{"query": "hotel limit", "kept": 1, "checked": 2}]
    rewrite.assert_not_called()


def test_rewrite_and_retry_once():
    searches = [found(chunk("parking")), found(chunk("annual leave 20 days"))]
    with patch("grader.search_documents", side_effect=searches) as search, \
         patch("grader.grade_chunk", side_effect=fake_grader(["leave"])), \
         patch("grader.ask_json", return_value=Rewrite(query="annual leave days")):
        out = corrective_search("vacation", "Employee")
    assert search.call_args_list[1].args[0] == "annual leave days"
    assert len(out["results"]) == 1 and len(out["attempts"]) == 2


def test_nothing_twice_means_i_dont_know():
    with patch("grader.search_documents", return_value=found(chunk("parking"))) as search, \
         patch("grader.grade_chunk", side_effect=fake_grader([])), \
         patch("grader.ask_json", return_value=Rewrite(query="office dress rules")):
        out = corrective_search("dress code", "Employee")
    assert out["results"] == [] and grader.NO_ANSWER in out["note"]
    assert search.call_count == 2  # only ONE retry, never more


def test_search_error_passes_through():
    with patch("grader.search_documents", return_value={"error": "Run ingest.py first."}):
        assert corrective_search("x", "Employee") == {"error": "Run ingest.py first."}


def test_role_is_passed_to_search():
    with patch("grader.search_documents", return_value=found()) as search, \
         patch("grader.ask_json", return_value=None):
        corrective_search("salary", "HR")
    assert search.call_args.args[1] == "HR"


def test_grading_model_down():
    with patch("grader.search_documents", return_value=found(chunk("a"))), \
         patch("grader.grade_chunk", side_effect=requests.ConnectionError):
        assert "not reachable" in corrective_search("x", "Employee")["error"]


# --- JSON + Pydantic checking of LLM replies ---

def llm_reply(content):
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    return resp


def test_ask_json_valid():
    with patch("grader.requests.post", return_value=llm_reply('{"reason": "about hotels", "relevant": true}')) as post:
        g = ask_json("sys", "user", Grade)
    assert g.relevant is True
    assert post.call_args.kwargs["json"]["format"] == Grade.model_json_schema()
    assert "timeout" in post.call_args.kwargs


@pytest.mark.parametrize("bad", ["not json", '{"reason": "x"}', '{"relevant": "maybe"}'])
def test_ask_json_invalid_returns_none(bad):
    with patch("grader.requests.post", return_value=llm_reply(bad)):
        assert ask_json("sys", "user", Grade) is None


def test_invalid_grade_counts_as_not_relevant():
    with patch("grader.ask_json", return_value=None):
        assert grader.grade_chunk("q", chunk("t")).relevant is False


# --- agent: code forces "I don't know" when only empty searches happened ---

def test_agent_says_i_dont_know_when_nothing_found(monkeypatch):
    monkeypatch.setattr(agent, "needs_tools", lambda q: True)
    monkeypatch.setitem(agent.TOOLS["search_documents"], "function",
                        MagicMock(return_value={"results": [], "attempts": []}))
    script = [{"role": "assistant", "content": "", "tool_calls": [
                  {"function": {"name": "search_documents", "arguments": {"query": "dress code"}}}]},
              {"role": "assistant", "content": "Business casual is required."}]  # made up!
    with patch("agent.chat", side_effect=script):
        result = agent.run_agent("What's the dress code?", "Employee")
    assert result["answer"] == grader.NO_ANSWER


@pytest.mark.live
def test_live_unknown_topic():
    result = agent.run_agent("What is the company dress code?", "Employee")
    assert result["answer"] == grader.NO_ANSWER, result

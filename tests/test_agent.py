"""Tests for agent.py.

Mocked tests use a FAKE LLM that follows a script (e.g. "first ask for the
weather tool, then answer"), and fake tools. This tests the loop logic exactly.
'live' tests use the real llama3.2 and real tools.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

import agent


def tool_call(name, **arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def final(text):
    return {"role": "assistant", "content": text}


@pytest.fixture
def fake_tools(monkeypatch):
    """Replace each real tool with a fake that records its inputs.
    Also make the router say 'needs tools' (router has its own tests)."""
    monkeypatch.setattr(agent, "needs_tools", lambda q: True)
    fakes = {}
    for name in agent.TOOLS:
        fake = MagicMock(return_value={"fake": name})
        monkeypatch.setitem(agent.TOOLS[name], "function", fake)
        fakes[name] = fake
    return fakes


def test_no_tool_needed(fake_tools):
    with patch("agent.chat", side_effect=[final("Hello!")]):
        result = agent.run_agent("Hi", "Employee")
    assert result["answer"] == "Hello!"
    assert result["trace"] == [] and result["steps"] == 1


def test_runs_requested_tool_and_records_trace(fake_tools):
    script = [tool_call("get_weather", city="Tokyo"), final("Tokyo is sunny.")]
    with patch("agent.chat", side_effect=script) as fake_llm:
        result = agent.run_agent("Weather in Tokyo?", "Employee")
    fake_tools["get_weather"].assert_called_once_with(city="Tokyo")
    assert result["answer"] == "Tokyo is sunny."
    assert result["trace"] == [{"step": 1, "tool": "get_weather",
                                "arguments": {"city": "Tokyo"}, "result": {"fake": "get_weather"}}]
    # The tool result was handed back to the LLM: system, user, assistant(tool call), tool, ...
    messages = fake_llm.call_args_list[1].args[0]  # same list object the agent kept growing
    assert messages[3]["role"] == "tool" and messages[3]["tool_name"] == "get_weather"


def test_three_tools_in_one_question(fake_tools):
    script = [
        tool_call("search_documents", query="hotel limit international"),
        tool_call("get_exchange_rate", from_currency="USD", to_currency="JPY", amount=220),
        tool_call("get_weather", city="Tokyo"),
        final("Limit ¥35,193, Tokyo 10 °C."),
    ]
    with patch("agent.chat", side_effect=script):
        result = agent.run_agent("Tokyo trip?", "Employee")
    assert [t["tool"] for t in result["trace"]] == ["search_documents", "get_exchange_rate", "get_weather"]
    assert result["stopped_early"] is False


def test_role_is_set_by_code_not_llm(fake_tools):
    # The LLM tries to sneak in role="Admin" — it must be ignored
    script = [tool_call("search_documents", query="salary bands", role="Admin"), final("...")]
    with patch("agent.chat", side_effect=script):
        agent.run_agent("salary bands?", "Employee")
    fake_tools["search_documents"].assert_called_once_with(query="salary bands", role="Employee")


def test_stops_after_max_steps(fake_tools):
    # The LLM keeps asking for tools forever; the last call (tools off) answers
    looping = [tool_call("get_weather", city="Paris")] * agent.MAX_STEPS
    with patch("agent.chat", side_effect=looping + [final("Paris is cloudy.")]) as fake_llm:
        result = agent.run_agent("Weather?", "Employee")
    assert result["stopped_early"] is True
    assert len(result["trace"]) == agent.MAX_STEPS
    assert fake_llm.call_args_list[-1].kwargs == {"use_tools": False}
    assert result["answer"] == "Paris is cloudy."


def test_unknown_tool_gives_polite_error(fake_tools):
    script = [tool_call("get_stock_price", symbol="AAPL"), final("I can't do that.")]
    with patch("agent.chat", side_effect=script):
        result = agent.run_agent("Apple stock?", "Employee")
    assert "Unknown tool" in result["trace"][0]["result"]["error"]


def test_missing_input_gives_polite_error(fake_tools):
    script = [tool_call("get_weather"), final("Which city?")]
    with patch("agent.chat", side_effect=script):
        result = agent.run_agent("Weather?", "Employee")
    assert "Missing input" in result["trace"][0]["result"]["error"]
    fake_tools["get_weather"].assert_not_called()


def test_arguments_as_json_string(fake_tools):
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}]}
    with patch("agent.chat", side_effect=[call, final("Cold.")]):
        agent.run_agent("Oslo?", "Employee")
    fake_tools["get_weather"].assert_called_once_with(city="Oslo")


def test_crashing_tool_does_not_crash_agent(fake_tools):
    fake_tools["get_weather"].side_effect = RuntimeError("boom")
    with patch("agent.chat", side_effect=[tool_call("get_weather", city="X"), final("Sorry.")]):
        result = agent.run_agent("X?", "Employee")
    assert "failed" in result["trace"][0]["result"]["error"]


def test_router_chat_offers_no_tools(fake_tools, monkeypatch):
    monkeypatch.setattr(agent, "needs_tools", lambda q: False)
    with patch("agent.chat", side_effect=[final("Hello!")]) as fake_llm:
        result = agent.run_agent("Hi!", "Employee")
    assert result["answer"] == "Hello!" and result["trace"] == []
    assert fake_llm.call_args.kwargs == {"use_tools": False}


@pytest.mark.parametrize("reply, expected", [("CHAT", False), ("chat", False), ("TOOLS", True)])
def test_router_reads_llm_reply(reply, expected):
    with patch("agent.chat", return_value=final(reply)):
        assert agent.needs_tools("anything") is expected


def test_conversion_waits_for_policy_search(fake_tools):
    # Step 1: model asks for search AND conversion with a guessed amount (same step)
    step1 = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "search_documents", "arguments": {"query": "hotel limit"}}},
        {"function": {"name": "get_exchange_rate",
                      "arguments": {"from_currency": "USD", "to_currency": "JPY", "amount": 0}}},
    ]}
    step2 = tool_call("get_exchange_rate", from_currency="USD", to_currency="JPY", amount=220)
    with patch("agent.chat", side_effect=[step1, step2, final("¥35,193")]) as fake_llm:
        result = agent.run_agent("Hotel limit in yen?", "Employee")
    # The guessed conversion was postponed, NOT run; only the real one ran
    assert "postponed" in result["trace"][1]["result"]
    fake_tools["get_exchange_rate"].assert_called_once_with(
        from_currency="USD", to_currency="JPY", amount=220)
    # A nudge was sent to the model after the postponed step
    messages = fake_llm.call_args_list[1].args[0]
    assert any(m["role"] == "user" and m["content"] == agent.NUDGE for m in messages)


def test_tool_error_is_not_turned_into_no_documents(fake_tools):
    # Found by the evaluation (fx-04): a currency error answered with the "no documents" phrase
    fake_tools["get_exchange_rate"].return_value = {"error": "Unknown currency code 'USD' or 'XYZ'."}
    script = [tool_call("get_exchange_rate", from_currency="USD", to_currency="XYZ", amount=100),
              final("I don't know based on the company documents.")]
    with patch("agent.chat", side_effect=script):
        result = agent.run_agent("100 USD to XYZ?", "Employee")
    assert result["answer"] == "Sorry, I couldn't get that: Unknown currency code 'USD' or 'XYZ'."


def test_described_but_not_called_tool_gets_nudged(fake_tools):
    # Found by the evaluation (mix-03): the model WROTE "I'll call get_exchange_rate" instead of calling it
    script = [tool_call("search_documents", query="hotel limit international"),
              final("The limit is 220 USD. I'll need to call get_exchange_rate with 220."),
              tool_call("get_exchange_rate", from_currency="USD", to_currency="GBP", amount=220),
              final("That is £176 per night.")]
    with patch("agent.chat", side_effect=script) as fake_llm:
        result = agent.run_agent("Hotel limit in pounds?", "Employee")
    assert result["answer"] == "That is £176 per night."
    fake_tools["get_exchange_rate"].assert_called_once()
    messages = fake_llm.call_args_list[-1].args[0]
    assert any(m["role"] == "user" and "Call get_exchange_rate now" in m["content"] for m in messages)


def test_llm_down():
    with patch("agent.chat", side_effect=requests.ConnectionError):
        result = agent.run_agent("Hi", "Employee")
    assert "not reachable" in result["answer"]


# --- live: real llama3.2 + real tools (needs Ollama, internet, ingest.py run) ---

@pytest.mark.live
def test_live_tokyo_uses_three_tools():
    result = agent.run_agent(
        "I'm flying to Tokyo next week. What's my hotel limit per night in yen, "
        "and what's the weather there now?", "Employee")
    used = {t["tool"] for t in result["trace"]}
    assert used == {"search_documents", "get_exchange_rate", "get_weather"}, result["trace"]
    conversions = [t for t in result["trace"]
                   if t["tool"] == "get_exchange_rate" and "converted" in t["result"]]
    assert conversions and conversions[-1]["result"]["amount"] == 220.0


@pytest.mark.live
def test_live_greeting_uses_no_tool():
    assert agent.run_agent("Hi!", "Employee")["trace"] == []

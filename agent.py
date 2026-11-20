"""Multi-tool agent: the LLM decides which tools to call, our code runs them.

Loop (max MAX_STEPS rounds):  ask LLM -> it requests tools -> we run them ->
give results back -> ask again ... until it answers without tools.

Run demo:  .\\.venv\\Scripts\\python.exe agent.py
Chat:      .\\.venv\\Scripts\\python.exe agent.py --chat
"""

import json
import sys

import requests

import config
from tools.currency import get_exchange_rate
from tools.docs import search_documents
from tools.weather import get_weather

MAX_STEPS = 4

SYSTEM_PROMPT = """You are Acme Corp's assistant for employees.

You have three tools:
- search_documents: Acme's company policies (HR, IT, travel: hotel and meal limits, leave, benefits...).
- get_exchange_rate: live currency rates. Pass "amount" to convert money; use the "converted" value it returns.
- get_weather: current weather in a city.

Rules:
1. Never guess policies, exchange rates or weather. Use a tool for each fact you need.
2. If the question needs several facts, call ALL the tools needed.
3. Policy money amounts are in US dollars (USD). To convert a policy amount, call
   search_documents FIRST and wait for its result. Only then call get_exchange_rate
   with from_currency "USD" and the exact amount you found. Never invent an amount.
4. Answer only from the tool results. Copy numbers exactly (use "converted" for money).
   If a tool returns an "error", tell the user briefly.
5. Answer in one to three short sentences."""

CHAT_PROMPT = """You are the friendly helpdesk assistant for employees of Acme Corp (a normal company).
You can answer questions about company policies, convert currencies and check the weather.
Reply in one or two short sentences."""

NUDGE = ("You now have the policy search result. Call get_exchange_rate now with "
         "from_currency \"USD\", the currency the user asked for, and the exact USD amount "
         "from the policy text as \"amount\".")

# Router: small models call a tool for almost everything (even "Hi!") when tools are
# offered. So we first ask, with NO tools offered, whether tools are needed at all.
ROUTER_PROMPT = """Decide if the user's message needs facts from Acme's company policies,
live exchange rates or live weather.
Reply with exactly one word:
TOOLS - it asks about a policy, money/currency, prices, limits, or weather
CHAT  - greeting, thanks, small talk, or a question about you

Examples:
"Hi!" -> CHAT
"Thanks, that helps" -> CHAT
"How many vacation days do I get?" -> TOOLS
"100 USD in EUR?" -> TOOLS
"Is it raining in London?" -> TOOLS"""

# --- Tool registry: ONE place that lists every tool -------------------------
# "schema" is what the LLM sees; "function" is what our code runs.
# "needs_role": our code adds the user's role — the LLM never chooses it (security).
TOOLS = {
    "search_documents": {
        "function": search_documents,
        "needs_role": True,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search Acme Corp's company policy documents (HR, IT, travel).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "A short search about ONE topic, e.g. 'hotel limit international'"},
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "get_exchange_rate": {
        "function": get_exchange_rate,
        "needs_role": False,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_exchange_rate",
                "description": "Live exchange rate between two currencies; optionally converts an amount.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {"type": "string", "description": "3-letter code, e.g. USD"},
                        "to_currency": {"type": "string", "description": "3-letter code, e.g. JPY"},
                        "amount": {"type": "number", "description": "Optional amount to convert, e.g. 220"},
                    },
                    "required": ["from_currency", "to_currency"],
                },
            },
        },
    },
    "get_weather": {
        "function": get_weather,
        "needs_role": False,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name, e.g. Tokyo"},
                    },
                    "required": ["city"],
                },
            },
        },
    },
}


def chat(messages, use_tools=True):
    """One call to the local LLM. Returns its message (may contain tool_calls)."""
    body = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},  # same question -> same choices (easier to test)
    }
    if use_tools:
        body["tools"] = [t["schema"] for t in TOOLS.values()]
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=config.LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]


def run_tool(name, arguments, role):
    """Look up the tool in the registry and run it. Never raises.
    Returns (arguments actually used, result)."""
    tool = TOOLS.get(name)
    if tool is None:
        return arguments, {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}."}

    if isinstance(arguments, str):  # some models send arguments as a JSON string
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    params = tool["schema"]["function"]["parameters"]
    # Keep only inputs the tool knows about (drops e.g. a sneaky "role": "Admin")
    args = {k: v for k, v in arguments.items() if k in params["properties"]}
    missing = [k for k in params["required"] if k not in args]
    if missing:
        return args, {"error": f"Missing input(s) for {name}: {', '.join(missing)}."}
    if tool["needs_role"]:
        args["role"] = role  # set by OUR code, from the logged-in user

    try:
        return args, tool["function"](**args)
    except Exception as e:  # tools shouldn't raise, but never let one crash the agent
        return args, {"error": f"Tool {name} failed: {type(e).__name__}."}


def needs_tools(question: str) -> bool:
    """Router: ask the LLM (no tools offered) whether this message needs facts."""
    message = chat([{"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": question}], use_tools=False)
    return "CHAT" not in message.get("content", "").upper()


POSTPONED = {"postponed": "Not run yet: the policy search in this same step had no result "
                          "yet. Read the search result, then call get_exchange_rate again "
                          "with the exact USD amount from the policy."}


def _must_wait(call, calls):
    """Dependency rule: a money conversion requested in the SAME step as a policy search
    probably uses a guessed amount — postpone it until the search result is known."""
    names = [c.get("function", {}).get("name") for c in calls]
    fn = call.get("function", {})
    args = fn.get("arguments")
    has_amount = isinstance(args, dict) and "amount" in args or isinstance(args, str) and "amount" in args
    return fn.get("name") == "get_exchange_rate" and has_amount and "search_documents" in names


def run_agent(question: str, role: str, max_steps: int = MAX_STEPS) -> dict:
    """Answer a question, using tools as the LLM decides.
    Returns {"answer", "trace", "steps", "stopped_early"}."""
    trace = []
    try:
        if not needs_tools(question):  # small talk: answer without offering tools
            message = chat([{"role": "system", "content": CHAT_PROMPT},
                            {"role": "user", "content": question}], use_tools=False)
            return {"answer": message.get("content", "").strip(), "trace": trace,
                    "steps": 1, "stopped_early": False}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for step in range(1, max_steps + 1):
            message = chat(messages)
            messages.append(message)

            calls = message.get("tool_calls") or []
            if not calls:  # no tool wanted -> this is the final answer
                return {"answer": message.get("content", "").strip(), "trace": trace,
                        "steps": step, "stopped_early": False}

            postponed = False
            for call in calls:
                name = call.get("function", {}).get("name", "")
                raw_args = call.get("function", {}).get("arguments", {})
                if _must_wait(call, calls):
                    used_args, result, postponed = raw_args, POSTPONED, True
                else:
                    used_args, result = run_tool(name, raw_args, role)
                trace.append({"step": step, "tool": name, "arguments": used_args, "result": result})
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
            if postponed:  # small models follow a direct instruction better than a tool note
                messages.append({"role": "user", "content": NUDGE})

        # Out of rounds: ask for an answer with what we have, tools switched off
        messages.append({"role": "user", "content": "Please give your final answer now, "
                                                    "using only the tool results above."})
        message = chat(messages, use_tools=False)
        return {"answer": message.get("content", "").strip(), "trace": trace,
                "steps": max_steps, "stopped_early": True}

    except requests.RequestException as e:
        return {"answer": f"Sorry, the AI model is not reachable right now ({type(e).__name__}). "
                          "Is Ollama running?", "trace": trace, "steps": 0, "stopped_early": True}


def short(result):
    """One-line summary of a tool result, for printing the trace."""
    if "error" in result:
        return "ERROR: " + result["error"]
    if "postponed" in result:
        return "POSTPONED (waiting for the policy search result)"
    if "results" in result:
        return ", ".join(c["source"] for c in result["results"])
    if "summary" in result:
        return result["summary"]
    if "rate" in result:
        text = f"1 {result['from']} = {result['rate']} {result['to']}"
        return text + (f"; {result['amount']} -> {result['converted']}" if "converted" in result else "")
    return str(result)[:100]


def print_result(result):
    for t in result["trace"]:
        print(f"  🔧 step {t['step']}: {t['tool']}({t['arguments']})\n      → {short(t['result'])}")
    if not result["trace"]:
        print("  (no tools used)")
    print(f"  💬 {result['answer']}")


if __name__ == "__main__":
    role = "Employee"
    if "--chat" in sys.argv:
        print(f"Acme assistant ({role}). Type 'quit' to stop.")
        while (q := input("\nAsk: ").strip()).lower() not in ("quit", "q", "exit"):
            if q:
                print_result(run_agent(q, role))
    else:
        for q in [
            "How many vacation days do I get per year?",
            "How much is 100 USD in EUR?",
            "What's the weather in Paris right now?",
            "Hi!",
            "I'm flying to Tokyo next week. What's my hotel limit per night in yen, "
            "and what's the weather there now?",
        ]:
            print(f"\n❓ {q}")
            print_result(run_agent(q, role))

"""Step 0 health check: proves every outside piece works before we build.

Run:  .\\.venv\\Scripts\\python.exe check_setup.py
"""

import requests

import config


def check_llm_chat():
    """Can the local LLM answer a message?"""
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.CHAT_MODEL,
            "messages": [{"role": "user", "content": "Say hello in 3 words."}],
            "stream": False,
        },
        timeout=config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    reply = r.json()["message"]["content"].strip()
    assert reply, "empty reply"
    return f"reply: {reply!r}"


def check_embeddings():
    """Can we turn text into a vector (list of numbers)?"""
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": "hello"},
        timeout=config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    vector = r.json()["embeddings"][0]
    assert len(vector) > 0, "empty vector"
    return f"vector length: {len(vector)}, first 3: {[round(x, 3) for x in vector[:3]]}"


def check_currency():
    """Can we get a live USD -> JPY rate?"""
    r = requests.get(
        f"{config.FRANKFURTER_URL}/latest",
        params={"from": "USD", "to": "JPY"},
        timeout=config.TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    rate = data["rates"]["JPY"]
    assert rate > 0, "rate not positive"
    return f"1 USD = {rate} JPY (date {data['date']})"


def check_weather():
    """Can we get live weather for Tokyo?"""
    r = requests.get(
        config.OPEN_METEO_URL,
        params={"latitude": 35.68, "longitude": 139.69, "current": "temperature_2m"},
        timeout=config.TIMEOUT,
    )
    r.raise_for_status()
    temp = r.json()["current"]["temperature_2m"]
    assert temp is not None, "no temperature"
    return f"Tokyo now: {temp} °C"


def check_tool_calling():
    """Does the LLM ask to call a tool when it should?"""
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.CHAT_MODEL,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": [weather_tool],
            "stream": False,
        },
        timeout=config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    calls = r.json()["message"].get("tool_calls") or []
    assert calls, "model did not call any tool"
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather", f"wrong tool: {fn['name']}"
    return f"model called {fn['name']}({fn['arguments']})"


CHECKS = [
    ("LLM chat", check_llm_chat),
    ("Embeddings", check_embeddings),
    ("Currency API (USD→JPY)", check_currency),
    ("Weather API (Tokyo)", check_weather),
    ("Tool calling", check_tool_calling),
]


def main():
    passed = 0
    for name, check in CHECKS:
        try:
            detail = check()
            print(f"✅ {name}: {detail}")
            passed += 1
        except Exception as e:  # one failure must not stop the other checks
            print(f"❌ {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()

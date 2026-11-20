"""Evaluate the whole agent on 30 questions and score it.

Real: llama3.2, embeddings, document search, grader, memory.
Mocked: the currency and weather APIs (fixed numbers -> repeatable expected answers).

Scores
- tools:   did the agent call exactly the expected tools?
- answer:  does the answer contain the expected facts (and none of the forbidden ones)?
- refusal: "I don't know" when it should, and NOT when it shouldn't
- memory:  was the fact saved / recalled?

Run:  .\\.venv\\Scripts\\python.exe -m eval.run_eval            (all 30, ~15 min)
      .\\.venv\\Scripts\\python.exe -m eval.run_eval mix-01 fx-02   (only some)
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import config

EVAL_DIR = Path(__file__).parent
QUESTIONS = EVAL_DIR / "questions.json"

# --- Mocked APIs: fixed, made-up numbers so expected answers never change -------
FAKE_RATES = {("USD", "EUR"): 0.9, ("EUR", "GBP"): 0.85, ("USD", "GBP"): 0.8, ("USD", "JPY"): 150.0,
              ("EUR", "USD"): 1.1, ("GBP", "USD"): 1.25, ("JPY", "USD"): 0.0067}
FAKE_WEATHER = {  # city -> (country, temperature, feels like, conditions)
    "tokyo": ("Japan", 12.0, 10.0, "Clear sky"),
    "paris": ("France", 8.0, 5.0, "Light rain"),
    "berlin": ("Germany", -2.0, -6.0, "Light snow"),
    "london": ("United Kingdom", 10.0, 8.0, "Overcast"),
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fake_fetch_rate(src, dst):
    if (src, dst) not in FAKE_RATES:
        return {"error": f"Unknown currency code '{src}' or '{dst}'. Use codes like USD, EUR, JPY."}
    return {"rate": FAKE_RATES[(src, dst)], "date": "2026-11-20", "as_of": _now()}


def fake_fetch_weather(city):
    key = city.strip().lower().split(",")[0]
    if key not in FAKE_WEATHER:
        return {"error": f"City '{city}' not found. Check the spelling, e.g. 'Tokyo' or 'Paris'."}
    country, temp, feels, cond = FAKE_WEATHER[key]
    name = key.title()
    return {"city": name, "country": country, "latitude": 0.0, "longitude": 0.0,
            "temperature_c": temp, "feels_like_c": feels, "humidity_pct": 60.0, "wind_kmh": 10.0,
            "conditions": cond, "summary": f"{name}, {country}: {cond}, {temp} °C (feels like {feels} °C), wind 10.0 km/h",
            "observed_at": "2026-11-20T12:00", "source": "Open-Meteo (mocked)", "as_of": _now()}


# --- Scoring (pure functions, unit-tested in tests/test_eval.py) ----------------

def normalize(text: str) -> str:
    """Lower-case, no thousands separators: '¥33,000.0' -> '¥33000.0'."""
    return text.lower().replace(",", "").replace("−", "-")


def contains(answer: str, expected: str) -> bool:
    """expected may list alternatives with '|': 'clear|sunny'."""
    a = normalize(answer)
    return any(normalize(alt) in a for alt in expected.split("|"))


def is_refusal(answer: str) -> bool:
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()


def score(case: dict, result: dict, saved_memories: list[str]) -> dict:
    """Return {metric: True/False} for every metric that applies to this case."""
    answer = result["answer"]
    used = {t["tool"] for t in result["trace"]}
    scores = {"tools": used == set(case.get("tools", []))}

    if not case.get("should_refuse"):
        must = case.get("must_contain", [])
        scores["answer"] = all(contains(answer, m) for m in must)
    else:
        scores["answer"] = True
    forbidden = case.get("must_not_contain", [])
    if forbidden:
        scores["answer"] = scores["answer"] and not any(normalize(f) in normalize(answer) for f in forbidden)

    scores["refusal"] = is_refusal(answer) == bool(case.get("should_refuse"))

    if "expect_saved" in case:
        scores["memory"] = any(case["expect_saved"] in m.lower() for m in saved_memories)
    if "expect_recalled" in case:
        scores["memory"] = any(case["expect_recalled"] in m.lower() for m in result.get("memories_used", []))
    return scores


def summarize(rows: list[dict]) -> dict:
    """{metric: (passed, total)} plus per-category pass counts (all metrics passed)."""
    metrics = {}
    for row in rows:
        for metric, ok in row["scores"].items():
            passed, total = metrics.get(metric, (0, 0))
            metrics[metric] = (passed + ok, total + 1)
    categories = {}
    for row in rows:
        passed, total = categories.get(row["category"], (0, 0))
        categories[row["category"]] = (passed + all(row["scores"].values()), total + 1)
    return {"metrics": metrics, "categories": categories,
            "all_passed": (sum(all(r["scores"].values()) for r in rows), len(rows))}


# --- Running --------------------------------------------------------------------

def run(only=None) -> list[dict]:
    import agent
    import memory

    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]

    tmp = Path(tempfile.mkdtemp(prefix="eval_"))
    config.CACHE_DB = tmp / "cache.db"  # fresh cache: never reuse REAL rates from earlier runs
    config.MEMORY_DB = tmp / "memory.db"  # don't touch the user's real memories

    rows = []
    with patch("tools.currency._fetch_rate", side_effect=fake_fetch_rate), \
         patch("tools.weather._fetch_weather", side_effect=fake_fetch_weather):
        for i, case in enumerate(cases, 1):
            user = f"eval-{case['id']}"
            memory.forget_me(user)
            for m in case.get("setup_memories", []):
                memory.save_memory(user, m)

            start = time.time()
            result = agent.run_agent(case["question"], case["role"], user_id=user)
            seconds = round(time.time() - start, 1)
            saved = [m["text"] for m in memory.list_memories(user)]
            memory.forget_me(user)

            scores = score(case, result, saved)
            rows.append({"id": case["id"], "category": case["category"], "question": case["question"],
                         "answer": result["answer"], "tools_used": [t["tool"] for t in result["trace"]],
                         "scores": scores, "seconds": seconds})
            marks = " ".join(f"{k}:{'✅' if v else '❌'}" for k, v in scores.items())
            print(f"[{i:2}/{len(cases)}] {case['id']:8} {seconds:5}s  {marks}")
            if not all(scores.values()):
                print(f"           tools={rows[-1]['tools_used']}  answer={result['answer'][:160]!r}")
    return rows


def write_report(rows: list[dict], summary: dict) -> Path:
    lines = ["# Evaluation report", "",
             f"Run: {_now()} · model `{config.CHAT_MODEL}` · {len(rows)} questions · APIs mocked", "",
             "## Scores", "", "| Metric | Passed | Score |", "|---|---|---|"]
    for metric, (p, t) in summary["metrics"].items():
        lines.append(f"| {metric} | {p}/{t} | {round(100 * p / t)}% |")
    p, t = summary["all_passed"]
    lines += [f"| **all checks passed** | **{p}/{t}** | **{round(100 * p / t)}%** |", "",
              "## By category", "", "| Category | All checks passed |", "|---|---|"]
    for cat, (p, t) in summary["categories"].items():
        lines.append(f"| {cat} | {p}/{t} |")
    lines += ["", "## Every question", "", "| ID | Result | Tools used | Answer |", "|---|---|---|---|"]
    for r in rows:
        mark = "✅" if all(r["scores"].values()) else "❌ " + ", ".join(k for k, v in r["scores"].items() if not v)
        answer = r["answer"].replace("|", "/").replace("\n", " ")[:140]
        lines.append(f"| {r['id']} | {mark} | {', '.join(r['tools_used']) or '—'} | {answer} |")
    report = EVAL_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EVAL_DIR / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    rows = run(only=sys.argv[1:] or None)
    summary = summarize(rows)
    print("\n=== Scores")
    for metric, (p, t) in summary["metrics"].items():
        print(f"  {metric:8} {p}/{t}  ({round(100 * p / t)}%)")
    p, t = summary["all_passed"]
    print(f"  all checks passed: {p}/{t}")
    if not sys.argv[1:]:
        print("Report:", write_report(rows, summary))

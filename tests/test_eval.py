"""Tests for the evaluation's scoring rules and mocked APIs (offline)."""

import json

from eval import run_eval
from eval.run_eval import contains, fake_fetch_rate, fake_fetch_weather, is_refusal, score, summarize


def result(answer, tools=(), memories=()):
    return {"answer": answer, "trace": [{"tool": t} for t in tools], "memories_used": list(memories)}


def test_questions_file_has_30_valid_cases():
    cases = json.loads(run_eval.QUESTIONS.read_text(encoding="utf-8"))
    assert len(cases) == 30
    assert len({c["id"] for c in cases}) == 30  # ids unique
    for c in cases:
        assert {"id", "category", "role", "question", "tools"} <= c.keys()


def test_contains_ignores_commas_case_and_allows_alternatives():
    assert contains("Your limit is ¥33,000.0 per night.", "33000")
    assert contains("It is SUNNY today", "clear|sunny")
    assert not contains("It is raining", "clear|sunny")


def test_refusal_detection():
    assert is_refusal("I don't know based on the company documents.")
    assert not is_refusal("You get 20 days.")


def test_score_good_docs_answer():
    case = {"tools": ["search_documents"], "must_contain": ["20"]}
    assert score(case, result("You get 20 days.", ["search_documents"]), []) == \
        {"tools": True, "answer": True, "refusal": True}


def test_score_wrong_tools_and_false_refusal():
    case = {"tools": ["search_documents"], "must_contain": ["20"]}
    s = score(case, result("I don't know based on the company documents.", []), [])
    assert s == {"tools": False, "answer": False, "refusal": False}


def test_score_refusal_with_leak_fails():
    case = {"tools": ["search_documents"], "should_refuse": True, "must_not_contain": ["85,000"]}
    leak = score(case, result("Senior Engineers earn 85,000 to 115,000.", ["search_documents"]), [])
    assert leak["answer"] is False and leak["refusal"] is False
    good = score(case, result("I don't know based on the company documents.", ["search_documents"]), [])
    assert all(good.values())


def test_score_memory():
    saved = {"tools": ["save_memory"], "expect_saved": "vegetarian"}
    assert score(saved, result("Noted!", ["save_memory"]), ["The user is vegetarian."])["memory"]
    recalled = {"tools": [], "expect_recalled": "vegetarian"}
    assert not score(recalled, result("Hello", [], []), [])["memory"]


def test_summarize():
    rows = [{"category": "docs", "scores": {"tools": True, "answer": True}},
            {"category": "docs", "scores": {"tools": True, "answer": False}}]
    s = summarize(rows)
    assert s["metrics"] == {"tools": (2, 2), "answer": (1, 2)}
    assert s["categories"] == {"docs": (1, 2)} and s["all_passed"] == (1, 2)


def test_mocked_apis_are_fixed():
    assert fake_fetch_rate("USD", "JPY")["rate"] == 150.0
    assert "error" in fake_fetch_rate("USD", "XYZ")
    assert fake_fetch_weather("Paris")["conditions"] == "Light rain"
    assert "not found" in fake_fetch_weather("Atlantisxyz")["error"]

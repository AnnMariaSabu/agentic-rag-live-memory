"""Corrective RAG: check search results before trusting them.

search -> LLM grades each chunk (JSON, checked by Pydantic) -> keep relevant ones.
Nothing relevant? -> LLM rewrites the query -> search + grade ONCE more.
Still nothing? -> return no results + a note, so the agent says "I don't know".
"""

import requests
from pydantic import BaseModel, Field, ValidationError

import config
from tools.docs import search_documents

NO_ANSWER = "I don't know based on the company documents."

GRADE_PROMPT = """You check search results for a company helpdesk.
First write the reason in this form:
"Question asks about <specific subject>. Piece mentions <subjects in the piece>."
Then decide:
- relevant = true if the piece mentions the question's specific subject,
  even if it does not fully answer it. Synonyms count as the same subject
  (e.g. vacation = annual leave = holiday, pay = salary, laptop = computer).
- relevant = false if it does not. General words like "policy", "company",
  "employees" or "rules" do NOT count as the same subject.
Reply in JSON: {"reason": "...", "relevant": true or false}"""

REWRITE_PROMPT = """A search of Acme's company policy documents found nothing useful.
Rewrite the search as a better short query (3 to 8 words) using simple, common
policy words and synonyms. Reply in JSON: {"query": "<new query>"}"""


class Grade(BaseModel):
    # FIRST and REQUIRED: the model must "think" before it decides (better grades).
    # (With a default value the schema made it optional and the model skipped it.)
    reason: str = Field(min_length=3)
    relevant: bool


class Rewrite(BaseModel):
    query: str = Field(min_length=2)


def ask_json(system: str, user: str, model_cls):
    """Ask the LLM for JSON matching model_cls. Returns a model instance, or None
    if the reply isn't valid (we never trust LLM output blindly)."""
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.CHAT_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "format": model_cls.model_json_schema(),  # Ollama forces this JSON shape
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    try:
        return model_cls.model_validate_json(r.json()["message"]["content"])
    except (ValidationError, KeyError, ValueError):
        return None


def grade_chunk(question: str, chunk: dict) -> Grade:
    grade = ask_json(GRADE_PROMPT, f"Question: {question}\n\nDocument piece:\n{chunk['text']}", Grade)
    return grade or Grade(reason="grader reply was not valid JSON", relevant=False)


def corrective_search(query: str, role: str, k: int = config.TOP_K) -> dict:
    """Like search_documents, but only returns chunks the grader judged relevant.
    Retries once with a rewritten query. Never raises."""
    attempts = []
    current = query
    try:
        for attempt in (1, 2):
            found = search_documents(current, role, k)
            if "error" in found:
                return found
            kept = []
            for chunk in found["results"]:
                grade = grade_chunk(current, chunk)
                if grade.relevant:
                    kept.append({**chunk, "grade_reason": grade.reason})
            attempts.append({"query": current, "kept": len(kept), "checked": len(found["results"])})
            if kept:
                return {"query": query, "role": role, "results": kept, "attempts": attempts}

            if attempt == 1:  # nothing useful: rewrite and try once more
                rewrite = ask_json(REWRITE_PROMPT, f"Search that failed: {query}", Rewrite)
                if rewrite is None or rewrite.query.strip().lower() == query.strip().lower():
                    break
                current = rewrite.query.strip()
    except requests.RequestException as e:
        return {"error": f"The grading model is not reachable ({type(e).__name__})."}

    return {"query": query, "role": role, "results": [], "attempts": attempts,
            "note": f"No relevant company documents found. Tell the user: {NO_ANSWER}"}


if __name__ == "__main__":
    # Demo: python grader.py
    for q in ["hotel limit international trips", "dress code policy", "can I bring my dog to work"]:
        out = corrective_search(q, "Employee")
        print(f"\n=== {q}\n  attempts: {out.get('attempts')}")
        for c in out.get("results", []):
            print(f"  ✅ [{c['source']}] {c['grade_reason']}")
        if not out.get("results"):
            print("  ❌", out.get("note") or out.get("error"))

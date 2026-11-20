"""A small cache with expiry (TTL = "time to live"), stored in SQLite.

Why: live data is slow to fetch and free APIs have limits. If we asked for
USD->JPY 2 minutes ago, reuse that answer instead of calling again.
But not forever: after the TTL the data counts as stale and is fetched again.
Errors are never cached (next call should retry).
"""

import json
import sqlite3
import time

import config


def _db():
    conn = sqlite3.connect(config.CACHE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY, value TEXT NOT NULL, stored_at REAL NOT NULL)""")
    return conn


def _now() -> float:
    return time.time()


def get(key: str, ttl_seconds: float):
    """(value, age_seconds) if a fresh entry exists, else None."""
    with _db() as conn:
        row = conn.execute("SELECT value, stored_at FROM cache WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    age = _now() - row[1]
    if age > ttl_seconds:  # too old: treat as missing
        return None
    return json.loads(row[0]), age


def put(key: str, value) -> None:
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), _now()))


def clear() -> None:
    with _db() as conn:
        conn.execute("DELETE FROM cache")


def cached_call(key: str, ttl_minutes: float, fetch):
    """Return (value, info). Uses the cache if fresh, else calls fetch() and stores
    the result (unless it is an error). info = {"cached": bool, "age_minutes": float}."""
    hit = get(key, ttl_minutes * 60)
    if hit is not None:
        value, age = hit
        return value, {"cached": True, "age_minutes": round(age / 60, 1)}
    value = fetch()
    if "error" not in value:
        put(key, value)
    return value, {"cached": False, "age_minutes": 0.0}

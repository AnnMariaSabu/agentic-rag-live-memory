"""Tests for Step 7: TTL cache, freshness info, and re-ingest via file hash."""

import shutil
from unittest.mock import MagicMock, patch

import pytest
import requests

import agent
import cache
import config
import ingest
from tests.conftest import fake_embed
from tools.currency import get_exchange_rate
from tools.weather import get_weather


class Clock:
    """A fake clock we can move forward, so tests don't have to wait 60 minutes."""
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(cache, "_now", c)
    return c


def fake_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


RATE_REPLY = {"date": "2026-11-20", "rates": {"JPY": 158.64}}


# --- the cache itself ---

def test_put_get_and_expiry(clock):
    cache.put("k", {"v": 1})
    clock.t += 59
    assert cache.get("k", ttl_seconds=60) == ({"v": 1}, 59)
    clock.t += 2  # now 61 s old
    assert cache.get("k", ttl_seconds=60) is None


def test_errors_are_never_cached(clock):
    fetch = MagicMock(return_value={"error": "down"})
    cache.cached_call("k", 60, fetch)
    cache.cached_call("k", 60, fetch)
    assert fetch.call_count == 2  # tried again, not served from cache


# --- currency: 60 min TTL ---

@patch("tools.currency.requests.get", return_value=fake_response(RATE_REPLY))
def test_currency_second_call_uses_cache(mock_get, clock):
    first = get_exchange_rate("USD", "JPY")
    clock.t += 10 * 60
    second = get_exchange_rate("usd", "jpy", amount=220)  # different amount, same rate
    assert mock_get.call_count == 1
    assert first["cached"] is False
    assert second["cached"] is True and second["age_minutes"] == 10.0
    assert second["converted"] == round(220 * 158.64, 2)
    assert second["as_of"] == first["as_of"]  # "as of" = when it was REALLY fetched


@patch("tools.currency.requests.get", return_value=fake_response(RATE_REPLY))
def test_currency_refetches_after_ttl(mock_get, clock):
    get_exchange_rate("USD", "JPY")
    clock.t += config.CURRENCY_TTL_MIN * 60 + 1
    assert get_exchange_rate("USD", "JPY")["cached"] is False
    assert mock_get.call_count == 2


@patch("tools.currency.requests.get", side_effect=requests.Timeout)
def test_currency_error_not_cached(mock_get, clock):
    get_exchange_rate("USD", "JPY")
    get_exchange_rate("USD", "JPY")
    assert mock_get.call_count == 2


# --- weather: 15 min TTL ---

GEO = {"results": [{"name": "Tokyo", "country": "Japan", "latitude": 35.7, "longitude": 139.7}]}
NOW = {"current": {"time": "2026-11-20T18:00", "temperature_2m": 10.0, "apparent_temperature": 7.0,
                   "relative_humidity_2m": 50, "wind_speed_10m": 5.0, "weather_code": 0}}


@patch("tools.weather.requests.get")
def test_weather_cache_and_ttl(mock_get, clock):
    mock_get.side_effect = [fake_response(GEO), fake_response(NOW)] * 2
    get_weather("Tokyo")
    clock.t += 5 * 60
    assert get_weather(" tokyo ")["cached"] is True  # case/space-insensitive key
    assert mock_get.call_count == 2  # geocode + weather, only once
    clock.t += config.WEATHER_TTL_MIN * 60
    assert get_weather("Tokyo")["cached"] is False
    assert mock_get.call_count == 4


# --- re-ingest only changed files (file hash) ---

@pytest.fixture
def docs_copy(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    shutil.copytree(config.DOCS_DIR, docs)
    monkeypatch.setattr(config, "DOCS_DIR", docs)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    with patch("rag.embeddings.embed", side_effect=fake_embed) as embed:
        yield docs, embed


def test_ingest_skips_unchanged_files(docs_copy):
    docs, embed = docs_copy
    assert ingest.run_ingest()["added"] == 8
    embed.reset_mock()
    counts = ingest.run_ingest()
    assert counts == {"added": 0, "updated": 0, "unchanged": 8, "removed": 0}
    embed.assert_not_called()  # nothing was re-embedded


def test_ingest_redoes_changed_and_removes_deleted(docs_copy):
    docs, embed = docs_copy
    ingest.run_ingest()
    (docs / "hr" / "parking_policy.txt").write_text("Parking is free on Fridays.", encoding="utf-8")
    (docs / "it" / "it_policy.txt").unlink()
    counts = ingest.run_ingest()
    assert counts == {"added": 0, "updated": 1, "unchanged": 6, "removed": 1}

    import chromadb
    collection = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION_NAME)
    parking = collection.get(where={"source": "hr/parking_policy.txt"})
    assert parking["documents"] == ["Parking is free on Fridays."]  # old chunks gone
    assert collection.get(where={"source": "it/it_policy.txt"})["ids"] == []


def test_ingest_rebuild_redoes_everything(docs_copy):
    ingest.run_ingest()
    assert ingest.run_ingest(rebuild=True)["added"] == 8


# --- agent freshness notes ---

def test_freshness_notes():
    trace = [
        {"tool": "get_exchange_rate", "result": {"from": "USD", "to": "JPY", "rate": 158.64,
                                                 "date": "2026-11-20", "as_of": "2026-11-20T09:00:39+00:00",
                                                 "cached": True, "age_minutes": 12.0}},
        {"tool": "get_weather", "result": {"city": "Tokyo", "summary": "...", "observed_at": "2026-11-20T18:00",
                                           "as_of": "2026-11-20T09:01:00+00:00", "cached": False}},
        {"tool": "search_documents", "result": {"results": []}},
    ]
    notes = agent.freshness_notes(trace)
    assert notes == [
        "Exchange rate USD→JPY: ECB rate of 2026-11-20, fetched 2026-11-20 09:00 UTC (from cache, 12.0 min old)",
        "Weather Tokyo: observed 2026-11-20 18:00 local time, fetched 2026-11-20 09:01 UTC (just fetched)",
    ]

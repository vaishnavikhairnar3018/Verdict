"""The speed-ups must not change results: same sources, same order, and sensible retries."""
import time

from fastapi.testclient import TestClient

import crew.search as search
import server
from crew.llm import read_429


def test_parallel_search_keeps_the_original_order(monkeypatch):
    # Make the FIRST query the slowest: with plain threading it would finish last,
    # but ids must still be numbered in query order.
    def slow_search(query):
        time.sleep({"a": 0.3, "b": 0.1, "c": 0.0}[query])
        return [{"title": query, "url": f"https://{query}.org", "snippet": ""}]

    monkeypatch.setattr(search, "web_search", slow_search)
    new, by_query = search.collect_sources(["a", "b", "c"], [])

    assert [s.title for s in new] == ["a", "b", "c"]
    assert by_query == {"a": ["S1"], "b": ["S2"], "c": ["S3"]}


def test_parallel_search_is_actually_faster(monkeypatch):
    monkeypatch.setattr(search, "web_search", lambda q: time.sleep(0.2) or [])
    start = time.time()
    search.collect_sources(["q1", "q2", "q3", "q4"], [])
    assert time.time() - start < 0.5        # one at a time would take 0.8 s


def test_duplicate_queries_are_searched_once(monkeypatch):
    calls = []
    monkeypatch.setattr(search, "web_search", lambda q: calls.append(q) or [])
    search.collect_sources(["same", "same", "other"], [])
    assert sorted(calls) == ["other", "same"]


def test_daily_quota_errors_skip_the_waiting():
    error = ("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'violations': "
             "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}]}}")
    assert read_429(error) == (True, None)


def test_per_minute_limits_use_geminis_own_retry_hint():
    error = ("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'@type': 'RetryInfo', "
             "'retryDelay': '23s'}, {'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel'}]}}")
    assert read_429(error) == (False, 23.0)


def test_runs_record_how_long_each_agent_took(fake_llm):
    body = TestClient(server.app).get("/api/research", params={"q": "Timed question?"}).text
    assert '"seconds":' in body
    assert '"timings":' in body and '"total_seconds":' in body

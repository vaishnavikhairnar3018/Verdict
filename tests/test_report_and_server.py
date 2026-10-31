from datetime import datetime, timedelta

from fastapi.testclient import TestClient

import crew.report as report
import server
from crew.graph import build_graph, initial_state


class Clock:
    """A fake datetime whose now() moves forward one second per call."""
    current = datetime(2026, 10, 31, 12, 0, 0)

    @classmethod
    def now(cls):
        cls.current += timedelta(seconds=1)
        return cls.current


def test_asking_the_same_question_twice_keeps_both_runs(fake_llm, monkeypatch):
    monkeypatch.setattr(report, "datetime", Clock)
    state = build_graph().invoke(initial_state("Same question?"))

    first, second = report.save(state), report.save(state)

    assert first != second
    assert first.exists() and second.exists()
    assert first.name.startswith("same-question-")


def test_markdown_report_lists_question_answer_and_sources(fake_llm):
    state = build_graph().invoke(initial_state("Report question?"))
    text = report.to_markdown(state)

    assert text.startswith("# Report question?")
    assert "## Answer" in text and "## Sources" in text


def test_api_lists_and_returns_saved_runs(fake_llm):
    state = build_graph().invoke(initial_state("API question?"))
    slug = report.save(state, {"calls": 6, "models": ["fake-model"]}).stem
    client = TestClient(server.app)

    runs = client.get("/api/runs").json()
    assert [r["slug"] for r in runs] == [slug]

    run = client.get(f"/api/runs/{slug}").json()
    assert run["question"] == "API question?"
    assert run["usage"]["models"] == ["fake-model"]


def test_api_rejects_path_traversal():
    client = TestClient(server.app)
    for bad in ["..%2F..%2F.env", "UPPERCASE", "has.dot", "nonexistent-run"]:
        assert client.get(f"/api/runs/{bad}").status_code == 404


def test_research_endpoint_streams_live_events(fake_llm):
    client = TestClient(server.app)
    body = client.get("/api/research", params={"q": "Streaming question?"}).text

    events = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]
    assert events[:2] == ["start", "finish"]               # researcher starts, then finishes
    assert events[-1] == "done"
    assert events.count("start") == events.count("finish")


def test_dashboard_page_is_served():
    client = TestClient(server.app)
    assert "<title>Verdict</title>" in client.get("/").text
    assert client.get("/static/logo.svg").status_code == 200

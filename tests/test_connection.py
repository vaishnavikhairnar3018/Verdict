"""A dropped or quiet connection must never block the server or lose the run."""
import time

from fastapi.testclient import TestClient

import crew.agents as agents
import server


def wait_until(condition, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_lock_is_released_even_if_the_browser_disconnects(fake_llm, tmp_path):
    client = TestClient(server.app)
    with client.stream("GET", "/api/research", params={"q": "Who leaves early?"}) as response:
        next(response.iter_lines())                    # read one line, then hang up

    # The crew finishes in the background, frees the lock and still saves the run.
    assert wait_until(lambda: not server.crew_lock.locked())
    assert wait_until(lambda: list(tmp_path.glob("who-leaves-early-*.json")))

    body = client.get("/api/research", params={"q": "Next visitor?"}).text
    assert "event: done" in body                       # the server is not stuck on "busy"


def test_quiet_periods_send_a_heartbeat(fake_llm, monkeypatch):
    monkeypatch.setattr(server, "HEARTBEAT", 0.05)

    def slow(*args, **kwargs):
        time.sleep(0.2)                                # an LLM call slower than the heartbeat
        return fake_llm(*args, **kwargs)

    monkeypatch.setattr(agents, "ask", slow)
    body = TestClient(server.app).get("/api/research", params={"q": "Slow question?"}).text

    assert ": still working" in body
    assert "event: done" in body

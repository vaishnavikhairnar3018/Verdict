"""Tavily when there's a key, DuckDuckGo otherwise, and never a crash if Tavily fails."""
import crew.search as search

DDG_RESULT = [{"title": "from duckduckgo", "href": "https://ddg.example", "body": "d"}]


def tavily_ok(query):
    return [{"title": "from tavily", "href": "https://tavily.example", "body": "t"}]


def tavily_broken(query):
    raise RuntimeError("out of credits")


def test_without_a_key_duckduckgo_is_used(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(search, "_tavily", tavily_ok)
    monkeypatch.setattr(search, "_duckduckgo", lambda q: DDG_RESULT)

    assert search.search_engine() == "duckduckgo"
    assert search._search_once("q") == DDG_RESULT


def test_with_a_key_tavily_is_used(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(search, "_tavily", tavily_ok)
    monkeypatch.setattr(search, "_duckduckgo", lambda q: DDG_RESULT)

    assert search.search_engine() == "tavily"
    assert search._search_once("q")[0]["title"] == "from tavily"


def test_if_tavily_fails_duckduckgo_takes_over(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(search, "_tavily", tavily_broken)
    monkeypatch.setattr(search, "_duckduckgo", lambda q: DDG_RESULT)

    assert search._search_once("q") == DDG_RESULT


def test_if_tavily_finds_nothing_duckduckgo_gets_a_chance(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(search, "_tavily", lambda q: [])
    monkeypatch.setattr(search, "_duckduckgo", lambda q: DDG_RESULT)

    assert search._search_once("q") == DDG_RESULT


def test_tavily_results_are_converted_to_the_common_format(monkeypatch):
    class FakeClient:
        def search(self, query, max_results):
            return {"results": [{"title": "T", "url": "https://t.org", "content": "x" * 5000, "score": 0.9}]}

    monkeypatch.setattr(search, "_tavily_client", FakeClient())
    [result] = search._tavily("q")

    assert result["title"] == "T" and result["href"] == "https://t.org"
    assert len(result["body"]) == search.SNIPPET_CHARS       # long page text is capped

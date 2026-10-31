import os
from concurrent.futures import ThreadPoolExecutor

from ddgs import DDGS
from tavily import TavilyClient

from .config import RESULTS_PER_QUERY
from .schemas import Source

SEARCH_WORKERS = 4   # searches at once; more risks the search engine blocking us
SNIPPET_CHARS = 700  # Tavily returns longer text than DuckDuckGo; cap it to keep prompts small

_tavily_client = None


def search_engine() -> str:
    """Tavily if a key is set (reliable from cloud servers), otherwise DuckDuckGo (no key needed)."""
    return "tavily" if os.getenv("TAVILY_API_KEY") else "duckduckgo"


def _tavily(query: str) -> list[dict]:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = _tavily_client.search(query, max_results=RESULTS_PER_QUERY)
    # Rename Tavily's fields to the ones DuckDuckGo uses, so the rest of the code doesn't care.
    return [{"title": r.get("title", ""), "href": r.get("url", ""),
             "body": (r.get("content") or "")[:SNIPPET_CHARS]}
            for r in response.get("results", [])]


def _duckduckgo(query: str) -> list[dict]:
    try:
        return DDGS().text(query, max_results=RESULTS_PER_QUERY)
    except Exception:
        return []   # ddgs raises an error when there are simply no results


def _search_once(query: str) -> list[dict]:
    if search_engine() == "tavily":
        try:
            results = _tavily(query)
            if results:
                return results
        except Exception as e:
            # Out of free credits, network problem...: one bad search must not stop the crew.
            print(f"    (Tavily failed: {e}; using DuckDuckGo)")
    return _duckduckgo(query)


def web_search(query: str) -> list[dict]:
    """Search the web. If nothing comes back, retry with a shorter, simpler query."""
    results = _search_once(query)
    if not results:
        simpler = " ".join(query.replace('"', "").split()[:4])   # keep the first 4 words
        if simpler != query:
            print(f"    no results, retrying as: {simpler}")
            results = _search_once(simpler)
    if not results:
        print(f"    ! no results for {query!r}")
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results]


def collect_sources(queries: list[str], known: list[Source]) -> tuple[list[Source], dict[str, list[str]]]:
    """Run every query, skip URLs we already have, and give each new page an id.

    Returns the new sources plus a map  query -> ids of sources it found.
    """
    queries = list(dict.fromkeys(queries))     # drop duplicate queries, keep the order

    # Searching is mostly waiting for the network, so run several searches at the same time.
    # pool.map returns results in the same order as `queries`, so source ids stay identical
    # to running them one by one.
    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
        all_results = list(pool.map(web_search, queries))

    seen = {s.url for s in known}
    new: list[Source] = []
    by_query: dict[str, list[str]] = {}

    for q, results in zip(queries, all_results):
        print(f"    search: {q}")
        by_query[q] = []
        for r in results:
            # Only real web links; anything else (javascript:, data:, empty) is dropped.
            if not r["url"].startswith(("http://", "https://")) or r["url"] in seen:
                continue
            seen.add(r["url"])
            source = Source(id=f"S{len(known) + len(new) + 1}", **r)
            new.append(source)
            by_query[q].append(source.id)

    return new, by_query


def format_sources(sources: list[Source]) -> str:
    """Wrap each source in tags so the LLM can tell data apart from instructions."""
    return "\n".join(
        f'<source id="{s.id}" url="{s.url}">\n{s.title}\n{s.snippet}\n</source>'
        for s in sources
    )

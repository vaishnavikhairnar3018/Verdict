from ddgs import DDGS

from .config import RESULTS_PER_QUERY
from .schemas import Source


def _search_once(query: str) -> list[dict]:
    try:
        return DDGS().text(query, max_results=RESULTS_PER_QUERY)
    except Exception:
        return []   # ddgs raises an error when there are simply no results


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
    seen = {s.url for s in known}
    new: list[Source] = []
    by_query: dict[str, list[str]] = {}

    for q in queries:
        print(f"    search: {q}")
        by_query[q] = []
        for r in web_search(q):
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

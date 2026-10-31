"""Shared test setup: a fake LLM and a fake web search, so tests are fast, free and offline."""
import re

import pytest

import crew.agents as agents
import crew.report as report
import crew.search as search
import server
from crew.schemas import (ClaimList, CounterPlan, CounterQuery, DraftClaim, FinalAnswer,
                          JudgeVerdict, Objection, ObjectionList, SearchPlan, VerdictList)


def fake_web_search(query: str) -> list[dict]:
    """Two results per query, each on its own website, plus one unsafe link that must be dropped."""
    key = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return [
        {"title": f"First page on {query}", "url": f"https://{key}-1.example.org/a", "snippet": "..."},
        {"title": f"Second page on {query}", "url": f"https://{key}-2.example.org/b", "snippet": "..."},
        {"title": "Unsafe", "url": "javascript:alert(1)", "snippet": "..."},
    ]


class FakeLLM:
    """Stands in for crew.llm.ask. Each test can change how the agents 'answer'."""

    def __init__(self):
        self.judge_confidence = 1.0          # what the Judge says for every claim
        self.judge_skips_last = False        # simulate the Judge forgetting a claim
        self.writer_answer = None            # override the Writer's text
        self.calls = []

    def __call__(self, system, prompt, schema, effort="low"):
        self.calls.append(schema.__name__)
        source_ids = re.findall(r'<source id="(S\d+)"', prompt)
        claim_ids = list(dict.fromkeys(re.findall(r"^#* ?(C\d+):", prompt, re.M)))

        if schema is SearchPlan:
            return SearchPlan(queries=["query one", "query two", "query three"])
        if schema is ClaimList:
            # Each claim cites two sources plus one id that doesn't exist (a "hallucinated" citation).
            pairs = [source_ids[i:i + 2] for i in range(0, min(len(source_ids), 6), 2)]
            return ClaimList(claims=[DraftClaim(text=f"Claim backed by {', '.join(p)}",
                                                source_ids=p + ["S999"]) for p in pairs])
        if schema is CounterPlan:
            return CounterPlan(items=[CounterQuery(claim_id=c, queries=[f"against {c}"]) for c in claim_ids])
        if schema is ObjectionList:
            return ObjectionList(objections=[Objection(claim_id=c, strength="none", objection="No problems found.",
                                                       source_ids=[]) for c in claim_ids])
        if schema is VerdictList:
            judged = claim_ids[:-1] if self.judge_skips_last else claim_ids
            return VerdictList(verdicts=[JudgeVerdict(claim_id=c, verdict="supported",
                                                      confidence=self.judge_confidence,
                                                      reasoning="Well supported.") for c in judged])
        if schema is FinalAnswer:
            ids = re.findall(r"^\[(C\d+)\]", prompt, re.M)
            return FinalAnswer(answer=self.writer_answer or " ".join(f"Fact [{i}]." for i in ids))
        raise AssertionError(f"Unexpected schema {schema}")


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Patch the LLM, the web search and the reports folder for one test."""
    llm = FakeLLM()
    monkeypatch.setattr(agents, "ask", llm)
    monkeypatch.setattr(search, "web_search", fake_web_search)
    monkeypatch.setattr(report, "REPORTS", tmp_path)
    monkeypatch.setattr(server, "REPORTS", tmp_path)
    return llm

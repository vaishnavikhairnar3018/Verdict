"""Run the whole crew (Researcher -> Skeptic -> Judge -> Writer) with the fake LLM."""
import re

from crew.config import KEEP_THRESHOLD, MAX_ROUNDS
from crew.graph import build_graph, initial_state


def run(question="Is this a test?"):
    return build_graph().invoke(initial_state(question))


def test_happy_path_produces_a_cited_answer(fake_llm):
    state = run()

    assert state["claims"], "the Researcher should extract claims"
    assert state["answer"].count("[C") == len(state["claims"])
    assert fake_llm.calls == ["SearchPlan", "ClaimList", "CounterPlan", "ObjectionList",
                              "VerdictList", "FinalAnswer"]


def test_hallucinated_citations_are_removed(fake_llm):
    state = run()
    known = {s.id for s in state["sources"]}
    for claim in state["claims"]:
        assert "S999" not in claim.source_ids
        assert set(claim.source_ids) <= known


def test_unsafe_links_never_become_sources(fake_llm):
    state = run()
    assert all(s.url.startswith("https://") for s in state["sources"])


def test_every_claim_gets_a_verdict_even_if_the_judge_skips_one(fake_llm):
    fake_llm.judge_skips_last = True
    state = run()

    verdicts = {v.claim_id: v for v in state["verdicts"]}
    assert set(verdicts) == {c.id for c in state["claims"]}
    skipped = [v for v in verdicts.values() if "did not rule" in v.reasoning]
    assert len(skipped) == 1 and skipped[0].verdict == "unverifiable"


def test_writer_cannot_cite_unverified_claims(fake_llm):
    fake_llm.judge_skips_last = True                      # the last claim will be dropped
    fake_llm.writer_answer = "True thing [C1]. Dropped thing [C3]. Invented thing [C99]."
    state = run()

    kept = {v.claim_id for v in state["verdicts"] if v.final_confidence >= KEEP_THRESHOLD}
    cited = set(re.findall(r"C\d+", state["answer"]))
    assert cited <= kept
    assert state["answer"] == "True thing [C1]. Dropped thing. Invented thing."


def test_low_confidence_triggers_another_round_then_an_honest_answer(fake_llm):
    fake_llm.judge_confidence = 0.1                       # nothing passes
    state = run()

    assert state["round"] == MAX_ROUNDS                   # the Researcher tried again
    assert state["answer"].startswith("Not enough verified evidence")
    assert "FinalAnswer" not in fake_llm.calls            # the Writer never invented an answer

"""The confidence formula: final = 0.6 x judge + 0.4 x independent websites - skeptic penalty."""
import pytest

from crew.agents import calibrate
from crew.schemas import Claim, Objection, Source


def make(urls):
    sources = {f"S{i}": Source(id=f"S{i}", title="t", url=u, snippet="s") for i, u in enumerate(urls, 1)}
    claim = Claim(id="C1", text="A claim", source_ids=list(sources))
    return claim, sources


def objection(strength):
    return Objection(claim_id="C1", strength=strength, objection="...", source_ids=[])


def test_three_independent_sites_and_no_objection_gives_full_confidence():
    claim, sources = make(["https://a.org/1", "https://b.org/2", "https://c.org/3"])
    assert calibrate(1.0, "supported", claim, None, sources) == 1.0


def test_pages_from_the_same_website_count_once():
    claim, sources = make(["https://a.org/1", "https://a.org/2", "https://a.org/3"])
    # 0.6 * 1.0 + 0.4 * (1/3) = 0.73
    assert calibrate(1.0, "supported", claim, None, sources) == 0.73


@pytest.mark.parametrize("strength, expected", [("none", 1.0), ("weak", 0.85), ("strong", 0.6)])
def test_skeptic_objections_lower_the_score(strength, expected):
    claim, sources = make(["https://a.org", "https://b.org", "https://c.org"])
    assert calibrate(1.0, "supported", claim, objection(strength), sources) == expected


@pytest.mark.parametrize("verdict", ["contested", "refuted", "unverifiable"])
def test_claims_that_are_not_supported_never_pass(verdict):
    claim, sources = make(["https://a.org", "https://b.org", "https://c.org"])
    assert calibrate(1.0, verdict, claim, None, sources) <= 0.4


def test_score_stays_between_0_and_1_even_if_the_model_sends_nonsense():
    claim, sources = make(["https://a.org"])
    assert calibrate(5.0, "supported", claim, None, sources) == 1.0
    assert calibrate(-3.0, "supported", claim, objection("strong"), sources) == 0.0

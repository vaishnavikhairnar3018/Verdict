import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

# ---------- Data the code creates ----------

class Source(BaseModel):
    id: str          # "S1", "S2", ... so the LLM can cite it
    title: str
    url: str
    snippet: str


# ---------- Shapes the LLM must answer in (structured outputs) ----------

class SearchPlan(BaseModel):
    queries: list[str]


class DraftClaim(BaseModel):
    text: str
    source_ids: list[str]


class ClaimList(BaseModel):
    claims: list[DraftClaim]


class Claim(DraftClaim):
    id: str   # "C1", "C2", ... given by our code, not the LLM


class CounterQuery(BaseModel):
    claim_id: str
    queries: list[str]   # two short searches from different angles


class CounterPlan(BaseModel):
    items: list[CounterQuery]


class Objection(BaseModel):
    claim_id: str
    strength: Literal["none", "weak", "strong"]
    objection: str
    source_ids: list[str]


class ObjectionList(BaseModel):
    objections: list[Objection]


class JudgeVerdict(BaseModel):
    claim_id: str
    verdict: Literal["supported", "contested", "refuted", "unverifiable"]
    confidence: float   # the Judge's own 0-1 estimate
    reasoning: str


class VerdictList(BaseModel):
    verdicts: list[JudgeVerdict]


class ScoredVerdict(JudgeVerdict):
    final_confidence: float   # our calibrated score, computed in code


class FinalAnswer(BaseModel):
    answer: str   # cites claims like [C1]


# ---------- Shared memory of the whole crew (LangGraph state) ----------

class CrewState(TypedDict):
    question: str
    round: int
    # operator.add means "append what a node returns" instead of "replace"
    sources: Annotated[list[Source], operator.add]
    claims: Annotated[list[Claim], operator.add]
    objections: Annotated[list[Objection], operator.add]
    verdicts: Annotated[list[ScoredVerdict], operator.add]
    answer: str

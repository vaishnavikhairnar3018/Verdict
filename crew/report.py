import json
import re
from datetime import datetime
from pathlib import Path

from .config import KEEP_THRESHOLD
from .schemas import CrewState

REPORTS = Path("reports")


def to_plain(state: CrewState) -> dict:
    """Turn every Pydantic object into a plain dict, so the state can be saved as JSON."""
    return {k: [x.model_dump() for x in v] if isinstance(v, list) else v for k, v in state.items()}


def save(state: CrewState, usage: dict | None = None) -> Path:
    """Write the Markdown report and the raw JSON state. Returns the .md path."""
    REPORTS.mkdir(exist_ok=True)
    topic = re.sub(r"[^a-z0-9]+", "-", state["question"].lower()).strip("-")[:60]
    # A timestamp makes every run's filename unique, so asking again never overwrites.
    slug = f"{topic}-{datetime.now():%Y%m%d-%H%M%S}"
    md = REPORTS / f"{slug}.md"
    md.write_text(to_markdown(state), encoding="utf-8")
    # The raw state is saved too, so the dashboard can reload past runs.
    raw = {**to_plain(state), "usage": usage or {}}
    (REPORTS / f"{slug}.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return md


ICONS = {"supported": "✅", "contested": "⚖️", "refuted": "❌", "unverifiable": "❔"}


def to_markdown(state: CrewState) -> str:
    sources = {s.id: s for s in state["sources"]}
    claims = {c.id: c for c in state["claims"]}
    objections = {o.claim_id: o for o in state["objections"]}
    verdicts = sorted(state["verdicts"], key=lambda v: v.final_confidence, reverse=True)

    lines = [f"# {state['question']}", "", "## Answer", "", state["answer"], "",
             "## Every claim we checked", "",
             "| Claim | Verdict | Confidence | Sources |", "|---|---|---|---|"]
    for v in verdicts:
        c = claims[v.claim_id]
        links = ", ".join(f"[{i}]({sources[i].url})" for i in c.source_ids)
        mark = "" if v.final_confidence >= KEEP_THRESHOLD else " *(dropped)*"
        lines.append(f"| **{c.id}** {c.text}{mark} | {ICONS[v.verdict]} {v.verdict} "
                     f"| {v.final_confidence:.2f} | {links} |")

    lines += ["", "## Why each claim got its verdict", ""]
    for v in verdicts:
        o = objections.get(v.claim_id)
        lines.append(f"- **{v.claim_id}**: {v.reasoning}")
        if o and o.strength != "none":
            lines.append(f"  - Skeptic ({o.strength}): {o.objection}")

    lines += ["", "## Sources", ""]
    lines += [f"- **{s.id}** [{s.title}]({s.url})" for s in state["sources"]]
    return "\n".join(lines) + "\n"

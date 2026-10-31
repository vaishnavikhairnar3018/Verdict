import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from crew.config import FALLBACK_MODEL, KEEP_THRESHOLD, MAX_ROUNDS, MODEL
from crew.graph import build_graph, initial_state
from crew.llm import reset_usage, usage
from crew.report import REPORTS, save, to_plain

app = FastAPI(title="Verdict")


def sse(event: str, data) -> str:
    """Format one Server-Sent Event: the browser receives it the moment we yield it."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def summarize(agent: str, update: dict) -> str:
    """One short line describing what an agent just did."""
    if agent == "researcher":
        return f"Found {len(update.get('sources', []))} sources, wrote {len(update.get('claims', []))} claims"
    if agent == "skeptic":
        objections = update.get("objections", [])
        strong = sum(o.strength == "strong" for o in objections)
        return f"Challenged {len(objections)} claims, {strong} strong objections"
    if agent == "judge":
        verdicts = update.get("verdicts", [])
        kept = sum(v.final_confidence >= KEEP_THRESHOLD for v in verdicts)
        return f"{kept} of {len(verdicts)} claims passed"
    return "Answer written from verified claims"


def load(slug: str) -> dict:
    if not re.fullmatch(r"[a-z0-9-]+", slug):          # blocks paths like ../../secret
        raise HTTPException(404)
    path = REPORTS / f"{slug}.json"
    if not path.is_file():
        raise HTTPException(404)
    return {**json.loads(path.read_text(encoding="utf-8")), "slug": slug}


# ============================== API ==============================

@app.get("/api/config")
def config():
    return {"model": MODEL, "fallback": FALLBACK_MODEL, "threshold": KEEP_THRESHOLD,
            "max_rounds": MAX_ROUNDS}


@app.get("/api/runs")
def list_runs():
    runs = []
    for path in sorted(REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        run = load(path.stem)
        runs.append({
            "slug": path.stem,
            "question": run["question"],
            "claims": len(run["verdicts"]),
            "kept": sum(v["final_confidence"] >= KEEP_THRESHOLD for v in run["verdicts"]),
            "sources": len(run["sources"]),
            "time": path.stat().st_mtime,
        })
    return runs


@app.get("/api/runs/{slug}")
def get_run(slug: str):
    return load(slug)


@app.get("/api/runs/{slug}/report.md")
def get_report(slug: str):
    load(slug)   # validates the slug
    return FileResponse(REPORTS / f"{slug}.md", media_type="text/markdown", filename=f"{slug}.md")


@app.get("/api/research")
def research(q: str):
    """Run the crew and stream progress to the browser as it happens."""
    def events():
        reset_usage()
        state = None
        try:
            # "tasks" fires when an agent starts and when it finishes; "values" is the full state.
            for mode, chunk in build_graph().stream(initial_state(q), stream_mode=["tasks", "values"]):
                if mode == "values":
                    state = chunk
                elif "result" not in chunk:
                    yield sse("start", {"agent": chunk["name"]})
                else:
                    yield sse("finish", {"agent": chunk["name"],
                                         "text": summarize(chunk["name"], chunk["result"] or {})})
            report = save(state, usage)
            yield sse("done", load(report.stem))
        except Exception as e:
            yield sse("failed", {"message": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream")


# ============================== Web page ==============================

WEB = Path(__file__).parent / "web"   # works no matter which folder the server starts from
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")

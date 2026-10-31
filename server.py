import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from crew.config import FALLBACK_MODEL, KEEP_THRESHOLD, MAX_ROUNDS, MODEL
from crew.graph import build_graph, initial_state
from crew.llm import reset_usage, usage
from crew.report import REPORTS, save

app = FastAPI(title="Verdict")

# CORS: which websites may call this API from a browser. On Render, set ALLOWED_ORIGINS to
# your Vercel URL (e.g. https://verdict.vercel.app). "*" allows any site; fine for local use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET"],
)

# Abuse protection: a public site must not let strangers burn through your free Gemini quota.
RUNS_PER_HOUR = int(os.getenv("RUNS_PER_HOUR", "5"))   # per visitor (IP address)
recent_runs: dict[str, list[float]] = defaultdict(list)
crew_lock = threading.Lock()                            # one crew at a time


def client_ip(request: Request) -> str:
    # Behind Render's proxy the real visitor IP is the first entry of X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else request.client.host


def over_limit(ip: str) -> bool:
    """True if this visitor already ran RUNS_PER_HOUR questions in the last hour."""
    now = time.time()
    recent_runs[ip] = [t for t in recent_runs[ip] if now - t < 3600]
    if len(recent_runs[ip]) >= RUNS_PER_HOUR:
        return True
    recent_runs[ip].append(now)
    return False


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
def research(request: Request, q: str = Query(min_length=3, max_length=300)):
    """Run the crew and stream progress to the browser as it happens."""
    limited = over_limit(client_ip(request))

    def events():
        if limited:
            yield sse("failed", {"message": f"You've asked {RUNS_PER_HOUR} questions this hour. "
                                            "Please try again later (this demo runs on a free API quota)."})
            return
        # The lock is taken inside the generator, so it is only held while a run really happens.
        if not crew_lock.acquire(blocking=False):
            yield sse("failed", {"message": "The crew is busy with another question. Try again in a minute."})
            return
        try:
            reset_usage()
            state = None
            run_start = time.time()
            started: dict[str, float] = {}
            timings: dict[str, float] = defaultdict(float)   # seconds per agent (all rounds)
            # "tasks" fires when an agent starts and when it finishes; "values" is the full state.
            for mode, chunk in build_graph().stream(initial_state(q), stream_mode=["tasks", "values"]):
                if mode == "values":
                    state = chunk
                elif "result" not in chunk:
                    started[chunk["name"]] = time.time()
                    yield sse("start", {"agent": chunk["name"]})
                else:
                    seconds = time.time() - started.get(chunk["name"], time.time())
                    timings[chunk["name"]] += seconds
                    yield sse("finish", {"agent": chunk["name"], "seconds": round(seconds),
                                         "text": summarize(chunk["name"], chunk["result"] or {})})
            usage["timings"] = {name: round(s) for name, s in timings.items()}
            usage["total_seconds"] = round(time.time() - run_start)
            report = save(state, usage)
            yield sse("done", load(report.stem))
        except Exception as e:
            yield sse("failed", {"message": str(e)})
        finally:
            crew_lock.release()

    # no-cache + X-Accel-Buffering stop hosting proxies from holding events back until the end.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


# ============================== Web page ==============================

# Serve web/ at the site root (index.html, app.js, styles.css...). Mounted last so /api wins.
WEB = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

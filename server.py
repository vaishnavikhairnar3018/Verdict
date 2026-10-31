import json
import os
import queue
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
    return len(recent_runs[ip]) >= RUNS_PER_HOUR


def count_run(ip: str):
    """Only questions that actually start a run count; a 'busy' refusal costs nothing."""
    recent_runs[ip].append(time.time())


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
            "max_rounds": MAX_ROUNDS,
            # Render sets RENDER_GIT_COMMIT, so we can check which version is live.
            "version": os.getenv("RENDER_GIT_COMMIT", "local")[:7]}


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


HEARTBEAT = 10   # seconds of silence before we send a "still working" ping


def run_crew(question: str, updates: queue.Queue):
    """Run the whole crew in a background thread, putting progress events on `updates`.

    It always runs to the end and always releases the lock, even if the browser that asked
    has gone away, so one dropped connection can never block the server for everyone.
    """
    try:
        reset_usage()
        state = None
        run_start = time.time()
        started: dict[str, float] = {}
        timings: dict[str, float] = defaultdict(float)   # seconds per agent (all rounds)
        # "tasks" fires when an agent starts and when it finishes; "values" is the full state.
        for mode, chunk in build_graph().stream(initial_state(question), stream_mode=["tasks", "values"]):
            if mode == "values":
                state = chunk
            elif "result" not in chunk:
                started[chunk["name"]] = time.time()
                updates.put(("start", {"agent": chunk["name"]}))
            else:
                seconds = time.time() - started.get(chunk["name"], time.time())
                timings[chunk["name"]] += seconds
                updates.put(("finish", {"agent": chunk["name"], "seconds": round(seconds),
                                        "text": summarize(chunk["name"], chunk["result"] or {})}))
        usage["timings"] = {name: round(s) for name, s in timings.items()}
        usage["total_seconds"] = round(time.time() - run_start)
        report = save(state, usage)
        updates.put(("done", load(report.stem)))
    except Exception as e:
        updates.put(("failed", {"message": str(e)}))
    finally:
        crew_lock.release()


@app.get("/api/research")
def research(request: Request, q: str = Query(min_length=3, max_length=300)):
    """Start the crew and stream its progress to the browser as it happens."""
    ip = client_ip(request)
    limited = over_limit(ip)

    def events():
        if limited:
            yield sse("failed", {"message": f"You've asked {RUNS_PER_HOUR} questions this hour. "
                                            "Please try again later (this demo runs on a free API quota)."})
            return
        if not crew_lock.acquire(blocking=False):
            yield sse("failed", {"message": "The crew is busy with another question. Try again in a minute."})
            return
        count_run(ip)
        updates: queue.Queue = queue.Queue()
        # The worker thread owns the lock from here on and releases it when the run ends.
        threading.Thread(target=run_crew, args=(q, updates), daemon=True).start()
        while True:
            try:
                event, data = updates.get(timeout=HEARTBEAT)
            except queue.Empty:
                # An SSE comment: browsers ignore it, but it stops proxies from closing a
                # connection that has been quiet for a while (e.g. during a slow search).
                yield ": still working\n\n"
                continue
            yield sse(event, data)
            if event in ("done", "failed"):
                return

    # no-cache + X-Accel-Buffering stop hosting proxies from holding events back until the end.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


# ============================== Web page ==============================

# Serve web/ at the site root (index.html, app.js, styles.css...). Mounted last so /api wins.
WEB = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

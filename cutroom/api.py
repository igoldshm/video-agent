"""The web service: a creator-facing API and app, on top of the harness.

Creators see videos, shots and plain-language progress. Step keys, retries, eval
scores and costs stay server-side; the trace for a run is available to the operator
at /ops/{id}/trace.
"""
from __future__ import annotations

import hmac
import html
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .harness import Harness
from .trace import render

STATIC = os.path.join(os.path.dirname(__file__), "static")
STAGES = {"plan_video": "Planning your shots", "choose_models": "Setting things up",
          "create_reference": "Designing your main subject", "approve_reference": "Designing your main subject",
          "select_assets": "Setting things up", "assemble": "Putting the cut together",
          "evaluate_final": "Final check", "deliver": "Publishing"}
CREATOR_STATUS = {"pending": "making", "running": "making", "awaiting_approval": "ready",
                  "completed": "published", "failed": "failed", "budget_exceeded": "paused",
                  "interrupted": "paused"}


class NewVideo(BaseModel):
    brief: str = Field(min_length=8, max_length=600)
    seconds: int = Field(default=12, ge=6, le=30)


class Change(BaseModel):
    note: str = Field(min_length=2, max_length=300)
    remember: bool = False


class Note(BaseModel):
    note: str


class OptionalNote(BaseModel):
    note: str = Field(default="", max_length=300)


class Jobs:
    """Runs execute on a small thread pool. A run is never executed twice at once."""

    def __init__(self, workers: int, sync: bool):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.active: set[str] = set()
        self.lock = threading.Lock()
        self.sync = sync  # tests run jobs inline

    def busy(self, run_id: str) -> bool:
        with self.lock:
            return run_id in self.active

    def submit(self, run_id: str, fn, *args) -> bool:
        with self.lock:
            if run_id in self.active:
                return False
            self.active.add(run_id)

        def work():
            try:
                fn(*args)
            finally:
                with self.lock:
                    self.active.discard(run_id)

        work() if self.sync else self.pool.submit(work)
        return True


def creator_view(h: Harness, run: dict) -> dict:
    """Everything the app shows about one video, and nothing about how it was made."""
    run_id, steps = run["id"], {s["key"]: s for s in h.store.steps(run["id"])}
    plan = steps["plan"]["output"] if "plan" in steps else None
    status = CREATOR_STATUS.get(run["status"], "making")
    if run["status"] == "awaiting_approval" and run["state"].get("pending_scope") == "reference:approve":
        status = "review_reference"
    in_flight = None
    if status == "making":
        for e in reversed(h.store.events(run_id)):
            if e["kind"] == "run_start":  # anything older belongs to a previous pass over this run
                break
            if e["kind"] == "step_start":
                in_flight = None if e["key"] in steps else (e["key"], e["data"].get("tool"))
                break
    view = {"id": run_id, "status": status, "brief": run["brief"], "title": plan["title"] if plan else None,
            "stage": "Getting started" if status == "making" else None, "shots": [], "progress": 0.0,
            "final_url": None, "watch_url": f"/v/{run_id}" if status == "published" else None, "message": None,
            "reference": None}
    ref = steps.get(f"ref:r{run['state'].get('reference_rev', 0)}")
    if ref and ref["output"].get("ok"):
        view["reference"] = {"url": f"/media/{run_id}/{os.path.basename(ref['output']['path'])}",
                             "approved": "reference:approve" in run["state"]["granted"],
                             "notes": run["state"].get("reference_notes", [])}
    if not plan:
        if status == "failed":
            view["message"] = "We couldn't plan this video. Try again in a moment."
        return view

    evals_on = run["config"].get("evals", True)

    def accepted(shot_id: str, rev: int) -> str | None:
        best = None
        for key, s in steps.items():
            if key.startswith(f"gen:{shot_id}:r{rev}:a") and s["output"]["ok"]:
                ev = steps.get("eval:" + key[4:])
                if not evals_on or (ev and ev["output"].get("passed")):
                    best = s["output"]["path"]
        return best

    def url(path: str) -> str:
        return f"/media/{run_id}/{os.path.basename(path)}"

    total = len(plan["shots"])
    for n, shot in enumerate(plan["shots"], 1):
        revision = run["state"]["revisions"].get(shot["id"], {"rev": 0, "notes": []})
        current = accepted(shot["id"], revision["rev"])
        earlier = next((p for r in range(revision["rev"] - 1, -1, -1) if (p := accepted(shot["id"], r))), None)
        takes = sum(k.startswith(f"gen:{shot['id']}:r{revision['rev']}:a") for k in steps)
        working = bool(in_flight and in_flight[0].split(":")[1:2] == [shot["id"]])
        kept = run["state"].get("fallbacks", {}).get(shot["id"])
        flag = None
        if not current and not working and kept and kept["rev"] == revision["rev"] and status != "making":
            current, flag = kept["path"], kept["kind"]  # no take passed review; see Harness._fall_back
        if current:
            state = "ready"
        elif working:
            state = "retaking" if (takes > 0 or earlier) else "filming"
            verb = {"evaluate_clip": "Reviewing", "create_keyframe": "Setting up",
                    "evaluate_keyframe": "Setting up"}.get(in_flight[1], "Retaking" if state == "retaking" else "Filming")
            view["stage"] = f"{verb} shot {n} of {total}"
        elif status == "failed" and run["state"].get("failed_shot") == shot["id"]:
            state = "failed"
        else:
            state = "queued"
        view["shots"].append({"id": shot["id"], "number": n, "beat": shot["beat"], "description": shot["description"],
                              "seconds": shot["duration"], "state": state, "notes": revision["notes"],
                              "flag": flag,
                              "clip_url": url(current or earlier) if (current or earlier) else None})
    ready = sum(s["state"] == "ready" for s in view["shots"])
    view["progress"] = round(ready / total, 2)
    if in_flight and in_flight[1] in STAGES:
        view["stage"] = STAGES[in_flight[1]]
    if status in ("ready", "published"):
        cuts = [s for k, s in steps.items() if k.startswith("assemble:")]
        if cuts:
            view["final_url"] = url(cuts[-1]["output"]["path"])
    if status == "failed":
        bad = next((s["number"] for s in view["shots"] if s["state"] == "failed"), None)
        view["message"] = (f"We couldn't get shot {bad} right after several takes." if bad
                           else "Something went wrong while making this video.")
    if status == "paused":
        view["message"] = ("This video reached its spending limit before it was finished."
                           if run["status"] == "budget_exceeded" else "This video was interrupted.")
    return view


def create_app(harness: Harness | None = None, sync: bool = False) -> FastAPI:
    h = harness or Harness(os.environ.get("CUTROOM_HOME", ".cutroom"))
    jobs = Jobs(int(os.environ.get("CUTROOM_WORKERS", "2")), sync)
    access_code = os.environ.get("CUTROOM_ACCESS_CODE", "")
    daily_videos = int(os.environ.get("CUTROOM_DAILY_VIDEOS", "20"))
    daily_usd = float(os.environ.get("CUTROOM_DAILY_USD", "15"))
    app = FastAPI(title="cutroom", docs_url="/ops/docs", redoc_url=None, openapi_url="/ops/openapi.json")

    def guard(x_access_code: str | None = Header(default=None)) -> None:
        if access_code and not hmac.compare_digest(x_access_code or "", access_code):
            raise HTTPException(401, "Enter the access code to continue.")

    def get_run(run_id: str) -> dict:
        try:
            return h.store.get_run(run_id)
        except KeyError:
            raise HTTPException(404, "We can't find that video.") from None

    def today() -> float:
        return time.time() - time.time() % 86400

    # Pick up whatever a previous process left unfinished. Steps already recorded are not repeated.
    for run_id in h.store.unfinished_runs():
        jobs.submit(run_id, h.run, run_id)

    # ---- public ----------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/config")
    def config():
        cfg = h.default_config()
        return {"needs_code": bool(access_code), "live": h.live, "max_seconds": int(cfg.max_total_seconds)}

    @app.get("/media/{run_id}/{name}")
    def media_file(run_id: str, name: str):
        # run ids are unguessable; <video> tags cannot send the access header
        if not re.fullmatch(r"[0-9a-f]{16}", run_id) or not re.fullmatch(r"[A-Za-z0-9_]+\.(mp4|jpg)", name):
            raise HTTPException(404)
        base = os.path.join(h.home, "runs", run_id)
        folder = "stills" if name.endswith(".jpg") else "" if name.startswith("final_") else "clips"
        path = os.path.join(base, folder, name)
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg" if name.endswith(".jpg") else "video/mp4")

    @app.get("/v/{run_id}", response_class=HTMLResponse)
    def watch(run_id: str):
        path = os.path.join(h.home, "deliveries", f"{run_id}.mp4")
        if not re.fullmatch(r"[0-9a-f]{16}", run_id) or not os.path.isfile(path):
            raise HTTPException(404, "This video is not published.")
        view = creator_view(h, get_run(run_id))
        title = html.escape(view["title"] or "Video")
        return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,"
                f"initial-scale=1'><title>{title}</title><style>body{{margin:0;background:#111018;color:#f1eff7;"
                f"font:16px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:100vh}}"
                f"main{{width:min(960px,100% - 32px);padding-block:24px}}video{{width:100%;border-radius:8px;"
                f"background:#000}}h1{{font-size:20px;margin:14px 0 0}}</style><main>"
                f"<video controls playsinline src='/published/{run_id}.mp4'></video><h1>{title}</h1></main></html>")

    @app.get("/published/{name}")
    def published(name: str):
        if not re.fullmatch(r"[0-9a-f]{16}\.mp4", name):
            raise HTTPException(404)
        path = os.path.join(h.home, "deliveries", name)
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4")

    # ---- creator API -------------------------------------------------------------
    @app.post("/api/session", dependencies=[Depends(guard)])
    def session():
        return {"ok": True}

    @app.post("/api/videos", status_code=201, dependencies=[Depends(guard)])
    def new_video(body: NewVideo):
        if h.store.runs_since(today()) >= daily_videos or h.store.spent_since(today()) >= daily_usd:
            raise HTTPException(429, "Today's limit for new videos has been reached. Please come back tomorrow.")
        base = h.default_config()
        cfg = replace(base, target_seconds=min(base.max_total_seconds, body.seconds))
        run_id = h.create(body.brief.strip(), cfg)
        jobs.submit(run_id, h.run, run_id)
        return creator_view(h, get_run(run_id))

    @app.get("/api/videos", dependencies=[Depends(guard)])
    def my_videos(ids: str = Query(default="")):
        out = []
        for run_id in [i for i in ids.split(",") if re.fullmatch(r"[0-9a-f]{16}", i)][:30]:
            try:
                out.append(creator_view(h, h.store.get_run(run_id)))
            except KeyError:
                pass
        return out

    @app.get("/api/videos/{run_id}", dependencies=[Depends(guard)])
    def video(run_id: str):
        return creator_view(h, get_run(run_id))

    @app.post("/api/videos/{run_id}/shots/{shot_id}/change", dependencies=[Depends(guard)])
    def change(run_id: str, shot_id: str, body: Change):
        run = get_run(run_id)
        view = creator_view(h, run)
        if shot_id not in [s["id"] for s in view["shots"]]:
            raise HTTPException(404, "We can't find that shot.")
        if jobs.busy(run_id) or view["status"] == "making":
            raise HTTPException(409, "Still working on this video. Try again when it's ready.")
        h.request_change(run_id, shot_id, body.note.strip(), body.remember)
        jobs.submit(run_id, h.run, run_id)
        return creator_view(h, get_run(run_id))

    @app.post("/api/videos/{run_id}/retry", dependencies=[Depends(guard)])
    def retry(run_id: str):
        run = get_run(run_id)
        if jobs.busy(run_id) or run["status"] not in ("failed", "interrupted"):
            raise HTTPException(409, "This video can't be retried right now.")
        if run["state"].get("failed_shot"):  # fresh takes for the shot that failed; everything else is kept
            h.request_change(run_id, run["state"]["failed_shot"], revoke_publish=False)
        jobs.submit(run_id, h.run, run_id)
        return creator_view(h, get_run(run_id))

    @app.post("/api/videos/{run_id}/publish", dependencies=[Depends(guard)])
    def publish(run_id: str):
        run = get_run(run_id)
        if (jobs.busy(run_id) or run["status"] != "awaiting_approval"
                or run["state"].get("pending_scope") != "deliver:publish"):
            raise HTTPException(409, "This video isn't ready to publish yet.")
        jobs.submit(run_id, h.approve, run_id)
        return creator_view(h, get_run(run_id))

    @app.post("/api/videos/{run_id}/reference/approve", dependencies=[Depends(guard)])
    def approve_reference(run_id: str):
        run = get_run(run_id)
        if (jobs.busy(run_id) or run["status"] != "awaiting_approval"
                or run["state"].get("pending_scope") != "reference:approve"):
            raise HTTPException(409, "There is no look waiting for your approval.")
        jobs.submit(run_id, h.approve, run_id)
        return creator_view(h, get_run(run_id))

    @app.post("/api/videos/{run_id}/reference/redo", dependencies=[Depends(guard)])
    def redo_reference(run_id: str, body: OptionalNote):
        view = creator_view(h, get_run(run_id))
        if jobs.busy(run_id) or view["status"] == "making" or not view["reference"]:
            raise HTTPException(409, "The look can't be changed right now.")
        h.redo_reference(run_id, body.note.strip())
        jobs.submit(run_id, h.run, run_id)
        return creator_view(h, get_run(run_id))

    @app.get("/api/preferences", dependencies=[Depends(guard)])
    def preferences():
        return h.store.preferences(h.default_config().project)

    @app.post("/api/preferences/remove", dependencies=[Depends(guard)])
    def remove_preference(body: Note):
        h.store.remove_preference(h.default_config().project, body.note)
        return h.store.preferences(h.default_config().project)

    # ---- operator ------------------------------------------------------------------
    @app.get("/ops/{run_id}/trace", response_class=HTMLResponse)
    def trace(run_id: str, code: str = ""):
        guard(code)
        get_run(run_id)
        return render(h.store, run_id)

    return app

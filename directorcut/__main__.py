"""Command line: python -m directorcut <command>"""
from __future__ import annotations

import argparse
import os

from .harness import PRESETS, Harness
from .settings import env


def summary(h: Harness, run: dict) -> None:
    print(f"run {run['id']}  status={run['status']}  cost=${run['cost_usd']:.2f}  "
          f"steps={len(h.store.steps(run['id']))}")
    if run["status"] == "awaiting_approval":
        print(f"  needs scope '{run['state']['pending_scope']}': python -m directorcut approve {run['id']}")
    last = h.store.events(run["id"])[-1]["data"]
    if last.get("detail") and run["status"] != "awaiting_approval":
        print(f"  {last['detail']}")
    delivered = os.path.join(h.home, "deliveries", f"{run['id']}.mp4")
    if run["status"] == "completed":
        print(f"  video: {delivered}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="directorcut")
    ap.add_argument("--home", default=env("HOME", ".directorcut"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="start a run from a brief")
    p.add_argument("brief")
    p.add_argument("--preset", choices=list(PRESETS), help="simulated-model presets; default uses what is connected")
    p.add_argument("--auto-approve", action="store_true", help="grant deliver:publish up front")
    p.add_argument("--crash-after", type=int, help="simulate the process dying after N steps")
    for name in ("resume", "approve", "trace"):
        sub.add_parser(name).add_argument("run_id")
    p = sub.add_parser("feedback", help="ask for a change to one shot")
    p.add_argument("run_id")
    p.add_argument("shot_id")
    p.add_argument("note")
    p.add_argument("--remember", action="store_true", help="also apply to future videos in this project")
    sub.add_parser("bench", help="run the benchmark across harness versions")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    if a.cmd == "bench":
        from .bench import main as bench_main
        os.makedirs(os.path.join(a.home, "bench"), exist_ok=True)
        return bench_main(os.path.join(a.home, "bench"))
    if a.cmd == "serve":
        import uvicorn

        from .api import create_app
        return uvicorn.run(create_app(Harness(a.home)), host="0.0.0.0", port=a.port)

    h = Harness(a.home)
    if a.cmd == "run":
        rid = h.create(a.brief, PRESETS[a.preset] if a.preset else None, a.auto_approve)
        summary(h, h.run(rid, crash_after=a.crash_after))
    elif a.cmd == "resume":
        summary(h, h.run(a.run_id))
    elif a.cmd == "approve":
        summary(h, h.approve(a.run_id))
    elif a.cmd == "feedback":
        summary(h, h.feedback(a.run_id, a.shot_id, a.note, a.remember))
    elif a.cmd == "trace":
        from .trace import render
        out = os.path.join(h.home, f"trace_{a.run_id}.html")
        with open(out, "w") as f:
            f.write(render(h.store, a.run_id))
        print(out)


if __name__ == "__main__":
    main()

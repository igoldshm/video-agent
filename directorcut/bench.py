"""Regression benchmark: the same briefs through each harness configuration.

Because the mock providers draw failures from the brief (not the run id), every
configuration faces the same sequence of provider errors and defects, so the
difference between rows is the harness and nothing else.

A delivered video counts as "clean" only if every clip in the final cut passes
the full check suite, run here after the fact and independently of whether the
configuration under test used evals.
"""
from __future__ import annotations

import json
import os
import time

from . import evals
from .harness import PRESETS, Harness
from .planner import HeuristicPlanner
from .providers import demo_providers

BRIEFS = [
    "A 12s calm launch video for a sleep tea brand",
    "15 sec energetic teaser for a trail running shoe",
    "Premium 9s reveal of a handcrafted gold watch",
    "A 12s explainer for a budgeting app for students",
    "18s bold fitness challenge announcement",
    "Gentle 9 sec promo for a meditation retreat",
    "A 15s product tour of a smart water bottle",
    "Elegant 12s teaser for a luxury candle line",
    "A 9s hiring ad for a neighbourhood bakery",
    "Fast 12 sec launch clip for a cycling app",
    "A 15s welcome video for an online pottery class",
    "Soft 12s story for a baby skincare range",
]


def run_benchmark(home: str, briefs: list[str] | None = None) -> dict:
    briefs = briefs or BRIEFS
    rows = []
    for preset, cfg in PRESETS.items():
        h = Harness(os.path.join(home, preset), providers=demo_providers(), planner=HeuristicPlanner(), judge=None)
        delivered = clean = gens = 0
        cost = 0.0
        t0 = time.time()
        for brief in briefs:
            rid = h.create(brief, cfg, auto_approve=True)
            run = h.run(rid)
            cost += run["cost_usd"]
            steps = {s["key"]: s for s in h.store.steps(rid)}
            gens += sum(k.startswith("gen:") for k in steps)
            if run["status"] != "completed":
                continue
            delivered += 1
            plan = steps["plan"]["output"]
            style, ok = plan["style"], True
            for shot, path in _accepted_clips(steps, plan):
                res = evals.evaluate_clip(path, shot["duration"], cfg.width, cfg.height, style["palette"],
                                          cfg.palette_tolerance)
                ok = ok and res["passed"]
            clean += ok
        n = len(briefs)
        rows.append({
            "config": preset, "briefs": n, "delivered": delivered, "clean_deliveries": clean,
            "defective_deliveries": delivered - clean,
            "clean_rate": round(clean / n, 3), "generations": gens,
            "cost_usd": round(cost, 2),
            "cost_per_clean_video_usd": round(cost / clean, 2) if clean else None,
            "wall_s": round(time.time() - t0, 1),
        })
    return {"briefs": len(briefs), "providers": "simulated (seeded failure injection)", "rows": rows}


def _accepted_clips(steps: dict, plan: dict) -> list[tuple[dict, str]]:
    out = []
    for shot in plan["shots"]:
        gens = sorted((k for k in steps if k.startswith(f"gen:{shot['id']}:") and steps[k]["output"]["ok"]),
                      key=lambda k: int(k.rsplit(":a", 1)[1]))
        out.append((shot, steps[gens[-1]]["output"]["path"]))
    return out


def to_markdown(result: dict) -> str:
    head = "| Config | Delivered | Clean | Shipped with a defect | Generations | Cost | Cost per clean video |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        per = f"${r['cost_per_clean_video_usd']:.2f}" if r["cost_per_clean_video_usd"] else "n/a"
        lines.append(f"| {r['config']} | {r['delivered']}/{r['briefs']} | {r['clean_deliveries']}/{r['briefs']} "
                     f"| {r['defective_deliveries']} | {r['generations']} | ${r['cost_usd']:.2f} | {per} |")
    return "\n".join(lines)


def main(home: str) -> None:
    result = run_benchmark(home)
    with open(os.path.join(home, "bench.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(to_markdown(result))

"""What the harness learns from its own eval history.

Every clip evaluation is already recorded as a step. This module reads those records
across all runs and turns them into two decisions for the next run:

- routing: once each model has enough evaluated clips, try models in order of
  expected cost per accepted clip (price / pass rate) instead of the configured order;
- guards: an issue a model keeps being rejected for becomes a standing instruction
  in every prompt sent to that model, instead of something fixed one retry at a time.

The result is computed once per run and checkpointed, so a resumed run does not
change course because other runs finished in the meantime.
"""
from __future__ import annotations

from collections import Counter

from .evals import REVISION_HINTS
from .store import Store

MIN_CLIPS_TO_ROUTE = 5   # per model, before history overrides the configured order
MIN_FAILS_FOR_GUARD = 3
MIN_RATE_FOR_GUARD = 0.2


def model_stats(store: Store) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for ev in store.steps_for_tool("evaluate_clip"):
        name = ev.get("provider")
        if not name or not ev.get("ok"):
            continue
        s = stats.setdefault(name, {"clips": 0, "passed": 0, "issues": Counter()})
        s["clips"] += 1
        s["passed"] += bool(ev.get("passed"))
        s["issues"].update(ev.get("failures", []))
    return stats


def choose_models(store: Store, configured: list[str], prices: dict[str, float]) -> dict:
    stats = model_stats(store)
    basis = {p: {"clips": stats.get(p, {}).get("clips", 0),
                 "pass_rate": round(stats[p]["passed"] / stats[p]["clips"], 3) if p in stats else None}
             for p in configured}
    order, reason = list(configured), "configured order (not enough history yet)"
    if all(basis[p]["clips"] >= MIN_CLIPS_TO_ROUTE for p in configured) and len(configured) > 1:
        def cost_per_accepted(p: str) -> float:
            return prices[p] / max(basis[p]["pass_rate"], 0.05)
        ranked = sorted(configured, key=cost_per_accepted)
        if ranked != order:
            order, reason = ranked, "reordered by expected cost per accepted clip"
        else:
            reason = "history agrees with the configured order"
    guards = {}
    for p in configured:
        s = stats.get(p)
        if not s:
            continue
        hints = [REVISION_HINTS[tag] for tag, n in s["issues"].most_common()
                 if tag in REVISION_HINTS and n >= MIN_FAILS_FOR_GUARD and n / s["clips"] >= MIN_RATE_FOR_GUARD]
        if hints:
            guards[p] = hints[:3]
    return {"order": order, "reason": reason, "basis": basis, "guards": guards}

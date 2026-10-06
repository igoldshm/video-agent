"""Planners turn a brief into a shot list plus the creative context every shot shares.

The creative context is the part of the plan that is not a shot: the look, the
recurring subject, the palette and the creator's standing preferences. It is written
into every generation prompt and given to the judge, which is how shot 4 stays in the
same world as shot 1.
"""
from __future__ import annotations

import os
import re
import textwrap

from .llm import Claude

PALETTES = {
    "calm": (["#1f5f8b", "#9fd3c7"], ("calm", "relax", "sleep", "gentle", "soft", "wellness", "meditat")),
    "energetic": (["#ff5e3a", "#ffb400"], ("energy", "sport", "fast", "launch", "bold", "fitness", "run")),
    "premium": (["#2b2d42", "#c9a227"], ("premium", "luxury", "elegant", "craft", "gold", "watch")),
    "fresh": (["#2a9d8f", "#e9c46a"], ()),
}
BEATS = ["Hook", "Problem", "Reveal", "Proof", "Detail", "Call to action"]

PLAN_SCHEMA = {
    "type": "object",
    "required": ["title", "style", "subject", "shots"],
    "properties": {
        "title": {"type": "string", "description": "Short title for the video, at most 8 words."},
        "subject": {"type": "string", "description": "The recurring subject or character, described concretely "
                    "enough (appearance, colours, materials) that every shot can depict the same one."},
        "style": {"type": "object", "required": ["mood", "palette", "look"], "properties": {
            "mood": {"type": "string"},
            "palette": {"type": "array", "items": {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"},
                        "minItems": 2, "maxItems": 2},
            "look": {"type": "string", "description": "One sentence: lighting, lens, colour grade, medium."}}},
        "shots": {"type": "array", "minItems": 2, "maxItems": 6, "items": {
            "type": "object", "required": ["beat", "description", "duration"], "properties": {
                "beat": {"type": "string", "description": "Two or three words naming the shot's role."},
                "description": {"type": "string", "description": "What is seen and how the camera moves. No text "
                                "overlays, no dialogue."},
                "duration": {"type": "number"}}}},
    },
}


def _finish(plan: dict, preferences: list[str], max_shot_seconds: float, max_total_seconds: float) -> dict:
    plan["preferences"] = preferences
    plan["shots"] = plan["shots"][:6]
    budget = max_total_seconds
    kept = []
    for i, shot in enumerate(plan["shots"], 1):
        shot["id"] = f"s{i}"
        shot["duration"] = round(min(max_shot_seconds, max(2.0, float(shot["duration"])), max(budget, 0)), 2)
        if shot["duration"] >= 2.0:
            kept.append(shot)
            budget -= shot["duration"]
    plan["shots"] = kept
    return plan


class HeuristicPlanner:
    """Deterministic, offline planner used in tests and demo mode."""

    name = "heuristic"

    def plan(self, brief: str, preferences: list[str], max_shot_seconds: float = 6.0,
             max_total_seconds: float = 30.0, target_seconds: float = 0) -> dict:
        text = brief.lower()
        m = re.search(r"(\d+)\s*(?:s\b|sec)", text)
        total = min(max_total_seconds, target_seconds or (max(6, int(m.group(1))) if m else 12))
        n = max(3, min(6, int(total // 3)))
        beats = BEATS[: n - 1] + [BEATS[-1]]
        mood = next((k for k, (_, words) in PALETTES.items() if any(w in text for w in words)), "fresh")
        subject = re.sub(r"\s+", " ", brief).strip().rstrip(".")
        return _finish({
            "title": textwrap.shorten(subject, 60, placeholder="…"), "subject": subject,
            "style": {"mood": mood, "palette": PALETTES[mood][0], "look": f"{mood} colour grade, soft light"},
            "shots": [{"beat": b, "description": f"{b} for: {subject}", "duration": round(total / n, 2)}
                      for b in beats],
        }, preferences, max_shot_seconds, max_total_seconds)


class ClaudePlanner:
    """LLM planner with schema-constrained output."""

    name = "claude"
    SYSTEM = (
        "You are a director planning a short video for a text-to-video model. Break the brief into shots that "
        "tell one clear story. Every shot is generated separately by a model that sees only that shot's "
        "description plus the shared subject and look, so describe the subject identically each time and "
        "never refer to 'the same' or 'previous' anything. Avoid on-screen text, logos and dialogue. "
        "Honour the creator's standing preferences.")

    def __init__(self, llm: Claude | None = None):
        self.llm = llm or Claude()

    def plan(self, brief: str, preferences: list[str], max_shot_seconds: float = 6.0,
             max_total_seconds: float = 30.0, target_seconds: float = 0) -> dict:
        prefs = "\n".join(f"- {p}" for p in preferences) or "(none)"
        ask = (f"Brief: {brief}\n\nCreator's standing preferences:\n{prefs}\n\n"
               f"Limits: each shot between 2 and {max_shot_seconds:g} seconds, "
               f"{max_total_seconds:g} seconds in total at most."
               + (f" Aim for about {target_seconds:g} seconds in total." if target_seconds else ""))
        plan = self.llm.tool_call(self.SYSTEM, ask, "submit_plan", PLAN_SCHEMA)
        plan["style"].setdefault("palette", PALETTES["fresh"][0])
        return _finish(plan, preferences, max_shot_seconds, max_total_seconds)


def default_planner():
    return ClaudePlanner() if os.environ.get("ANTHROPIC_API_KEY") else HeuristicPlanner()

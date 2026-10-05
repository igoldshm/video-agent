"""Quality checks on generated media.

Two layers. Technical checks read the pixels and container of the file and catch
broken output for free. The vision judge looks at sampled frames and decides whether
the clip shows what the shot asked for, in the look the project established. Neither
trusts what a provider says it produced. Every result carries measured values and
issue tags, so a rejection is explainable and can steer the next attempt.
"""
from __future__ import annotations

from . import media
from .llm import Claude, image_block


def _check(name: str, passed: bool, value: float, threshold: str) -> dict:
    return {"name": name, "passed": bool(passed), "value": round(float(value), 3), "threshold": threshold}


def _luma(frame: bytes) -> float:
    px = len(frame) // 3
    return sum(0.299 * frame[i] + 0.587 * frame[i + 1] + 0.114 * frame[i + 2]
               for i in range(0, len(frame), 3)) / px


def _mean_rgb(frame: bytes) -> tuple[float, float, float]:
    px = len(frame) // 3
    return tuple(sum(frame[c::3]) / px for c in range(3))  # type: ignore[return-value]


def _diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _distance_to_palette(rgb: list[float], palette: list[str]) -> float:
    """Distance from a colour to the nearest blend of the two palette colours, so any
    mix of the palette counts as on-style and only a drift away from it is penalised."""
    a, b = (media.hex_rgb(c) for c in palette[:2])
    ab = [y - x for x, y in zip(a, b)]
    t = sum((p - x) * d for p, x, d in zip(rgb, a, ab)) / (sum(d * d for d in ab) or 1)
    t = min(1.0, max(0.0, t))
    return sum((p - (x + t * d)) ** 2 for p, x, d in zip(rgb, a, ab)) ** 0.5


def evaluate_clip(path: str, duration: float, width: int, height: int, palette: list[str] | None = None,
                  palette_tolerance: float = 40.0) -> dict:
    """Technical checks. `palette` is only checked when given: it suits flat graphic
    output, not photographic footage, where the judge assesses style instead."""
    info = media.probe(path)
    frames = media.thumb_frames(path)
    checks = [
        _check("duration", abs(info["duration"] - duration) <= 0.15 * duration, info["duration"],
               f"{duration}s +/-15%"),
        _check("resolution", (info["width"], info["height"]) == (width, height),
               info["width"] * 10000 + info["height"], f"{width}x{height}"),
    ]
    if frames:
        black = sum(_luma(f) < 16 for f in frames) / len(frames)
        checks.append(_check("no_black_frames", black <= 0.2, black, "<=20% black frames"))
        mid = frames[len(frames) // 2]
        motion = max(_diff(frames[0], mid), _diff(mid, frames[-1]), _diff(frames[0], frames[-1]))
        checks.append(_check("has_motion", motion >= 2.0, motion, ">=2.0 mean pixel change"))
        if palette:
            per_frame = [_mean_rgb(f) for f in frames]  # averaged over the clip, not one frame
            got = [sum(m[i] for m in per_frame) / len(per_frame) for i in range(3)]
            dist = _distance_to_palette(got, palette)
            checks.append(_check("style_continuity", dist <= palette_tolerance, dist,
                                 f"<={palette_tolerance} RGB distance from project palette"))
    else:
        checks.append(_check("decodable", False, 0, "at least one frame"))
    failures = [c["name"] for c in checks if not c["passed"]]
    return {"passed": not failures, "failures": failures, "checks": checks,
            "score": round(sum(c["passed"] for c in checks) / len(checks), 3)}


def evaluate_final(path: str, expected_duration: float, width: int, height: int, expect_audio: bool) -> dict:
    info = media.probe(path)
    checks = [
        _check("duration", abs(info["duration"] - expected_duration) <= 0.1 * expected_duration + 0.2,
               info["duration"], f"{expected_duration}s +/-10%"),
        _check("resolution", (info["width"], info["height"]) == (width, height),
               info["width"] * 10000 + info["height"], f"{width}x{height}"),
        _check("audio_track", info["has_audio"] == expect_audio, info["has_audio"], f"audio={expect_audio}"),
    ]
    failures = [c["name"] for c in checks if not c["passed"]]
    return {"passed": not failures, "failures": failures, "checks": checks}


# ---------------------------------------------------------------------------
# Vision judge
# ---------------------------------------------------------------------------
JUDGE_TAGS = ["off_brief", "style_drift", "artifacts", "garbled_text", "bad_anatomy", "too_dark"]

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["matches_shot", "visual_quality", "consistent_with_reference", "issues", "note"],
    "properties": {
        "matches_shot": {"type": "integer", "minimum": 1, "maximum": 5,
                         "description": "5 = shows exactly the described subject and action; 1 = unrelated."},
        "visual_quality": {"type": "integer", "minimum": 1, "maximum": 5,
                           "description": "5 = clean and stable; 1 = severe warping, flicker or melting."},
        "consistent_with_reference": {"type": "integer", "minimum": 1, "maximum": 5,
                                      "description": "Same subject and look as the reference frame. Use 5 when "
                                                     "no reference frame was given."},
        "issues": {"type": "array", "items": {"type": "string", "enum": JUDGE_TAGS}},
        "note": {"type": "string", "description": "One sentence on the most important problem, or 'ok'."},
    },
}


class VisionJudge:
    """Shows a vision model three frames of the clip and asks if it is the shot we wanted."""

    SYSTEM = ("You review single shots of an AI-generated video before they go into the final cut. Judge only "
              "what is visible in the frames. Be strict about the subject being the right one and about visual "
              "defects, lenient about framing and small stylistic choices.")
    PASS_AT = 3

    def __init__(self, llm: Claude | None = None):
        self.llm = llm or Claude()

    def judge(self, path: str, shot: str, subject: str, look: str, reference_path: str | None = None) -> dict:
        content: list[dict] = [{"type": "text", "text":
                                f"Shot wanted: {shot}\nRecurring subject: {subject}\nLook: {look}\n\n"
                                "Three frames from the generated clip, in order:"}]
        content += [image_block(f) for f in media.jpeg_frames(path, 3)]
        if reference_path:
            content.append({"type": "text", "text": "Reference frame from an earlier accepted shot of this video:"})
            content.append(image_block(media.jpeg_frames(reference_path, 1)[0]))
        verdict = self.llm.tool_call(self.SYSTEM, content, "submit_review", JUDGE_SCHEMA, max_tokens=400)
        scores = {k: int(verdict[k]) for k in ("matches_shot", "visual_quality", "consistent_with_reference")}
        issues = [t for t in verdict.get("issues", []) if t in JUDGE_TAGS]
        passed = all(v >= self.PASS_AT for v in scores.values())
        if not passed and not issues:  # make sure every rejection carries an actionable tag
            issues = ["off_brief" if scores["matches_shot"] < self.PASS_AT else
                      "style_drift" if scores["consistent_with_reference"] < self.PASS_AT else "artifacts"]
        return {"passed": passed, "scores": scores, "issues": issues if not passed else [],
                "note": str(verdict.get("note", ""))[:300]}


# What the agent adds to the next prompt after a given failure (controllability).
REVISION_HINTS = {
    "style_continuity": "Stay strictly within the project colour palette.",
    "has_motion": "Include clear subject or camera motion throughout.",
    "no_black_frames": "Well lit from the first frame; no fades from black.",
    "duration": "Fill the full requested duration.",
    "off_brief": "Show exactly the described subject and action, clearly in frame.",
    "style_drift": "Match the established subject, colours and lighting precisely.",
    "artifacts": "Clean, stable image with no warping, flicker or melting shapes.",
    "garbled_text": "No on-screen text, logos or lettering anywhere.",
    "bad_anatomy": "Natural, correct hands, faces and body proportions.",
    "too_dark": "Bright, well-exposed lighting.",
}

"""The tools the agent can call. Each has a typed input and the scopes it needs."""
from __future__ import annotations

import json
import os
import shutil
import time

from pydantic import BaseModel, Field

from . import evals, learning, media
from .providers import GenRequest, Provider
from .registry import CallInfo, Registry, Tool, ToolFailed
from .store import Store

DEMO_BEDS = [  # id, base frequency of the placeholder tone, tags
    ("bed_calm", 196, ("calm", "soft", "slow")),
    ("bed_energetic", 330, ("energetic", "fast", "bold")),
    ("bed_premium", 147, ("premium", "elegant", "slow")),
    ("bed_fresh", 262, ("fresh", "bright", "friendly")),
]


class PlanIn(BaseModel):
    brief: str = Field(min_length=3)
    preferences: list[str] = []
    max_shot_seconds: float = 6.0
    max_total_seconds: float = 30.0
    target_seconds: float = 0


class ChooseModelsIn(BaseModel):
    providers: list[str] = Field(min_length=1)


class AssetsIn(BaseModel):
    mood: str


class GenerateIn(BaseModel):
    provider: str
    prompt: str
    duration: float = Field(gt=0, le=30)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    palette: list[str] = Field(min_length=2)
    seed: int = 0
    shot_id: str
    noise_key: str = ""
    out_path: str


class EvalClipIn(BaseModel):
    path: str
    provider: str = ""
    duration: float
    width: int
    height: int
    palette: list[str] | None = None
    palette_tolerance: float = 40.0
    judge: bool = False
    shot: str = ""
    subject: str = ""
    look: str = ""
    reference_path: str | None = None


class AssembleIn(BaseModel):
    clips: list[str] = Field(min_length=1)
    audio: str | None = None
    out_path: str


class EvalFinalIn(BaseModel):
    path: str
    expected_duration: float
    width: int
    height: int
    expect_audio: bool


class DeliverIn(BaseModel):
    path: str
    run_id: str
    title: str


def build_registry(providers: dict[str, Provider], planner, judge, store: Store, home: str,
                   backoff_s: float = 0.0) -> Registry:
    reg = Registry(backoff_s=backoff_s)
    simulated = all(p.simulated for p in providers.values())

    def plan_video(inp: PlanIn, call: CallInfo) -> dict:
        return planner.plan(inp.brief, inp.preferences, inp.max_shot_seconds, inp.max_total_seconds,
                            inp.target_seconds)

    def choose_models(inp: ChooseModelsIn, call: CallInfo) -> dict:
        unknown = [p for p in inp.providers if p not in providers]
        if unknown:
            raise ToolFailed(f"unknown providers {unknown}")
        return learning.choose_models(store, inp.providers, {n: p.price_per_s for n, p in providers.items()})

    def select_assets(inp: AssetsIn, call: CallInfo) -> dict:
        """Pick a music bed by mood from the asset library (assets/library.json: a list
        of {"file", "tags"}). With no library, demo mode synthesises a tone and live
        mode ships a silent cut rather than inventing music."""
        library = os.path.join(home, "assets", "library.json")
        if os.path.exists(library):
            with open(library) as f:
                items = json.load(f)
            scored = sorted(items, key=lambda a: inp.mood in a.get("tags", []), reverse=True)
            if scored:
                pick = scored[0]
                return {"audio": os.path.join(home, "assets", pick["file"]), "asset_id": pick["file"],
                        "reason": f"tags {pick.get('tags', [])} against mood '{inp.mood}'"}
        if not simulated:
            return {"audio": None, "asset_id": None, "reason": "asset library is empty"}
        asset_id, freq, tags = max(DEMO_BEDS, key=lambda a: inp.mood in a[2])
        path = os.path.join(home, "assets", f"{asset_id}.m4a")
        if not os.path.exists(path):
            media.tone_bed(path, freq)
        return {"audio": path, "asset_id": asset_id, "reason": f"tags {list(tags)} against mood '{inp.mood}'"}

    def generate_clip(inp: GenerateIn, call: CallInfo) -> dict:
        if inp.provider not in providers:
            raise ToolFailed(f"unknown provider '{inp.provider}'")
        provider = providers[inp.provider]
        req = GenRequest(prompt=inp.prompt, duration=inp.duration, width=inp.width, height=inp.height,
                         palette=inp.palette, seed=inp.seed, shot_id=inp.shot_id,
                         noise_key=inp.noise_key, attempt=call.attempt)
        if provider.simulated:
            provider.generate(req, inp.out_path)
        else:
            raw = inp.out_path + ".raw.mp4"
            provider.generate(req, raw)
            try:  # every model returns a different size, rate and length: bring it to the project's
                media.normalize(raw, inp.out_path, inp.width, inp.height, inp.duration)
            except RuntimeError as e:
                raise ToolFailed(f"{provider.name}: returned a file that could not be decoded ({e})") from e
            finally:
                if os.path.exists(raw):
                    os.remove(raw)
        return {"path": inp.out_path, "provider": provider.name,
                "cost_usd": round(provider.price_per_s * inp.duration, 4)}

    def evaluate_clip(inp: EvalClipIn, call: CallInfo) -> dict:
        res = evals.evaluate_clip(inp.path, inp.duration, inp.width, inp.height, inp.palette, inp.palette_tolerance)
        res["provider"] = inp.provider
        res["judge"] = None
        if res["passed"] and inp.judge and judge is not None:  # only spend a judge call on a clip that plays
            verdict = judge.judge(inp.path, inp.shot, inp.subject, inp.look, inp.reference_path)
            res["judge"] = verdict
            if not verdict["passed"]:
                res["passed"], res["failures"] = False, verdict["issues"]
        res["hints"] = [evals.REVISION_HINTS[f] for f in res["failures"] if f in evals.REVISION_HINTS]
        return res

    def assemble(inp: AssembleIn, call: CallInfo) -> dict:
        media.concat(inp.clips, inp.audio, inp.out_path)
        return {"path": inp.out_path}

    def evaluate_final(inp: EvalFinalIn, call: CallInfo) -> dict:
        return evals.evaluate_final(**inp.model_dump())

    def deliver(inp: DeliverIn, call: CallInfo) -> dict:
        # Publishing makes the cut available at its public watch link.
        dest_dir = os.path.join(home, "deliveries")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{inp.run_id}.mp4")
        shutil.copyfile(inp.path, dest)
        with open(dest + ".json", "w") as f:
            json.dump({"run_id": inp.run_id, "title": inp.title, "delivered_at": time.time()}, f)
        return {"delivered_to": dest}

    for name, model, fn, scopes, desc in [
        ("plan_video", PlanIn, plan_video, ("plan",), "Turn a brief into creative context and a shot list."),
        ("choose_models", ChooseModelsIn, choose_models, ("plan",),
         "Order the video models and set prompt guards from eval history."),
        ("select_assets", AssetsIn, select_assets, ("assets:read",), "Pick library assets that fit the mood."),
        ("generate_clip", GenerateIn, generate_clip, ("generate",), "Generate one clip with a named model."),
        ("evaluate_clip", EvalClipIn, evaluate_clip, ("evaluate",), "Technical checks, then the vision judge."),
        ("assemble", AssembleIn, assemble, ("media:process",), "Join the chosen clips and lay the audio bed."),
        ("evaluate_final", EvalFinalIn, evaluate_final, ("evaluate",), "Check the final cut."),
        ("deliver", DeliverIn, deliver, ("deliver:publish",), "Publish the final cut. Needs creator approval."),
    ]:
        reg.register(Tool(name, model, fn, scopes, desc))
    return reg

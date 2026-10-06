"""The agent harness: a checkpointed, permissioned, budgeted loop around the tools.

plan -> choose models -> select assets -> per shot: generate -> evaluate -> revise/fail over
     -> assemble -> evaluate final cut -> deliver (gated on creator approval)

Every tool call goes through RunContext.step(), which records its output under a
stable key. Re-running a run replays recorded steps instead of re-executing them,
so resuming after a crash, an approval or a piece of feedback only pays for new work.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field, fields

from .evals import VisionJudge
from .planner import default_planner
from .providers import default_image_models, default_providers
from .registry import PermissionRequired, Registry, ToolFailed
from .store import Store
from .tools import build_registry

DEFAULT_SCOPES = ["plan", "assets:read", "generate", "evaluate", "media:process"]
AUTO = object()


@dataclass
class Config:
    name: str = "v3-failover"
    providers: list[str] = field(default_factory=lambda: ["fastgen", "proline"])
    tool_retries: int = 2            # retries on transient provider errors
    evals: bool = True               # gate each clip on automated checks
    judge: bool = False              # also ask the vision judge (needs real footage)
    palette_check: bool = True       # pixel-level palette check (suits flat graphics only)
    attempts_per_provider: int = 2   # generations per shot before failing over
    budget_usd: float = 5.0
    width: int = 320
    height: int = 180
    palette_tolerance: float = 40.0
    max_shot_seconds: float = 6.0
    max_total_seconds: float = 30.0
    target_seconds: float = 0        # 0 = let the planner decide
    best_effort: bool = False        # if no take passes review, keep the closest one and flag it
    reference_mode: bool = False     # draw a subject reference and get it approved; shots are judged against it
    keyframes: bool = False          # give each shot its own opening frame, made from the reference
    keyframe_attempts: int = 2
    project: str = "default"


PRESETS = {
    "v1-naive": Config(name="v1-naive", providers=["fastgen"], tool_retries=0, evals=False,
                       attempts_per_provider=1),
    "v2-evals": Config(name="v2-evals", providers=["fastgen"], tool_retries=2, evals=True,
                       attempts_per_provider=3),
    "v3-failover": Config(),
}


class SimulatedCrash(Exception):
    """Raised by the --crash-after test hook to stand in for the process dying."""


class BudgetExceeded(Exception):
    pass


class ShotFailed(Exception):
    def __init__(self, message: str, shot_id: str | None = None):
        super().__init__(message)
        self.shot_id = shot_id


class RunContext:
    def __init__(self, store: Store, registry: Registry, run_id: str, cfg: Config, state: dict,
                 crash_after: int | None = None):
        self.store, self.registry, self.run_id, self.cfg, self.state = store, registry, run_id, cfg, state
        self.crash_after, self.executed, self.replaying = crash_after, 0, False

    def emit(self, kind: str, key: str = "", **data) -> None:
        self.store.emit(self.run_id, kind, key, data)

    def decide(self, key: str, **data) -> None:
        """Record why the agent chose its next move (once, not again on replay)."""
        if not self.replaying:
            self.emit("decision", key, **data)

    def step(self, key: str, tool: str, soft: bool = False, **args) -> dict:
        """Run a tool once per (run, key). `soft` records a tool failure as an outcome
        the agent can react to instead of aborting the run."""
        cached = self.store.get_step(self.run_id, key)
        self.replaying = cached is not None
        if cached is not None:
            self.emit("step_replayed", key, tool=tool)
            return cached
        if self.crash_after is not None and self.executed >= self.crash_after:
            raise SimulatedCrash(key)
        self.executed += 1
        self.emit("step_start", key, tool=tool)
        t0 = time.time()
        try:
            out = self.registry.call(
                tool, args, set(self.state["granted"]), retries=self.cfg.tool_retries,
                on_retry=lambda n, err: self.emit("tool_retry", key, tool=tool, attempt=n + 1, error=err))
            out = {"ok": True, **out}
        except ToolFailed as e:
            if not soft:
                self.emit("step_failed", key, tool=tool, error=str(e))
                raise
            out = {"ok": False, "error": str(e)}
        ms = round((time.time() - t0) * 1000, 1)
        cost = float(out.get("cost_usd", 0.0))
        self.store.save_step(self.run_id, key, tool, out, cost, ms)
        self.emit("step_ok" if out["ok"] else "step_failed", key, tool=tool, ms=ms, cost_usd=cost,
                  **({} if out["ok"] else {"error": out["error"]}))
        return out


class Harness:
    def __init__(self, home: str = ".cutroom", store: Store | None = None, providers=None, planner=None,
                 judge=AUTO, backoff_s: float = 0.0, images: dict | None = None):
        self.home = os.path.abspath(home)
        os.makedirs(self.home, exist_ok=True)
        self.store = store or Store(os.path.join(self.home, "cutroom.db"))
        self.providers = providers or default_providers()
        self.live = not all(p.simulated for p in self.providers.values())
        if judge is AUTO:  # the judge needs real footage to look at, and a key to look with
            judge = VisionJudge() if self.live and os.environ.get("ANTHROPIC_API_KEY") else None
        self.judge = judge
        self.planner = planner or default_planner()
        if self.live and not backoff_s:
            backoff_s = float(os.environ.get("CUTROOM_BACKOFF_S", "3"))  # real APIs need real pauses
        self.images = images if images is not None else default_image_models()
        self.registry = build_registry(self.providers, self.planner, judge, self.store, self.home, backoff_s,
                                       self.images)

    def default_config(self) -> Config:
        """Settings for whatever this harness is connected to."""
        reference = {"reference_mode": bool(self.images.get("reference")),
                     "keyframes": bool(self.images.get("reference") and self.images.get("keyframe"))}
        if not self.live:
            return Config(providers=list(self.providers), **reference)
        return Config(
            name="live", providers=list(self.providers), judge=self.judge is not None, palette_check=False,
            tool_retries=3, best_effort=True, **reference,
            width=848, height=480,
            budget_usd=float(os.environ.get("CUTROOM_BUDGET_USD", "3")),
            max_shot_seconds=float(os.environ.get("CUTROOM_MAX_SHOT_SECONDS", "5")),
            max_total_seconds=float(os.environ.get("CUTROOM_MAX_SECONDS", "15")),
        )

    # lifecycle --------------------------------------------------------------
    def create(self, brief: str, config: Config | None = None, auto_approve: bool = False) -> str:
        cfg = config or self.default_config()
        granted = DEFAULT_SCOPES + (["reference:approve", "deliver:publish"] if auto_approve else [])
        return self.store.create_run(brief, asdict(cfg), {"granted": granted, "revisions": {}})

    def run(self, run_id: str, crash_after: int | None = None) -> dict:
        """Start or resume a run. Safe to call repeatedly."""
        run = self.store.get_run(run_id)
        known = {f.name for f in fields(Config)}
        cfg, state = Config(**{k: v for k, v in run["config"].items() if k in known}), run["state"]
        ctx = RunContext(self.store, self.registry, run_id, cfg, state, crash_after)
        self.store.set_status(run_id, "running")
        ctx.emit("run_start", config=cfg.name, resumed=bool(self.store.steps(run_id)))
        status, detail = "completed", ""
        state.pop("failed_shot", None)
        try:
            self._execute(ctx, run["brief"])
        except PermissionRequired as e:
            status, detail = "awaiting_approval", e.scope
            state["pending_scope"] = e.scope
            ctx.emit("permission_required", tool=e.tool, scope=e.scope)
        except BudgetExceeded as e:
            status, detail = "budget_exceeded", str(e)
        except ShotFailed as e:
            status, detail = "failed", str(e)
            state["failed_shot"] = e.shot_id
        except ToolFailed as e:
            status, detail = "failed", str(e)
        except SimulatedCrash as e:
            status, detail = "interrupted", f"process died before step '{e}'"
        except Exception as e:  # a bug must not leave the run looking alive
            status, detail = "failed", f"internal error: {type(e).__name__}: {e}"
        self.store.save_state(run_id, state)
        self.store.set_status(run_id, status)
        ctx.emit("run_end", status=status, detail=detail, cost_usd=self.store.total_cost(run_id))
        return self.store.get_run(run_id)

    def approve(self, run_id: str, scope: str | None = None) -> dict:
        state = self.store.get_run(run_id)["state"]
        scope = scope or state.pop("pending_scope", None)
        if scope and scope not in state["granted"]:
            state["granted"].append(scope)
            self.store.emit(run_id, "permission_granted", "", {"scope": scope})
        state.pop("pending_scope", None)
        self.store.save_state(run_id, state)
        return self.run(run_id)

    def request_change(self, run_id: str, shot_id: str, note: str = "", remember: bool = False,
                       revoke_publish: bool = True) -> None:
        """Record a creator's change to one shot without running. Bumps that shot's
        revision so it gets new step keys, and withdraws publish approval: a changed
        cut has to be approved again."""
        run = self.store.get_run(run_id)
        state = run["state"]
        rev = state["revisions"].setdefault(shot_id, {"rev": 0, "notes": []})
        rev["rev"] += 1
        if note:
            rev["notes"].append(note)
            self.store.add_feedback(run_id, shot_id, note)
            if remember:
                self.store.add_preference(run["config"]["project"], note)
        if revoke_publish:
            state["granted"] = [s for s in state["granted"] if s != "deliver:publish"]
        self.store.save_state(run_id, state)
        self.store.emit(run_id, "feedback", shot_id, {"note": note, "remember": remember})

    def redo_reference(self, run_id: str, note: str = "") -> None:
        """Creator rejects the subject reference. A new one is drawn, approval is needed
        again, and every shot is refilmed, since each was built on the old reference."""
        run = self.store.get_run(run_id)
        state = run["state"]
        state["reference_rev"] = state.get("reference_rev", 0) + 1
        if note:
            state.setdefault("reference_notes", []).append(note)
        plan = self.store.get_step(run_id, "plan")
        keys = [s["key"] for s in self.store.steps(run_id)]
        for shot in (plan or {}).get("shots", []):
            if any(k.startswith((f"gen:{shot['id']}:", f"key:{shot['id']}:")) for k in keys):
                state["revisions"].setdefault(shot["id"], {"rev": 0, "notes": []})["rev"] += 1
        state["granted"] = [s for s in state["granted"] if s not in ("reference:approve", "deliver:publish")]
        self.store.save_state(run_id, state)
        self.store.emit(run_id, "feedback", "reference", {"note": note})

    def feedback(self, run_id: str, shot_id: str, note: str, remember: bool = False) -> dict:
        """Request a change and run it straight away, keeping any approval already given."""
        self.request_change(run_id, shot_id, note, remember, revoke_publish=False)
        return self.run(run_id)

    def raise_budget(self, run_id: str, budget_usd: float) -> dict:
        cfg = self.store.get_run(run_id)["config"]
        cfg["budget_usd"] = budget_usd
        self.store.save_config(run_id, cfg)
        return self.run(run_id)

    # the workflow -------------------------------------------------------------
    def _execute(self, ctx: RunContext, brief: str) -> None:
        cfg = ctx.cfg
        run_dir = os.path.join(self.home, "runs", ctx.run_id)
        plan = ctx.step("plan", "plan_video", brief=brief, preferences=self.store.preferences(cfg.project),
                        max_shot_seconds=cfg.max_shot_seconds, max_total_seconds=cfg.max_total_seconds,
                        target_seconds=cfg.target_seconds)
        if not plan["shots"]:
            raise ShotFailed("the plan has no shots")
        route = ctx.step("route", "choose_models", providers=cfg.providers)
        assets = ctx.step("assets", "select_assets", mood=plan["style"]["mood"])
        noise_key = hashlib.sha256(brief.encode()).hexdigest()[:12]
        subject_ref = self._reference(ctx, plan, run_dir) if cfg.reference_mode else None
        clips: list[str] = []
        for shot in plan["shots"]:
            # With a subject reference every shot is judged against it, and each shot gets its own
            # opening frame derived from it. No shot is ever filmed from the reference itself: that
            # would make every shot open on the same picture. Without a reference, later shots are
            # judged against the first accepted clip.
            reference = subject_ref or (clips[0] if clips else None)
            start = self._keyframe(ctx, plan, shot, subject_ref, run_dir) if subject_ref and cfg.keyframes else None
            clips.append(self._produce_shot(ctx, plan, route, shot, run_dir, noise_key, reference, start))
        # the cut is identified by which clips are in it, so a changed shot means a new assemble step
        cut = hashlib.sha256("|".join(os.path.basename(c) for c in clips).encode()).hexdigest()[:8]
        final = ctx.step(f"assemble:{cut}", "assemble", clips=clips, audio=assets["audio"],
                         out_path=os.path.join(run_dir, f"final_{cut}.mp4"))
        if cfg.evals:
            check = ctx.step(f"final_eval:{cut}", "evaluate_final", path=final["path"],
                             expected_duration=sum(s["duration"] for s in plan["shots"]),
                             width=cfg.width, height=cfg.height, expect_audio=bool(assets["audio"]))
            if not check["passed"]:
                raise ShotFailed(f"final cut failed checks: {check['failures']}")
        ctx.step(f"deliver:{cut}", "deliver", path=final["path"], run_id=ctx.run_id, title=plan["title"])

    def _reference(self, ctx: RunContext, plan: dict, run_dir: str) -> str:
        """Draw the subject once, then stop until the creator approves it."""
        rr, notes = ctx.state.get("reference_rev", 0), ctx.state.get("reference_notes", [])
        style = plan["style"]
        prompt = " ".join([f"{plan.get('subject', plan['title'])}.", f"Look: {style.get('look', style['mood'])}.",
                           "The whole subject clearly visible, centred, simple uncluttered background, "
                           "no text.", *plan.get("preferences", []), *notes])
        ref = ctx.step(f"ref:r{rr}", "create_reference", prompt=prompt, seed=rr,
                       out_path=os.path.join(run_dir, "stills", f"ref_r{rr}.jpg"))
        ctx.step(f"ref_ok:r{rr}", "approve_reference", path=ref["path"])  # pauses here without the scope
        return ref["path"]

    def _keyframe(self, ctx: RunContext, plan: dict, shot: dict, subject_ref: str, run_dir: str) -> str | None:
        """The shot's opening frame: the referenced subject placed in this shot's scene.
        It is reviewed before any video is made from it, because a still costs a fraction
        of a clip. If none passes, the closest is used; if none can be made, the shot is
        filmed from text alone."""
        cfg = ctx.cfg
        revision = ctx.state["revisions"].get(shot["id"], {"rev": 0, "notes": []})
        rev, style = revision["rev"], plan["style"]
        best: list[tuple[int, str]] = []
        hints: list[str] = []
        for k in range(cfg.keyframe_attempts):
            tag = f"{shot['id']}:r{rev}:k{k}"
            prompt = " ".join([f"The same subject as in the reference image, unchanged: {plan.get('subject', '')}.",
                               f"Scene: {shot['description'].rstrip('.')}.",
                               f"Look: {style.get('look', style['mood'])}.", "No text.",
                               *revision["notes"], *hints])
            key = ctx.step(f"key:{tag}", "create_keyframe", soft=True, prompt=prompt, reference_path=subject_ref,
                           seed=rev * 100 + k, width=cfg.width, height=cfg.height,
                           out_path=os.path.join(run_dir, "stills", f"key_{shot['id']}_r{rev}_k{k}.jpg"))
            if not key["ok"]:
                continue
            if not (cfg.evals and cfg.judge):
                return key["path"]
            ev = ctx.step(f"keyeval:{tag}", "evaluate_keyframe", path=key["path"], shot=shot["description"],
                          subject=plan.get("subject", ""), look=style.get("look", ""), reference_path=subject_ref)
            if ev["passed"]:
                return key["path"]
            best.append((sum(ev["judge"]["scores"].values()), key["path"]))
            hints = ev.get("hints", [])
            ctx.decide(f"keyeval:{tag}", action="redraw_keyframe", reason=f"failed {ev['failures']}", hints=hints)
        if best:
            ctx.decide(f"key:{shot['id']}:r{rev}", action="use_closest_keyframe", reason="no keyframe passed review")
            return max(best)[1]
        ctx.decide(f"key:{shot['id']}:r{rev}", action="film_from_text", reason="no keyframe could be made")
        return None

    def _produce_shot(self, ctx: RunContext, plan: dict, route: dict, shot: dict, run_dir: str,
                      noise_key: str, reference: str | None, start_image: str | None = None) -> str:
        cfg = ctx.cfg
        revision = ctx.state["revisions"].get(shot["id"], {"rev": 0, "notes": []})
        rev, attempt, hints = revision["rev"], 0, []
        candidates: list[tuple[int, str]] = []  # (judge score, path) of takes that play but were rejected
        style = plan["style"]
        context = [shot["description"].rstrip(".") + ".", f"Subject: {plan.get('subject', '')}.",
                   f"Look: {style.get('look', style['mood'])}."]
        for provider in route["order"]:
            guards = route["guards"].get(provider, [])  # what this model keeps getting wrong
            for _ in range(cfg.attempts_per_provider):
                tag = f"{shot['id']}:r{rev}:a{attempt}"
                if self.store.get_step(ctx.run_id, f"gen:{tag}") is None:  # only new work counts against budget
                    estimate = self.providers[provider].price_per_s * shot["duration"]
                    spent = self.store.total_cost(ctx.run_id)
                    if spent + estimate > cfg.budget_usd + 1e-9:
                        raise BudgetExceeded(f"${spent:.2f} spent, next clip ~${estimate:.2f}, "
                                             f"budget ${cfg.budget_usd:.2f}")
                extras = list(dict.fromkeys([*plan.get("preferences", []), *revision["notes"], *guards, *hints]))
                gen = ctx.step(f"gen:{tag}", "generate_clip", soft=True, provider=provider,
                               prompt=" ".join(context + extras), duration=shot["duration"],
                               width=cfg.width, height=cfg.height, palette=style["palette"],
                               seed=rev * 100 + attempt, shot_id=shot["id"], noise_key=noise_key,
                               image_path=start_image,
                               out_path=os.path.join(run_dir, "clips", f"{shot['id']}_r{rev}_a{attempt}.mp4"))
                attempt += 1
                if not gen["ok"]:
                    ctx.decide(f"gen:{tag}", action="leave_provider", provider=provider,
                               reason="provider kept erroring after retries")
                    break
                if not cfg.evals:
                    return gen["path"]
                ev = ctx.step(f"eval:{tag}", "evaluate_clip", path=gen["path"], provider=provider,
                              duration=shot["duration"], width=cfg.width, height=cfg.height,
                              palette=style["palette"] if cfg.palette_check else None,
                              palette_tolerance=cfg.palette_tolerance, judge=cfg.judge,
                              shot=shot["description"], subject=plan.get("subject", ""),
                              look=style.get("look", ""), reference_path=reference)
                if ev["passed"]:
                    return gen["path"]
                if ev.get("judge"):
                    candidates.append((sum(ev["judge"]["scores"].values()), gen["path"]))
                hints = ev.get("hints", [])
                ctx.decide(f"eval:{tag}", action="regenerate", provider=provider,
                           reason=f"failed {ev['failures']}", hints=hints)
        # Nothing passed. Degrade instead of throwing the creator's video away:
        # a failed change keeps the take it was meant to replace; otherwise the closest take is
        # kept and flagged. Either way the creator sees it and decides.
        previous = self._previous_clip(ctx, shot["id"], rev)
        if previous:
            return self._fall_back(ctx, shot["id"], rev, "kept_previous", previous)
        if cfg.best_effort and candidates:
            return self._fall_back(ctx, shot["id"], rev, "best_effort", max(candidates)[1])
        raise ShotFailed(f"shot {shot['id']}: no acceptable clip after {attempt} attempts "
                         f"across {route['order']}", shot["id"])

    def _previous_clip(self, ctx: RunContext, shot_id: str, rev: int) -> str | None:
        """The clip this shot used before the current revision, if it had one."""
        steps = {s["key"]: s["output"] for s in self.store.steps(ctx.run_id)}
        for r in range(rev - 1, -1, -1):
            kept = ctx.state.get("fallbacks", {}).get(shot_id)
            if kept and kept["rev"] == r:
                return kept["path"]
            best = None
            for key, out in steps.items():
                if key.startswith(f"gen:{shot_id}:r{r}:a") and out["ok"]:
                    ev = steps.get("eval:" + key[4:])
                    if not ctx.cfg.evals or (ev and ev.get("passed")):
                        best = out["path"]
            if best:
                return best
        return None

    def _fall_back(self, ctx: RunContext, shot_id: str, rev: int, kind: str, path: str) -> str:
        ctx.state.setdefault("fallbacks", {})[shot_id] = {"rev": rev, "kind": kind, "path": path}
        ctx.decide(f"shot:{shot_id}:r{rev}", action=kind, clip=os.path.basename(path),
                   reason="no take passed review")
        return path

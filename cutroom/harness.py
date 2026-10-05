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
from .providers import default_providers
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
                 judge=AUTO, backoff_s: float = 0.0):
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
        self.registry = build_registry(self.providers, self.planner, judge, self.store, self.home, backoff_s)

    def default_config(self) -> Config:
        """Settings for whatever this harness is connected to."""
        if not self.live:
            return Config(providers=list(self.providers))
        return Config(
            name="live", providers=list(self.providers), judge=self.judge is not None, palette_check=False,
            tool_retries=3,
            width=848, height=480,
            budget_usd=float(os.environ.get("CUTROOM_BUDGET_USD", "3")),
            max_shot_seconds=float(os.environ.get("CUTROOM_MAX_SHOT_SECONDS", "5")),
            max_total_seconds=float(os.environ.get("CUTROOM_MAX_SECONDS", "15")),
        )

    # lifecycle --------------------------------------------------------------
    def create(self, brief: str, config: Config | None = None, auto_approve: bool = False) -> str:
        cfg = config or self.default_config()
        granted = DEFAULT_SCOPES + (["deliver:publish"] if auto_approve else [])
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
        clips: list[str] = []
        for shot in plan["shots"]:
            reference = clips[0] if clips else None  # later shots are judged against the first accepted one
            clips.append(self._produce_shot(ctx, plan, route, shot, run_dir, noise_key, reference))
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

    def _produce_shot(self, ctx: RunContext, plan: dict, route: dict, shot: dict, run_dir: str,
                      noise_key: str, reference: str | None) -> str:
        cfg = ctx.cfg
        revision = ctx.state["revisions"].get(shot["id"], {"rev": 0, "notes": []})
        rev, attempt, hints = revision["rev"], 0, []
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
                hints = ev.get("hints", [])
                ctx.decide(f"eval:{tag}", action="regenerate", provider=provider,
                           reason=f"failed {ev['failures']}", hints=hints)
        raise ShotFailed(f"shot {shot['id']}: no acceptable clip after {attempt} attempts "
                         f"across {route['order']}", shot["id"])

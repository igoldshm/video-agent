import os

import pytest
from pydantic import BaseModel

from directorcut import evals
from directorcut.harness import PRESETS, Config, Harness
from directorcut.providers import GenRequest, MockProvider
from directorcut.registry import PermissionRequired, Registry, Tool, ToolFailed, TransientError

BRIEF = "A 12s calm launch video for a sleep tea brand"
PALETTE = ["#1f5f8b", "#9fd3c7"]


def clean_providers():
    return {"fastgen": MockProvider("fastgen", 0.04), "proline": MockProvider("proline", 0.12)}


@pytest.fixture
def h(tmp_path):
    return Harness(str(tmp_path / "home"))


def test_keys_in_the_environment_switch_nothing_on_by_accident(tmp_path):
    h = Harness(str(tmp_path / "home"))
    assert not h.live and h.judge is None and h.default_config().palette_check


def executed(h, rid):
    return [e["key"] for e in h.store.events(rid) if e["kind"] in ("step_ok", "step_failed")]


def test_run_delivers_a_playable_video(h):
    run = h.run(h.create(BRIEF, auto_approve=True))
    assert run["status"] == "completed"
    out = os.path.join(h.home, "deliveries", f"{run['id']}.mp4")
    info = evals.media.probe(out)
    assert info["has_video"] and info["has_audio"] and abs(info["duration"] - 12) < 1.5


def test_resume_after_crash_never_repeats_or_repays_a_step(tmp_path):
    whole = Harness(str(tmp_path / "a"))
    baseline = whole.run(whole.create(BRIEF, auto_approve=True))

    h = Harness(str(tmp_path / "b"))
    rid = h.create(BRIEF, auto_approve=True)
    assert h.run(rid, crash_after=4)["status"] == "interrupted"
    assert h.run(rid, crash_after=3)["status"] == "interrupted"  # dies a second time
    run = h.run(rid)
    assert run["status"] == "completed"
    keys = executed(h, rid)
    assert len(keys) == len(set(keys)), "a step ran twice"
    assert keys == executed(whole, baseline["id"])
    assert run["cost_usd"] == baseline["cost_usd"]


def test_delivery_waits_for_creator_approval(h):
    rid = h.create(BRIEF)
    run = h.run(rid)
    assert run["status"] == "awaiting_approval" and run["state"]["pending_scope"] == "deliver:publish"
    assert not os.path.exists(os.path.join(h.home, "deliveries", f"{rid}.mp4"))
    before = len(executed(h, rid))
    assert h.approve(rid)["status"] == "completed"
    assert executed(h, rid)[before:] == [k for k in executed(h, rid) if k.startswith("deliver:")]


def test_feedback_regenerates_only_the_named_shot(tmp_path):
    h = Harness(str(tmp_path / "home"), providers=clean_providers())
    rid = h.create(BRIEF, auto_approve=True)
    h.run(rid)
    before = set(executed(h, rid))
    run = h.feedback(rid, "s2", "Make it warmer and slower.", remember=True)
    assert run["status"] == "completed"
    new = [k for k in executed(h, rid) if k not in before]
    assert [k.split(":")[0] for k in new] == ["gen", "eval", "assemble", "final_eval", "deliver"]
    assert all(":s2:r1:" in k for k in new[:2])
    # the note is in the regenerated prompt, and remembered for the project's next video
    rid2 = h.create("A 9s teaser for the same brand", auto_approve=True)
    h.run(rid2)
    assert h.store.get_step(rid2, "plan")["preferences"] == ["Make it warmer and slower."]


@pytest.mark.parametrize("defect,check", [("black", "no_black_frames"), ("frozen", "has_motion"),
                                          ("off_palette", "style_continuity"), ("short", "duration")])
def test_evals_catch_each_defect_from_the_pixels(tmp_path, defect, check):
    path = str(tmp_path / f"{defect}.mp4")
    MockProvider("t", 0.1, force_defect=defect).generate(
        GenRequest("prompt", 3, 320, 180, PALETTE, 0, "s1", "k"), path)
    assert check in evals.evaluate_clip(path, 3, 320, 180, PALETTE)["failures"]


def test_good_clip_passes_all_checks(tmp_path):
    path = str(tmp_path / "ok.mp4")
    MockProvider("t", 0.1).generate(GenRequest("prompt", 3, 320, 180, PALETTE, 0, "s1", "k"), path)
    assert evals.evaluate_clip(path, 3, 320, 180, PALETTE)["passed"]


def test_bad_clips_are_rejected_then_the_agent_fails_over(tmp_path):
    providers = {"fastgen": MockProvider("fastgen", 0.04, force_defect="black"),
                 "proline": MockProvider("proline", 0.12)}
    h = Harness(str(tmp_path / "home"), providers=providers)
    rid = h.create(BRIEF, auto_approve=True)
    assert h.run(rid)["status"] == "completed"
    accepted = [s["output"]["provider"] for s in h.store.steps(rid)
                if s["key"].startswith("gen:") and s["key"].endswith(":a2")]
    assert accepted and set(accepted) == {"proline"}


def test_dead_provider_is_left_after_retries(tmp_path):
    providers = {"fastgen": MockProvider("fastgen", 0.04, p_error=1.0), "proline": MockProvider("proline", 0.12)}
    h = Harness(str(tmp_path / "home"), providers=providers)
    rid = h.create(BRIEF, auto_approve=True)
    run = h.run(rid)
    assert run["status"] == "completed"
    assert sum(e["kind"] == "tool_retry" for e in h.store.events(rid)) == 2 * 4  # 2 retries x 4 shots
    assert run["cost_usd"] == pytest.approx(0.12 * 12)  # failed calls are not billed


def test_budget_stops_the_run_and_raising_it_resumes(h):
    rid = h.create(BRIEF, Config(budget_usd=0.2), auto_approve=True)
    run = h.run(rid)
    assert run["status"] == "budget_exceeded" and run["cost_usd"] <= 0.2
    assert h.raise_budget(rid, 5.0)["status"] == "completed"


def test_naive_preset_ships_whatever_it_gets(tmp_path):
    providers = {"fastgen": MockProvider("fastgen", 0.04, force_defect="black")}
    h = Harness(str(tmp_path / "home"), providers=providers)
    assert h.run(h.create(BRIEF, PRESETS["v1-naive"], auto_approve=True))["status"] == "completed"


def test_registry_validates_scopes_arguments_and_retries():
    class In(BaseModel):
        n: int

    calls = []

    def flaky(inp, call):
        calls.append(call.attempt)
        if call.attempt < 2:
            raise TransientError("503")
        return {"n": inp.n}

    reg = Registry()
    reg.register(Tool("flaky", In, flaky, scopes=("generate",)))
    with pytest.raises(PermissionRequired):
        reg.call("flaky", {"n": 1}, granted=set())
    with pytest.raises(ToolFailed, match="invalid arguments"):
        reg.call("flaky", {"n": "x"}, granted={"generate"})
    with pytest.raises(ToolFailed, match="gave up"):
        reg.call("flaky", {"n": 1}, granted={"generate"}, retries=1)
    calls.clear()
    assert reg.call("flaky", {"n": 1}, granted={"generate"}, retries=2) == {"n": 1}
    assert calls == [0, 1, 2]

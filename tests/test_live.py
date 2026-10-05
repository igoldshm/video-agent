"""The real-model path, exercised end to end against stubbed HTTP responses."""
import json
import os

import httpx
import pytest

from cutroom import learning, media
from cutroom.evals import VisionJudge
from cutroom.harness import Harness
from cutroom.llm import Claude
from cutroom.planner import ClaudePlanner
from cutroom.providers import GenRequest, MockProvider, ReplicateProvider
from cutroom.registry import ToolFailed, TransientError
from cutroom.store import Store

SCHEMA = {"latest_version": {"openapi_schema": {"components": {"schemas": {"Input": {"properties": {
    "prompt": {"type": "string"}, "seed": {"type": "integer"},
    "aspect_ratio": {"enum": ["16:9", "9:16"]}, "resolution": {"enum": ["480p", "720p"]},
    "duration": {"enum": [5, 10]}}}}}}}}
PLAN = {"title": "Snow day", "subject": "a golden retriever puppy with a red collar",
        "style": {"mood": "warm", "palette": ["#f2c078", "#5b8fb9"], "look": "soft morning light, 35mm"},
        "shots": [{"beat": "Opening", "description": "The puppy steps onto fresh snow", "duration": 4},
                  {"beat": "Play", "description": "The puppy leaps through a snowdrift", "duration": 9}]}


@pytest.fixture(scope="module")
def clip_bytes(tmp_path_factory):
    """A real, decodable mp4 at an odd size and length, like a model would return."""
    path = str(tmp_path_factory.mktemp("src") / "model_output.mp4")
    MockProvider("src", 0).generate(GenRequest("x", 6, 640, 384, ["#f2c078", "#5b8fb9"], 1, "s1"), path)
    with open(path, "rb") as f:
        return f.read()


class FakeWeb:
    """Stands in for Replicate, its file host and the Anthropic API."""

    def __init__(self, clip: bytes, reviews=None, statuses=None):
        self.clip, self.reviews, self.statuses = clip, list(reviews or []), list(statuses or [])
        self.predictions, self.llm_calls, self.file_auth = [], [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "api.anthropic.com":
            body = json.loads(request.content)
            self.llm_calls.append(body)
            name = body["tool_choice"]["name"]
            out = PLAN if name == "submit_plan" else (self.reviews.pop(0) if self.reviews else
                  {"matches_shot": 5, "visual_quality": 4, "consistent_with_reference": 5, "issues": [], "note": "ok"})
            return httpx.Response(200, json={"content": [{"type": "tool_use", "name": name, "input": out}]})
        if host == "files.example":
            self.file_auth.append(request.headers.get("authorization"))
            return httpx.Response(200, content=self.clip)
        if request.method == "GET" and path.startswith("/v1/models/"):
            return httpx.Response(200, json=SCHEMA)
        if request.method == "POST":
            if self.statuses:
                return httpx.Response(self.statuses.pop(0), json={"detail": "nope"})
            self.predictions.append(json.loads(request.content)["input"])
            return httpx.Response(201, json={"status": "starting", "urls": {"get": "https://api.replicate.com/v1/predictions/p1"}})
        return httpx.Response(200, json={"status": "succeeded", "output": ["https://files.example/out.mp4"]})


def replicate(web, model="acme/fast-video", price=0.02):
    return ReplicateProvider(model, price, token="r8_test", transport=httpx.MockTransport(web), poll_s=0)


def test_replicate_adapter_sends_only_what_the_model_accepts(tmp_path, clip_bytes):
    web = FakeWeb(clip_bytes)
    out = str(tmp_path / "clip.mp4")
    replicate(web).generate(GenRequest("a puppy in snow", 4, 848, 480, ["#000000", "#ffffff"], 7, "s1"), out)
    assert web.predictions == [{"prompt": "a puppy in snow", "seed": 7, "aspect_ratio": "16:9",
                                "resolution": "480p", "duration": 5}]  # 4s shot -> shortest allowed length >= 4
    assert web.file_auth == [None], "the API token must not be sent to the file host"
    assert media.probe(out)["has_video"]


def test_replicate_adapter_separates_retryable_from_fatal(tmp_path, clip_bytes):
    req = GenRequest("x", 4, 848, 480, ["#000000", "#ffffff"], 0, "s1")
    with pytest.raises(TransientError):
        replicate(FakeWeb(clip_bytes, statuses=[429])).generate(req, str(tmp_path / "a.mp4"))
    with pytest.raises(ToolFailed):
        replicate(FakeWeb(clip_bytes, statuses=[422])).generate(req, str(tmp_path / "b.mp4"))
    with pytest.raises(ToolFailed, match="REPLICATE_API_TOKEN"):
        ReplicateProvider("acme/fast-video", 0.02, token="").generate(req, str(tmp_path / "c.mp4"))


def test_planner_output_is_fitted_to_the_limits(clip_bytes):
    web = FakeWeb(clip_bytes)
    planner = ClaudePlanner(Claude(api_key="k", transport=httpx.MockTransport(web)))
    plan = planner.plan("A puppy discovers snow", ["No people"], max_shot_seconds=5, max_total_seconds=8)
    assert [(s["id"], s["duration"]) for s in plan["shots"]] == [("s1", 4.0), ("s2", 4.0)]  # 9s clamped to what is left
    assert plan["preferences"] == ["No people"] and "No people" in web.llm_calls[0]["messages"][0]["content"]


def test_live_run_judges_real_frames_and_revises_the_prompt(tmp_path, clip_bytes):
    reject = {"matches_shot": 2, "visual_quality": 4, "consistent_with_reference": 2,
              "issues": ["style_drift"], "note": "different dog, no collar"}
    ok = {"matches_shot": 5, "visual_quality": 4, "consistent_with_reference": 5, "issues": [], "note": "ok"}
    web = FakeWeb(clip_bytes, reviews=[ok, reject])  # shot 1 passes, shot 2's first take is rejected
    llm = Claude(api_key="k", transport=httpx.MockTransport(web))
    h = Harness(str(tmp_path / "home"), providers={"fast-video": replicate(web)}, planner=ClaudePlanner(llm),
                judge=VisionJudge(llm))
    cfg = h.default_config()
    assert h.live and cfg.judge and not cfg.palette_check and (cfg.width, cfg.height) == (848, 480)
    run = h.run(h.create("A puppy discovers snow", cfg, auto_approve=True))
    assert run["status"] == "completed", h.store.events(run["id"])[-1]

    prompts = [p["prompt"] for p in web.predictions]
    assert len(prompts) == 3 and all("golden retriever puppy with a red collar" in p and "35mm" in p for p in prompts)
    assert "Match the established subject" in prompts[2] and "Match the established subject" not in prompts[1]

    reviews = [c for c in web.llm_calls if c["tool_choice"]["name"] == "submit_review"]
    images = [sum(b["type"] == "image" for b in c["messages"][0]["content"]) for c in reviews]
    assert images == [3, 4, 4], "three frames of the clip, plus a reference frame once shot 1 is accepted"

    final = media.probe(os.path.join(h.home, "deliveries", f"{run['id']}.mp4"))
    assert (final["width"], final["height"]) == (848, 480) and not final["has_audio"]
    assert abs(final["duration"] - 9) < 0.5, "a 4s and a 5s shot, each trimmed from the 6s the model returned"
    assert run["cost_usd"] == pytest.approx((4 + 5 + 5) * 0.02)


def test_history_reorders_models_and_adds_guards():
    store = Store()
    rid = store.create_run("b", {}, {})
    def record(i, provider, passed, failures=()):
        store.save_step(rid, f"eval:{provider}:{i}", "evaluate_clip",
                        {"ok": True, "provider": provider, "passed": passed, "failures": list(failures)}, 0, 0)
    prices = {"cheap": 0.02, "good": 0.05}
    assert learning.choose_models(store, ["cheap", "good"], prices)["order"] == ["cheap", "good"]  # no history yet
    for i in range(10):
        record(i, "cheap", passed=i < 2, failures=() if i < 2 else ("garbled_text",))
        record(i, "good", passed=True)
    route = learning.choose_models(store, ["cheap", "good"], prices)
    assert route["order"] == ["good", "cheap"]  # 0.05/1.0 beats 0.02/0.2 per accepted clip
    assert route["guards"] == {"cheap": ["No on-screen text, logos or lettering anywhere."]}
    assert route["basis"]["cheap"] == {"clips": 10, "pass_rate": 0.2}

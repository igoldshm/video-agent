"""The creator-facing service."""
import pytest
from fastapi.testclient import TestClient

from cutroom.api import create_app
from cutroom.harness import Harness
from cutroom.planner import HeuristicPlanner
from cutroom.providers import MockProvider

BRIEF = "A calm product film for a sleep tea brand"


def harness(tmp_path, **providers):
    providers = providers or {"fastgen": MockProvider("fastgen", 0.04), "proline": MockProvider("proline", 0.12)}
    return Harness(str(tmp_path / "home"), providers=providers, planner=HeuristicPlanner(), judge=None)


def test_creator_flow_from_brief_to_published_link(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTROOM_ACCESS_CODE", "letmein")
    client = TestClient(create_app(harness(tmp_path), sync=True))
    assert client.get("/").status_code == 200 and "Describe the video you want" in client.get("/").text
    assert client.get("/api/config").json() == {"needs_code": True, "live": False, "max_seconds": 30}
    assert client.post("/api/videos", json={"brief": BRIEF}).status_code == 401
    auth = {"X-Access-Code": "letmein"}

    r = client.post("/api/videos", json={"brief": BRIEF, "seconds": 9}, headers=auth)
    v = r.json()
    assert r.status_code == 201 and v["status"] == "ready" and v["final_url"] and v["watch_url"] is None
    assert [s["state"] for s in v["shots"]] == ["ready"] * 3 and sum(s["seconds"] for s in v["shots"]) == 9
    # nothing about how it was made reaches the creator
    assert not any(word in r.text for word in ("gen:", "eval:", "cost", "provider", "fastgen", "scope"))
    assert client.get(v["final_url"]).headers["content-type"] == "video/mp4"
    assert client.get(v["shots"][0]["clip_url"]).status_code == 200
    assert client.get(f"/v/{v['id']}").status_code == 404, "not public until published"

    before = v["shots"][1]["clip_url"]
    v = client.post(f"/api/videos/{v['id']}/shots/s2/change", json={"note": "Warmer light", "remember": True},
                    headers=auth).json()
    assert v["status"] == "ready" and v["shots"][1]["notes"] == ["Warmer light"]
    assert v["shots"][1]["clip_url"] != before and v["shots"][0]["clip_url"].endswith("s1_r0_a0.mp4")
    assert client.get("/api/preferences", headers=auth).json() == ["Warmer light"]

    v = client.post(f"/api/videos/{v['id']}/publish", json={}, headers=auth).json()
    assert v["status"] == "published" and v["watch_url"] == f"/v/{v['id']}"
    assert "<video" in client.get(v["watch_url"]).text
    assert client.get(f"/published/{v['id']}.mp4").status_code == 200
    assert client.post(f"/api/videos/{v['id']}/publish", json={}, headers=auth).status_code == 409

    # a change after publishing needs approval again
    v = client.post(f"/api/videos/{v['id']}/shots/s1/change", json={"note": "Closer"}, headers=auth).json()
    assert v["status"] == "ready"
    assert client.get(f"/api/videos?ids={v['id']},nope", headers=auth).json()[0]["id"] == v["id"]
    assert client.get(f"/ops/{v['id']}/trace").status_code == 401
    assert "Event timeline" in client.get(f"/ops/{v['id']}/trace?code=letmein").text
    assert client.get("/media/../../etc/passwd").status_code == 404


def test_daily_limit_stops_new_videos(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTROOM_DAILY_VIDEOS", "1")
    client = TestClient(create_app(harness(tmp_path), sync=True))
    assert client.post("/api/videos", json={"brief": BRIEF}).status_code == 201
    r = client.post("/api/videos", json={"brief": BRIEF})
    assert r.status_code == 429 and "limit" in r.json()["detail"]


def test_failed_shot_is_explained_and_can_be_retried(tmp_path):
    bad = MockProvider("fastgen", 0.04, force_defect="black")
    h = harness(tmp_path, fastgen=bad)
    client = TestClient(create_app(h, sync=True))
    v = client.post("/api/videos", json={"brief": BRIEF}).json()
    assert v["status"] == "failed" and v["message"] == "We couldn't get shot 1 right after several takes."
    bad.force_defect = None  # the model recovers
    v = client.post(f"/api/videos/{v['id']}/retry", json={}).json()
    assert v["status"] == "ready" and v["shots"][0]["clip_url"].endswith("s1_r1_a0.mp4")


def test_unfinished_videos_continue_when_the_service_restarts(tmp_path):
    h = harness(tmp_path)
    rid = h.create(BRIEF)
    assert h.run(rid, crash_after=5)["status"] == "interrupted"
    h.store.set_status(rid, "running")  # what a killed process leaves behind
    paid = h.store.total_cost(rid)
    client = TestClient(create_app(h, sync=True))  # "restart"
    v = client.get(f"/api/videos/{rid}").json()
    assert v["status"] == "ready"
    keys = [e["key"] for e in h.store.events(rid) if e["kind"] == "step_ok"]
    assert len(keys) == len(set(keys)) and h.store.total_cost(rid) > paid


def test_failed_change_keeps_the_earlier_take_instead_of_losing_the_video(tmp_path):
    """Seen live: a change request hit provider errors and the finished video became 'failed'."""
    fast = MockProvider("fastgen", 0.04)
    client = TestClient(create_app(harness(tmp_path, fastgen=fast), sync=True))
    v = client.post("/api/videos", json={"brief": BRIEF, "seconds": 9}).json()
    original = v["shots"][1]["clip_url"]
    fast.force_defect = "black"  # every new take is now unusable
    v = client.post(f"/api/videos/{v['id']}/shots/s2/change", json={"note": "Two arms, not three"}).json()
    assert v["status"] == "ready" and v["final_url"]
    assert v["shots"][1]["clip_url"] == original and v["shots"][1]["flag"] == "kept_previous"
    assert v["shots"][0]["flag"] is None
    fast.force_defect = None  # a later change that works replaces it and clears the flag
    v = client.post(f"/api/videos/{v['id']}/shots/s2/change", json={"note": "Try once more"}).json()
    assert v["shots"][1]["flag"] is None and v["shots"][1]["clip_url"].endswith("s2_r2_a0.mp4")


def test_look_is_approved_before_filming_and_each_shot_gets_its_own_opening_frame(tmp_path):
    from cutroom.providers import MockImageModel
    h = Harness(str(tmp_path / "home"), providers={"fastgen": MockProvider("fastgen", 0.04)},
                planner=HeuristicPlanner(), judge=None,
                images={"reference": MockImageModel("ref"), "keyframe": MockImageModel("key")})
    client = TestClient(create_app(h, sync=True))
    v = client.post("/api/videos", json={"brief": BRIEF, "seconds": 9}).json()
    rid = v["id"]
    assert v["status"] == "review_reference" and v["reference"]["url"].endswith("ref_r0.jpg")
    assert client.get(v["reference"]["url"]).headers["content-type"] == "image/jpeg"
    keys = lambda: [s["key"] for s in h.store.steps(rid)]  # noqa: E731
    assert not any(k.startswith(("gen:", "key:")) for k in keys()), "nothing is filmed before the look is approved"
    assert client.post(f"/api/videos/{rid}/publish", json={}).status_code == 409

    v = client.post(f"/api/videos/{rid}/reference/redo", json={"note": "Make it taller"}).json()
    assert v["status"] == "review_reference" and v["reference"]["url"].endswith("ref_r1.jpg")
    assert v["reference"]["notes"] == ["Make it taller"]

    v = client.post(f"/api/videos/{rid}/reference/approve", json={}).json()
    assert v["status"] == "ready" and v["reference"]["approved"]
    steps = {s["key"]: s["output"] for s in h.store.steps(rid)}
    frames = [steps[f"key:s{i}:r0:k0"]["path"] for i in (1, 2, 3)]
    assert len(set(frames)) == 3 and not any("ref_" in f for f in frames), "one distinct opening frame per shot"
    assert client.get(f"/media/{rid}/key_s1_r0_k0.jpg").status_code == 200
    assert client.post(f"/api/videos/{rid}/reference/approve", json={}).status_code == 409
    assert h.store.total_cost(rid) == pytest.approx(2 * 0.01 + 3 * 0.01 + 9 * 0.04)  # 2 references, 3 keyframes, 9s

    # changing the look after filming needs approval again and refilms every shot
    v = client.post(f"/api/videos/{rid}/reference/redo", json={}).json()
    assert v["status"] == "review_reference"
    v = client.post(f"/api/videos/{rid}/reference/approve", json={}).json()
    assert v["status"] == "ready" and all(s["clip_url"].endswith("_r1_a0.mp4") for s in v["shots"])

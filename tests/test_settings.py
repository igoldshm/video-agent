"""Settings, and continuity with deployments made under the project's earlier name."""
import os

from fastapi.testclient import TestClient

from directorcut.api import create_app
from directorcut.harness import Harness
from directorcut.planner import HeuristicPlanner
from directorcut.providers import MockProvider, default_image_models, default_providers
from directorcut.settings import env, parse_model


def demo(home):
    return Harness(str(home), providers={"fastgen": MockProvider("fastgen", 0.04)}, planner=HeuristicPlanner(),
                   judge=None)


def test_old_setting_names_still_work_and_new_names_win(monkeypatch):
    monkeypatch.setenv("CUTROOM_BUDGET_USD", "7")
    assert env("BUDGET_USD") == "7"
    monkeypatch.setenv("DIRECTORCUT_BUDGET_USD", "9")
    assert env("BUDGET_USD") == "9"
    monkeypatch.setenv("DIRECTORCUT_BUDGET_USD", "")  # an empty new setting must not hide a filled old one
    assert env("BUDGET_USD") == "7"


def test_access_code_set_under_the_old_name_is_still_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTROOM_ACCESS_CODE", "letmein")
    client = TestClient(create_app(demo(tmp_path / "home"), sync=True))
    assert client.get("/api/config").json()["needs_code"] is True
    assert client.post("/api/videos", json={"brief": "A calm film about tea"}).status_code == 401
    assert client.post("/api/session", json={}, headers={"X-Access-Code": "letmein"}).status_code == 200


def test_a_placeholder_price_does_not_stop_the_service(monkeypatch):
    """Seen live: a setting still containing the word PRICE crashed the app on start."""
    assert parse_model("acme/video=PRICE", 0.05) == ("acme/video", 0.05)
    assert parse_model("acme/video=0.02", 0.05) == ("acme/video", 0.02)
    assert parse_model("acme/video", 0.05) == ("acme/video", 0.05) and parse_model("  ", 0.05) is None
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")
    monkeypatch.setenv("CUTROOM_VIDEO_MODELS", "acme/i2v=PRICE,acme/t2v=0.03")
    monkeypatch.setenv("DIRECTORCUT_IMAGE_MODEL", "acme/t2i=PRICE")
    assert [(p.model, p.price_per_s) for p in default_providers().values()] == [("acme/i2v", 0.05), ("acme/t2v", 0.03)]
    assert default_image_models()["reference"].price_per_image == 0.04


def test_videos_made_before_the_rename_are_still_there(tmp_path):
    home = tmp_path / "home"
    before = demo(home)
    os.rename(home / "directorcut.db", home / "cutroom.db")  # what an existing deployment has on disk
    rid = Harness(str(home), providers=before.providers, planner=HeuristicPlanner(), judge=None).create("A calm film")
    after = demo(home)
    assert after.store.get_run(rid)["brief"] == "A calm film" and not (home / "directorcut.db").exists()

import pytest


@pytest.fixture(autouse=True)
def no_live_keys(monkeypatch):
    """Tests never reach a real API: live adapters are exercised against stubbed HTTP."""
    for var in ("REPLICATE_API_TOKEN", "ANTHROPIC_API_KEY", "CUTROOM_ACCESS_CODE", "CUTROOM_VIDEO_MODELS",
                "CUTROOM_DEMO_DELAY", "CUTROOM_DAILY_VIDEOS", "CUTROOM_DAILY_USD"):
        monkeypatch.delenv(var, raising=False)

import pytest

NAMES = ("ACCESS_CODE", "VIDEO_MODELS", "IMAGE_MODEL", "KEYFRAME_MODEL", "DEMO_DELAY", "DEMO_REFERENCE",
         "DAILY_VIDEOS", "DAILY_USD", "BUDGET_USD", "HOME", "LLM_MODEL")


@pytest.fixture(autouse=True)
def no_live_keys(monkeypatch):
    """Tests never reach a real API: live adapters are exercised against stubbed HTTP."""
    for var in ("REPLICATE_API_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for name in NAMES:
        for prefix in ("DIRECTORCUT_", "CUTROOM_"):
            monkeypatch.delenv(prefix + name, raising=False)

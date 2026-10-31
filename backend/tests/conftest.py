import os

import pytest

# Tests must be deterministic and offline: never call a real LLM.
os.environ["GROQ_API_KEY"] = ""
os.environ["LLM_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("LLM_ENABLED", "false")
    yield
    get_settings.cache_clear()

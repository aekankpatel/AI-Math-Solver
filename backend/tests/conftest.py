import os

import pytest

# Tests must be deterministic and offline: never call a real LLM.
os.environ["GROQ_API_KEY"] = ""
os.environ["LLM_ENABLED"] = "false"
# Computations run inline in tests (the sandbox has its own tests, which start real workers),
# and request limits are off unless a test turns them on.
os.environ["SANDBOX_ENABLED"] = "false"
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
os.environ["RATE_LIMIT_PER_DAY"] = "0"


def reset_runtime_state() -> None:
    from app import sandbox, service
    from app.budget import get_budget
    from app.config import get_settings

    get_settings.cache_clear()
    get_budget.cache_clear()
    service.get_client_limits.cache_clear()
    sandbox.shutdown_runner()


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "false")
    reset_runtime_state()
    yield
    reset_runtime_state()

"""LLM provider factory. Groq is the default provider; the model is configurable.

The LLM is optional: without ``GROQ_API_KEY`` the application runs fully on
its deterministic rule-based planner and template explanations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


@lru_cache(maxsize=4)
def _build(model: str, temperature: float, timeout: float, retries: int, key: str) -> Any:
    from langchain_groq import ChatGroq

    # The Groq SDK retries 408/409/429/5xx with exponential backoff and honours Retry-After.
    return ChatGroq(model=model, temperature=temperature, timeout=timeout, max_retries=retries, api_key=key)


# Sent instead of a key when GROQ_AUTH=proxy. Not a secret: the egress proxy replaces
# the Authorization header for api.groq.com with the configured network secret.
PROXY_AUTH_PLACEHOLDER = "proxy-injected-credential"


def api_key_for_sdk() -> str | None:
    s = get_settings()
    if s.groq_auth == "proxy":
        return PROXY_AUTH_PLACEHOLDER
    return s.groq_api_key


def get_chat_model() -> Any | None:
    s = get_settings()
    if not s.llm_configured:
        return None
    return _build(s.llm_model, s.llm_temperature, s.llm_timeout_seconds, s.llm_max_retries, api_key_for_sdk() or "")

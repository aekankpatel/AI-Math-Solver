"""LLM provider factory. Groq is the default provider; the model is configurable.

The LLM is optional: without ``GROQ_API_KEY`` the application runs fully on
its deterministic rule-based planner and template explanations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


@lru_cache(maxsize=4)
def _build(model: str, temperature: float, timeout: float, key: str) -> Any:
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, timeout=timeout, max_retries=1, api_key=key)


def get_chat_model() -> Any | None:
    s = get_settings()
    if not s.llm_configured:
        return None
    return _build(s.llm_model, s.llm_temperature, s.llm_timeout_seconds, s.groq_api_key or "")

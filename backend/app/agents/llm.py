"""LLM provider factory. Groq is the default provider; the model is configurable.

The LLM is optional: without ``GROQ_API_KEY`` the application runs fully on
its deterministic rule-based planner and template explanations.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from app.config import get_settings

# Latest x-ratelimit-* response headers from Groq (limits and remaining quota; no credentials).
# Groq: *-requests = requests per DAY, *-tokens = tokens per MINUTE.
_RATE_LOCK = threading.Lock()
RATE_LIMITS: dict[str, str] = {}


def _capture_rate_limits(response: Any) -> None:
    headers = {
        k.lower(): v for k, v in response.headers.items() if k.lower().startswith(("x-ratelimit", "retry-after"))
    }
    if headers:
        with _RATE_LOCK:
            RATE_LIMITS.update(headers)


def total_tokens(message: Any) -> int | None:
    """Total tokens Groq reported for one response message (None if absent)."""
    usage = getattr(message, "usage_metadata", None) or {}
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    meta = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
    return int(meta["total_tokens"]) if meta.get("total_tokens") is not None else None


# Called with the token count of every successful Groq response (set by the API service so the
# deployment-wide budget sees real usage; unset in the benchmark and in tests).
_USAGE_LISTENER: list[Any] = []
_FAILURE_LISTENER: list[Any] = []


def set_usage_listener(fn: Any | None, on_failure: Any | None = None) -> None:
    _USAGE_LISTENER[:] = [fn] if fn else []
    _FAILURE_LISTENER[:] = [on_failure] if on_failure else []


def charge_usage(message: Any) -> None:
    for fn in list(_USAGE_LISTENER):
        fn(total_tokens(message))


def report_failure(exc: BaseException) -> None:
    """Tell the budget about a failed tutor call (rate limits pause the LLM)."""
    for fn in list(_FAILURE_LISTENER):
        fn(f"{type(exc).__name__}: {exc}")


def rate_limits() -> dict[str, str]:
    with _RATE_LOCK:
        return dict(RATE_LIMITS)


@lru_cache(maxsize=4)
def _build(model: str, temperature: float, timeout: float, retries: int, key: str) -> Any:
    import httpx
    from langchain_groq import ChatGroq

    # httpx honours HTTPS_PROXY and the CA bundle from the environment (trust_env); the response hook
    # only records rate-limit headers. The Groq SDK retries 408/409/429/5xx and honours Retry-After.
    client = httpx.Client(timeout=timeout, event_hooks={"response": [_capture_rate_limits]})
    return ChatGroq(
        model=model, temperature=temperature, timeout=timeout, max_retries=retries, api_key=key, http_client=client
    )


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

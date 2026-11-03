"""Runtime configuration loaded from environment variables / a local .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str | None = None
    # "api_key" (default, local development): the SDK sends GROQ_API_KEY.
    # "proxy": an egress proxy injects the Authorization header for api.groq.com
    # (e.g. a cloud network secret); no key is stored in this process.
    groq_auth: Literal["api_key", "proxy"] = "api_key"
    llm_model: str = "openai/gpt-oss-120b"  # any Groq model with tool calling; see README
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1  # SDK-level retries (rate limits, transient errors)
    # Set to false to force the deterministic rule-based planner even when a key exists.
    llm_enabled: bool = True
    # Optional LLM tutor explanations (one extra Groq call per solved problem). Off by default:
    # template explanations are always available and the free-tier quota goes to planning.
    llm_explanations: bool = False
    # Structured-output method for the planner: "function_calling" (tool call) or "json_schema"
    # (Groq strict structured outputs, supported by the openai/gpt-oss models).
    llm_structured_method: Literal["function_calling", "json_schema"] = "function_calling"

    # ---- global Groq budget (rolling 24 h, whole deployment). Groq free tier for
    # openai/gpt-oss-120b: 200k tokens/day, 1,000 requests/day, 8k tokens/minute.
    llm_daily_token_budget: int = 150_000
    llm_daily_request_budget: int = 800
    llm_tokens_per_call_estimate: int = 2_500  # charged for a call without a usage report
    llm_failure_threshold: int = 3  # consecutive failures that pause the LLM ...
    llm_failure_cooldown_seconds: float = 120.0  # ... for this long
    llm_budget_state_path: str | None = None  # persist the ledger (JSON) across restarts

    # ---- per-client limits (in memory; run a single API process)
    rate_limit_per_minute: int = 10
    rate_limit_per_day: int = 300
    llm_calls_per_client_per_day: int = 40  # beyond this a client is served offline
    trusted_proxy_hops: int = 0  # >0 only behind proxies that append X-Forwarded-For

    # ---- computation sandbox
    sandbox_enabled: bool = True
    sandbox_workers: int = 2
    sandbox_timeout_seconds: float = 10.0
    sandbox_memory_mb: int = 1536
    sandbox_queue_timeout_seconds: float = 15.0
    sandbox_max_jobs_per_worker: int = 200

    # ---- HTTP hardening
    max_request_bytes: int = 16_384
    enable_docs: bool = False  # /docs and /openapi.json
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    max_sessions: int = 500
    log_level: str = "INFO"

    @property
    def llm_configured(self) -> bool:
        if not self.llm_enabled:
            return False
        return self.groq_auth == "proxy" or bool(self.groq_api_key)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

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
    llm_max_retries: int = 2  # SDK-level retries (rate limits, transient errors)
    # Set to false to force the deterministic rule-based planner even when a key exists.
    llm_enabled: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    max_sessions: int = 500

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

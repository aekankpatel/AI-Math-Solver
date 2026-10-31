"""Runtime configuration loaded from environment variables / a local .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 30.0
    # Set to false to force the deterministic rule-based planner even when a key exists.
    llm_enabled: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    max_sessions: int = 500

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_enabled and self.groq_api_key)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

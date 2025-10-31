from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
        frozen=True,
    )

    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", alias="ANTHROPIC_MODEL")
    anthropic_base_url: Optional[str] = Field(default=None, alias="ANTHROPIC_BASE_URL")
    anthropic_max_output_tokens: int = Field(default=1024, alias="ANTHROPIC_MAX_TOKENS")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    session_dir: Path = Field(default=Path("storage/sessions/default"), alias="SESSION_DIR")
    artifacts_dir: Path = Field(default=Path("storage/artifacts"), alias="ARTIFACTS_DIR")
    logs_dir: Path = Field(default=Path("storage/logs"), alias="LOGS_DIR")
    allow_domains: List[str] = Field(default_factory=list, alias="ALLOW_DOMAINS")
    token_budget: int = Field(default=3500, alias="TOKENS_BUDGET")
    max_steps: int = Field(default=40, alias="MAX_STEPS")
    max_concurrent_tasks: int = Field(default=2, alias="MAX_CONCURRENT_TASKS")
    task_timeout_seconds: int = Field(default=1800, alias="TASK_TIMEOUT_SECONDS")
    step_timeout_seconds: int = Field(default=120, alias="STEP_TIMEOUT_SECONDS")
    sse_heartbeat_seconds: int = Field(default=10, alias="SSE_HEARTBEAT_SECONDS")

    @field_validator("session_dir", "artifacts_dir", "logs_dir", mode="before")
    def _expand_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("allow_domains", mode="before")
    def _split_domains(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


def get_settings() -> Settings:
    """Load settings once and reuse (singleton style)."""

    return Settings()

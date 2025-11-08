from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from dotenv import load_dotenv

load_dotenv()


LLMProvider = Literal["openai", "anthropic"]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Application configuration loaded from environment variables."""

    llm_provider: LLMProvider = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    allowlist_hosts: tuple[str, ...] = field(default_factory=lambda: ("duckduckgo.com",))
    max_steps: int = 40
    step_timeout_s: int = 30
    run_timeout_s: int = 600
    allow_js_eval: bool = False
    allow_file_upload: bool = False
    headless_default: bool = True
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    runs_dir: Path = Path("runs")
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        llm_raw = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        llm_provider: LLMProvider = "openai" if llm_raw not in {"openai", "anthropic"} else llm_raw  # type: ignore[assignment]

        allowlist_env = os.getenv("ALLOWLIST_HOSTS", "duckduckgo.com")
        allowlist = tuple(
            sorted({entry.strip() for entry in allowlist_env.split(",") if entry.strip()})
        )

        settings = cls(
            llm_provider=llm_provider,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            allowlist_hosts=allowlist,
            max_steps=int(os.getenv("MAX_STEPS", "40")),
            step_timeout_s=int(os.getenv("STEP_TIMEOUT_S", "30")),
            run_timeout_s=int(os.getenv("RUN_TIMEOUT_S", "600")),
            allow_js_eval=_bool_env("ALLOW_JS_EVAL", False),
            allow_file_upload=_bool_env("ALLOW_FILE_UPLOAD", False),
            headless_default=_bool_env("HEADLESS_DEFAULT", True),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
            runs_dir=Path(os.getenv("RUNS_DIR", "runs")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )
        return settings

    def ensure_directories(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def update_allowlist(self, hosts: Sequence[str]) -> None:
        cleaned = tuple(sorted({host.strip() for host in hosts if host.strip()}))
        if cleaned:
            self.allowlist_hosts = cleaned

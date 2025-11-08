from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class RunOptions(BaseModel):
    provider: str | None = Field(default=None, pattern=r"^(openai|anthropic)$")
    headless: bool | None = True
    allowlist: list[str] | None = None
    max_steps: int | None = Field(default=None, ge=1, le=100)
    timeout: int | None = Field(default=None, ge=10, le=900)
    capture_screenshots: bool = True


class RunRequest(BaseModel):
    task: str = Field(..., min_length=4)
    start_url: HttpUrl | None = None
    options: RunOptions | None = None


class EventPayload(BaseModel):
    event: str
    data: dict[str, Any]

from __future__ import annotations

from dataclasses import replace
from typing import AsyncIterator, Tuple

import orjson
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import EventSourceResponse

from ..browser.sandbox import SandboxPolicy
from ..config import Settings
from ..errors import SandboxViolation
from ..llm.anthropic_client import AnthropicClient
from ..llm.base import LLMClient
from ..llm.openai_client import OpenAIClient
from ..logging import setup_logging
from ..core.agent import Agent
from ..core.planner import Planner
from ..types import RunResult
from .schemas import EventPayload, RunRequest

app = FastAPI(title="Browser Agent API")


def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_directories()
    setup_logging(settings.log_level, settings.log_dir / "agent.log")
    return settings


def build_llm(provider: str, settings: Settings) -> LLMClient:
    if provider == "openai":
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        return OpenAIClient(settings.openai_api_key)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        return AnthropicClient(settings.anthropic_api_key)
    raise HTTPException(status_code=400, detail=f"Unsupported provider {provider}")


def build_sandbox(settings: Settings) -> SandboxPolicy:
    return SandboxPolicy.from_hosts(
        hosts=settings.allowlist_hosts,
        max_steps=settings.max_steps,
        step_timeout_s=settings.step_timeout_s,
        run_timeout_s=settings.run_timeout_s,
        allow_js_eval=settings.allow_js_eval,
        allow_file_upload=settings.allow_file_upload,
    )


def apply_options(base_settings: Settings, request: RunRequest) -> Tuple[Settings, bool, bool | None]:
    options = request.options
    settings = replace(base_settings)
    if options and options.provider:
        settings.llm_provider = options.provider  # type: ignore[assignment]
    if options and options.allowlist:
        settings.allowlist_hosts = tuple(sorted({host.strip() for host in options.allowlist if host.strip()}))
    if options and options.max_steps:
        settings.max_steps = options.max_steps
    if options and options.timeout:
        settings.run_timeout_s = options.timeout
    headless_override = options.headless if options else None
    capture = True if not options else options.capture_screenshots
    return settings, capture, headless_override


async def perform_run(request: RunRequest, base_settings: Settings) -> RunResult:
    settings, capture, headless_override = apply_options(base_settings, request)
    sandbox = build_sandbox(settings)
    llm_client = build_llm(settings.llm_provider, settings)
    planner = Planner(llm_client)
    agent = Agent(
        planner=planner,
        settings=settings,
        sandbox=sandbox,
        runs_dir=settings.runs_dir,
        capture_screenshots=capture,
    )
    try:
        result = await agent.run(
            task=request.task,
            start_url=str(request.start_url) if request.start_url else None,
            headless=headless_override,
        )
    finally:
        if hasattr(llm_client, "close"):
            await getattr(llm_client, "close")()
    return result


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run_endpoint(
    request: RunRequest,
    settings: Settings = Depends(get_settings),
    stream: bool = Query(default=False),
):
    if stream:
        async def event_stream() -> AsyncIterator[str]:
            result = await perform_run(request, settings)
            payload = EventPayload(event="result", data=result.model_dump())
            body = orjson.dumps(payload.model_dump()).decode()
            yield f"event: {payload.event}\ndata: {body}\n\n"

        return EventSourceResponse(event_stream())

    result = await perform_run(request, settings)
    return result.model_dump()

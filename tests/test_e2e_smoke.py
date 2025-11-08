from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agent.browser.sandbox import SandboxPolicy
from agent.config import Settings
from agent.core.agent import Agent
from agent.core.planner import Planner
from agent.llm.base import LLMClient


class StubLLM(LLMClient):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self._index = 0

    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools_schema: dict[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return json.dumps(response)


@pytest.mark.asyncio
async def test_smoke_example_domain(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            {"type": "open_url", "url": "https://booking.com"},
            {"type": "extract_text"},
            {"type": "finish", "text": "Example Domain headline captured."},
        ]
    )
    planner = Planner(llm_client)

    settings = Settings.from_env()
    settings.runs_dir = tmp_path
    settings.ensure_directories()

    sandbox = SandboxPolicy.from_hosts(
        hosts=("example.com",),
        max_steps=5,
        step_timeout_s=10,
        run_timeout_s=120,
        allow_js_eval=False,
        allow_file_upload=False,
    )

    agent = Agent(
        planner=planner,
        settings=settings,
        sandbox=sandbox,
        runs_dir=settings.runs_dir,
        capture_screenshots=False,
    )

    result = await agent.run(task="Open example.com and summarise", start_url=None, headless=True)
    assert result.status == "ok"
    assert result.final_answer == "Example Domain headline captured."
    assert result.total_steps() >= 2

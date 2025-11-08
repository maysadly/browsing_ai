from __future__ import annotations

import json
import logging
from typing import Any

from ..errors import ParsingError
from ..llm.base import LLMClient
from ..llm.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from ..types import Action, Observation, json_repair

logger = logging.getLogger(__name__)


class Planner:
    """LLM-powered planner that maps observations to actions."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def decide_action(
        self,
        task: str,
        observation: Observation,
        memory_context: str,
        sandbox_summary: str,
    ) -> Action:
        messages: list[dict[str, Any]] = []
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["observation"]})
            messages.append({"role": "assistant", "content": json.dumps(example["action"])})

        user_content = (
            f"Task: {task}\n"
            f"Sandbox: {sandbox_summary}\n"
            f"Observation:\n{observation.summary()}\n"
            f"Memory:\n{memory_context or '[empty]'}"
        )
        messages.append({"role": "user", "content": user_content})

        schema = Action.model_json_schema()
        raw = await self._llm.complete(
            system=SYSTEM_PROMPT,
            messages=messages,
            tools_schema=schema,
            max_tokens=512,
            temperature=0.0,
        )
        logger.debug("LLM raw response: %s", raw)
        try:
            return json_repair(raw, Action)
        except ParsingError:
            logger.exception("Failed to parse LLM response")
            raise

from __future__ import annotations

import json
import logging
from typing import Any

from ..errors import ParsingError
from ..llm.base import LLMClient
from ..llm.prompts import FEW_SHOT_EXAMPLES, PLAN_SYSTEM_PROMPT, SYSTEM_PROMPT
from ..types import Action, Observation, PlanDraft, PlanItem, json_repair

logger = logging.getLogger(__name__)


class Planner:
    """LLM-powered planner that maps observations to actions."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def create_plan(self, task: str, observation: Observation | None = None) -> list[PlanItem]:
        """Request a structured execution plan for the given task."""

        task = task.strip()
        if not task:
            return []

        context_lines = [f"Task: {task}"]
        if observation is not None:
            context_lines.append("Initial context:")
            context_lines.append(observation.summary())

        messages: list[dict[str, Any]] = [{"role": "user", "content": "\n".join(context_lines)}]
        schema = PlanDraft.model_json_schema()

        raw = await self._llm.complete(
            system=PLAN_SYSTEM_PROMPT,
            messages=messages,
            tools_schema=schema,
            max_tokens=512,
            temperature=0.1,
        )
        logger.debug("LLM raw plan response: %s", raw)
        try:
            draft = json_repair(raw, PlanDraft)
        except ParsingError:
            logger.exception("Failed to parse plan response")
            raise

        plan_items: list[PlanItem] = []
        for index, item in enumerate(draft.plan, start=1):
            title = item.title.strip()
            if not title:
                continue
            notes: list[str] = []
            if item.success_criteria:
                criteria = item.success_criteria.strip()
                if criteria:
                    notes.append(criteria)
            plan_items.append(PlanItem(index=index, title=title, notes=notes))

        return plan_items

    async def decide_action(
        self,
        task: str,
        observation: Observation,
        memory_context: str,
        sandbox_summary: str,
        blocker_hint: str,
    ) -> Action:
        messages: list[dict[str, Any]] = []
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["observation"]})
            messages.append({"role": "assistant", "content": json.dumps(example["action"])})

        user_content = (
            f"Task: {task}\n"
            f"Sandbox: {sandbox_summary}\n"
            f"Evaluator hints: {blocker_hint}\n"
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

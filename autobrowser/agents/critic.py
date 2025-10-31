from __future__ import annotations

from typing import List, Optional

from loguru import logger

from autobrowser.core.schemas import Observation, Task
from autobrowser.llm.provider import OpenAIProvider


class CriticAgent:
    """Analyses failures and proposes recovery steps."""

    def __init__(self, llm: OpenAIProvider) -> None:
        self._llm = llm
        self._system_prompt = llm.prompt_registry.load("critic.system.txt")

    def suggest_recovery(
        self,
        task: Task,
        history: List[str],
        observation: Optional[Observation],
    ) -> str:
        observation_text = observation.note if observation else "No observation captured."
        messages = [
            {
                "role": "user",
                "content": (
                    f"Goal: {task.goal}\n"
                    f"Observation: {observation_text}\n"
                    f"Recent history:\n" + "\n".join(f"- {entry}" for entry in history[-5:])
                ),
            }
        ]
        response = self._llm.chat(system=self._system_prompt, messages=messages, temperature=0.4)
        suggestion = response["choices"][0]["message"].get("content", "").strip()
        logger.info("Critic: suggestion generated", task_id=task.id, suggestion=suggestion)
        return suggestion

from __future__ import annotations

from typing import List

from loguru import logger

from autobrowser.core.schemas import PageEntity, Task
from autobrowser.llm.provider import AnthropicProvider


class ContentAgent:
    """Generates concise summaries and final reports."""

    def __init__(self, llm: AnthropicProvider) -> None:
        self._llm = llm
        self._system_prompt = llm.prompt_registry.load("writer.system.txt")

    def summarize_entities(self, task: Task, entities: List[PageEntity], words: int = 120) -> str:
        bullet_points = "\n".join(
            f"- {entity.title}: {entity.summary}" for entity in entities if entity.summary
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Goal: {task.goal}\n"
                    f"Write a concise summary under {words} words.\n"
                    f"Focus on the main findings.\n\n"
                    f"Context:\n{bullet_points}"
                ),
            }
        ]
        logger.info(
            "Content agent: summarizing entities",
            task_id=task.id,
            entities=len(entities),
            words=words,
        )
        response = self._llm.chat(system=self._system_prompt, messages=messages, temperature=0.3)
        return response["choices"][0]["message"].get("content", "").strip()

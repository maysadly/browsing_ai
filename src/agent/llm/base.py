from __future__ import annotations

import abc
from typing import Any


class LLMClient(abc.ABC):
    """Abstract base class representing a language model client."""

    @abc.abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools_schema: dict[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return the next action payload as a JSON string."""

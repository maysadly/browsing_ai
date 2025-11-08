from __future__ import annotations

import logging
from typing import Any

import httpx

from ..errors import LLMError
from .base import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Minimal OpenAI Chat Completions client enforcing JSON output."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            timeout=httpx.Timeout(30.0, read=60.0),
        )

    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools_schema: dict[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "choose_action", "parameters": tools_schema},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "choose_action"}},
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = await self._client.post("/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            logger.exception("OpenAI completion failed: %s", exc)
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        data = response.json()
        message = data["choices"][0]["message"]
        if message.get("tool_calls"):
            return message["tool_calls"][0]["function"]["arguments"]
        content = message.get("content")
        if isinstance(content, list) and content:
            return content[0].get("text", "")
        if isinstance(content, str):
            return content
        return "{}"

    async def close(self) -> None:
        await self._client.aclose()

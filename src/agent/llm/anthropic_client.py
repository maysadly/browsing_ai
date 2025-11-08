from __future__ import annotations

import logging
from typing import Any

import httpx

from ..errors import LLMError
from .base import LLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Anthropic Messages API client returning JSON-only responses."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620") -> None:
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            timeout=httpx.Timeout(30.0, read=60.0),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
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
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_schema", "json_schema": tools_schema},
        }
        try:
            response = await self._client.post("/messages", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            logger.exception("Anthropic completion failed: %s", exc)
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        data = response.json()
        content = data.get("content", [])
        if content and content[0].get("type") == "tool_result":
            return content[0].get("content", "")
        if content and content[0].get("type") == "text":
            return content[0].get("text", "")
        return "{}"

    async def close(self) -> None:
        await self._client.aclose()

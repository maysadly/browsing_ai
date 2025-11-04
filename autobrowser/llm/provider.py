from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    from autobrowser.core.config import Settings
else:
    Settings = Any  # type: ignore[misc]

try:
    from autobrowser.core.config import get_settings
except ModuleNotFoundError:  # pragma: no cover
    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")


class PromptRegistry:
    """Loads reusable system prompts from disk."""

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self._base_path = base_path or Path(__file__).resolve().parent / "prompts"
        self._cache: Dict[str, str] = {}

    def load(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        path = self._base_path / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt '{name}' not found at {path}")
        content = path.read_text(encoding="utf-8").strip()
        self._cache[name] = content
        return content


class OpenAIProvider:
    """Adapter around the OpenAI SDK providing chat completion helpers."""

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        *,
        prompt_registry: Optional[PromptRegistry] = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required to use OpenAIProvider")
        self._settings = settings or get_settings()
        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )
        self.model = self._settings.openai_model
        self.prompt_registry = prompt_registry or PromptRegistry()

    def chat(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        logger.info(
            "LLM chat request",
            model=self.model,
            messages=len(messages),
            has_tools=bool(tools),
            temperature=temperature,
        )
        payload_messages: List[Dict[str, Any]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        try:
            response = self._chat_with_retry(
                model=self.model,
                messages=payload_messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            logger.info("LLM chat response received", model=self.model)
            return response.model_dump()
        except Exception:
            logger.exception("LLM chat request failed", model=self.model)
            raise

    def count_tokens(self, text: str) -> int:
        """Proxy to the token counter helper."""

        from autobrowser.core.memory import TokenCounter

        counter = TokenCounter(self.model)
        return counter.count_text(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
    def _chat_with_retry(self, **payload: Any):
        logger.trace("LLM retry attempt", model=self.model)
        return self._client.chat.completions.create(**payload)

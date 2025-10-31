from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from typing import TYPE_CHECKING

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


class AnthropicProvider:
    """Adapter around the Anthropic SDK providing chat completion helpers."""

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        *,
        prompt_registry: Optional[PromptRegistry] = None,
    ) -> None:
        if Anthropic is None:
            raise RuntimeError("anthropic package is required to use AnthropicProvider")
        self._settings = settings or get_settings()
        client_kwargs: Dict[str, Any] = {"api_key": self._settings.anthropic_api_key}
        if self._settings.anthropic_base_url:
            client_kwargs["base_url"] = self._settings.anthropic_base_url
        self._client = Anthropic(**client_kwargs)
        self.model = self._settings.anthropic_model
        self._default_max_output_tokens = self._settings.anthropic_max_output_tokens
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
        conversations, system_prompt = self._prepare_messages(system, messages)
        tool_definitions = self._prepare_tools(tools)
        max_tokens = max_output_tokens or self._default_max_output_tokens
        try:
            response = self._chat_with_retry(
                model=self.model,
                system=system_prompt,
                messages=conversations,
                tools=tool_definitions,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            logger.info("LLM chat response received", model=self.model)
            return self._convert_response(response)
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
        cleaned = {key: value for key, value in payload.items() if value is not None}
        return self._client.messages.create(**cleaned)

    def _prepare_messages(
        self,
        system: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        system_prompts: List[str] = []
        if system:
            system_prompts.append(system)

        prepared: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if isinstance(content, str) and content.strip():
                    system_prompts.append(content.strip())
                continue
            if role == "tool":
                prepared.append(self._convert_tool_result_message(message))
                continue
            if role not in {"user", "assistant"}:
                logger.warning("Planner: unsupported role, coercing to user", role=role)
                role = "user"
            prepared.append(
                {
                    "role": role,
                    "content": self._to_blocks(content),
                }
            )

        system_prompt = "\n".join(filter(None, system_prompts)) or None
        return prepared, system_prompt

    def _prepare_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            converted.append(
                {
                    "name": function.get("name", "dispatch"),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted or None

    def _to_blocks(self, content: Any) -> List[Dict[str, str]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            blocks: List[Dict[str, str]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    blocks.append({"type": "text", "text": item.get("text", "")})
                else:
                    blocks.append({"type": "text", "text": json.dumps(item)})
            return blocks or [{"type": "text", "text": ""}]
        return [{"type": "text", "text": str(content)}]

    def _convert_tool_result_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        tool_use_id = message.get("tool_call_id") or message.get("id") or "tool-result"
        content = message.get("content") or ""
        if isinstance(content, list):
            text = "\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in content
            )
        else:
            text = str(content)
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": text,
                }
            ],
        }

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                arguments = getattr(block, "input", {})
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", ""),
                            "arguments": json.dumps(arguments),
                        },
                    }
                )

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage = getattr(response, "usage", None)
        usage_payload: Optional[Dict[str, Any]] = None
        if usage is not None:
            try:
                usage_payload = usage.model_dump()
            except AttributeError:  # pragma: no cover
                usage_payload = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                }

        return {
            "id": getattr(response, "id", ""),
            "model": getattr(response, "model", self.model),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": getattr(response, "stop_reason", None),
                }
            ],
            "usage": usage_payload,
        }

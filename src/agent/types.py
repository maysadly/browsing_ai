from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Literal, TypeVar

import orjson
from pydantic import BaseModel, Field, ValidationError

from .errors import ParsingError

ActionType = Literal[
    "open_url",
    "click",
    "type",
    "press",
    "select",
    "scroll",
    "wait",
    "extract_text",
    "screenshot",
    "evaluate_js",
    "upload_file",
    "finish",
    "abort",
]


class Action(BaseModel):
    """Single browser operation produced by planner."""

    type: ActionType
    selector: str | None = None
    text: str | None = None
    keys: str | None = None
    url: str | None = None
    js: str | None = None
    timeout_ms: int | None = Field(default=None, ge=100)
    scroll: Literal["up", "down", "to_top", "to_bottom"] | None = None

    model_config = {"extra": "forbid"}


class InteractiveElement(BaseModel):
    tag: str
    role: str | None = None
    text: str | None = None
    selector: str | None = None
    attributes: dict[str, str | None] = Field(default_factory=dict)


class Observation(BaseModel):
    url: str | None
    title: str | None
    visible_text: str
    interactive_elements: list[InteractiveElement]
    errors: list[str]
    page_size: tuple[int, int] | None = None
    network_idle: bool = False
    screenshot_path: str | None = None

    def summary(self) -> str:
        parts = [
            f"URL: {self.url or 'unknown'}",
            f"Title: {self.title or 'unknown'}",
            "Visible text snippet:",
            self.visible_text[:800],
        ]
        if self.interactive_elements:
            parts.append("Interactive elements:")
            for element in self.interactive_elements[:15]:
                descriptor = (
                    f"- {element.tag} role={element.role or '-'} text={element.text!r} selector={element.selector}"
                )
                parts.append(descriptor)
        if self.errors:
            parts.append("Recent errors: " + "; ".join(self.errors[:5]))
        return "\n".join(parts)


class AgentStep(BaseModel):
    step_index: int
    observation: Observation
    action: Action
    result_summary: str
    artifacts: list[str]


RunStatus = Literal["ok", "failed", "aborted", "needs_human"]


class RunResult(BaseModel):
    steps: list[AgentStep]
    status: RunStatus
    final_answer: str | None = None
    error: str | None = None

    def total_steps(self) -> int:
        return len(self.steps)


T = TypeVar("T", bound=BaseModel)


class JSONRepair:
    COMMON_REPLACEMENTS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r",\s*([}\]])"), r"\1"),
        (re.compile(r"(\{|\[)\s*,"), r"\1"),
        (re.compile(r"\bNone\b"), "null"),
    ]

    @staticmethod
    def _normalise_quotes(text: str) -> str:
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")

        def repl(match: re.Match[str]) -> str:
            return f'"{match.group(1)}": "{match.group(2)}"'

        text = re.sub(r"'([^']+)'\s*:\s*'([^']*)'", repl, text)
        text = text.replace("'", '"')
        return text

    @classmethod
    def repair(cls, payload: str) -> str:
        content = payload.strip()
        if content.startswith("```"):
            content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = cls._normalise_quotes(content)
        for pattern, replacement in cls.COMMON_REPLACEMENTS:
            content = pattern.sub(replacement, content)
        if "{" in content and not content.strip().startswith("{"):
            content = content[content.index("{") :]
        if "}" in content and not content.strip().endswith("}"):
            content = content[: content.rindex("}") + 1]
        return content


def json_repair(payload: str, model: type[T]) -> T:
    cleaned = JSONRepair.repair(payload)
    attempts = [cleaned]
    brace_delta = cleaned.count("{") - cleaned.count("}")
    if brace_delta > 0:
        attempts.append(cleaned + "}" * brace_delta)
    elif brace_delta < 0:
        attempts.append("{" * (-brace_delta) + cleaned)

    for attempt in attempts:
        try:
            parsed = orjson.loads(attempt)
            return model.model_validate(parsed)
        except (orjson.JSONDecodeError, ValidationError):  # pragma: no cover - try fallback
            continue
        except TypeError as exc:
            raise ParsingError(f"Invalid JSON structure: {exc}") from exc

    try:
        parsed = json.loads(cleaned)
        return model.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:  # pragma: no cover - diagnostics
        raise ParsingError(f"Failed to repair JSON payload: {payload}") from exc

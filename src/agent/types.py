from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Literal, TypeVar

import orjson
from pydantic import BaseModel, Field, ValidationError, model_validator

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
    "await_user_login",
    "switch_tab",
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
    message: str | None = None
    reason: str = Field(..., min_length=1)
    plan_status: Literal["pending", "in_progress", "done", "ongoing"] | None = None
    tab_index: int | None = Field(default=None, ge=0)
    tab_url_contains: str | None = None
    tab_title_contains: str | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _normalise_plan_status(self) -> "Action":
        if self.plan_status == "ongoing":
            self.plan_status = "in_progress"
        return self

    @model_validator(mode="after")
    def _validate_switch_tab(self) -> "Action":
        if self.type == "switch_tab":
            if (
                self.tab_index is None
                and not self.tab_url_contains
                and not self.tab_title_contains
            ):
                raise ValueError(
                    "switch_tab requires tab_index or tab_url_contains/tab_title_contains hint"
                )
        return self


class TabSnapshot(BaseModel):
    index: int
    url: str | None = None
    title: str | None = None
    is_active: bool = False


class OverlaySnapshot(BaseModel):
    selector: str | None = None
    text: str | None = None
    title: str | None = None
    width: float | None = None
    height: float | None = None


class InteractiveElement(BaseModel):
    tag: str
    role: str | None = None
    text: str | None = None
    selector: str | None = None
    attributes: dict[str, str | None] = Field(default_factory=dict)
    viewport_top: float | None = None
    viewport_bottom: float | None = None
    viewport_left: float | None = None
    viewport_right: float | None = None
    viewport_visible_ratio: float | None = None
    under_modal: bool = False


class ModalSnapshot(BaseModel):
    selector: str | None = None
    title: str | None = None
    text: str | None = None
    width: float | None = None
    height: float | None = None
    top: float | None = None
    right: float | None = None
    bottom: float | None = None
    left: float | None = None
    actions: list[InteractiveElement] = Field(default_factory=list)
    fields: list[InteractiveElement] = Field(default_factory=list)


class ScrollMetrics(BaseModel):
    scroll_top: float = Field(alias="scrollTop")
    scroll_height: float = Field(alias="scrollHeight")
    client_height: float = Field(alias="clientHeight")
    progress: float

    model_config = {"populate_by_name": True}


class Observation(BaseModel):
    url: str | None
    title: str | None
    visible_text: str
    interactive_elements: list[InteractiveElement]
    errors: list[str]
    page_size: tuple[int, int] | None = None
    network_idle: bool = False
    screenshot_path: str | None = None
    scroll: ScrollMetrics | None = None
    open_tabs: list[TabSnapshot] = Field(default_factory=list)
    overlays: list[OverlaySnapshot] = Field(default_factory=list)
    modal: ModalSnapshot | None = None

    def summary(self) -> str:
        parts = [
            f"URL: {self.url or 'unknown'}",
            f"Title: {self.title or 'unknown'}",
            "Visible text snippet:",
            self.visible_text[:800],
        ]
        if self.open_tabs:
            parts.append("Open tabs:")
            for tab in self.open_tabs[:6]:
                marker = "*" if tab.is_active else "-"
                label = tab.title or tab.url or "untitled"
                parts.append(f"{marker} [{tab.index}] {label}")
        if self.overlays:
            parts.append("Detected overlays:")
            for overlay in self.overlays[:3]:
                label = overlay.selector or "overlay"
                size = ""
                if overlay.width and overlay.height:
                    size = f" ({overlay.width:.0f}x{overlay.height:.0f})"
                context = overlay.title or (overlay.text[:80] if overlay.text else "")
                if context:
                    parts.append(f"- {label}{size} -> {context}")
                else:
                    parts.append(f"- {label}{size}")
        if self.modal:
            modal = self.modal
            title = modal.title or modal.selector or "modal"
            location = ""
            if modal.width and modal.height:
                location = f" ({modal.width:.0f}x{modal.height:.0f})"
            parts.append(f"Active modal: {title}{location}")
            if modal.text:
                parts.append(modal.text[:400])
            if modal.actions:
                parts.append("Modal actions:")
                for action in modal.actions[:6]:
                    parts.append(
                        f"- {action.tag} selector={action.selector} text={action.text!r}"
                    )
            if modal.fields:
                parts.append("Modal fields:")
                for field in modal.fields[:6]:
                    parts.append(
                        f"- {field.tag} selector={field.selector} text={field.text!r}"
                    )
        if self.scroll is not None:
            parts.append(
                "Scroll metrics: "
                f"progress={self.scroll.progress * 100:.1f}% "
                f"(top={self.scroll.scroll_top:.0f} / height={self.scroll.scroll_height:.0f}, viewport={self.scroll.client_height:.0f})"
            )
        if self.interactive_elements:
            parts.append("Interactive elements:")
            for element in self._interactive_sample(limit=18):
                descriptor = (
                    f"- {element.tag} role={element.role or '-'} text={element.text!r} selector={element.selector}"
                )
                if element.under_modal:
                    descriptor += " [modal]"
                if element.viewport_top is not None:
                    descriptor += f" (top={element.viewport_top:.0f})"
                parts.append(descriptor)
        if self.errors:
            parts.append("Recent errors: " + "; ".join(self.errors[:5]))
        return "\n".join(parts)

    def _interactive_sample(self, limit: int) -> list[InteractiveElement]:
        if not self.interactive_elements:
            return []

        def sort_key(element: InteractiveElement) -> tuple[float, float]:
            top = element.viewport_top if element.viewport_top is not None else float("inf")
            left = element.viewport_left if element.viewport_left is not None else 0.0
            return (top, left)

        seen: set[tuple[str | None, str | None]] = set()
        sorted_elements = sorted(self.interactive_elements, key=sort_key)

        keyword_selectors = (
            "vacancy",
            "serp",
            "apply",
            "response",
            "resume",
            "favorite",
            "blacklist",
        )
        keyword_texts = (
            "отклик",
            "вакан",
            "связа",
            "apply",
            "резюм",
            "respond",
            "contact",
        )

        head: list[InteractiveElement] = []
        priority: list[InteractiveElement] = []
        others: list[InteractiveElement] = []

        for element in sorted_elements:
            signature = (element.selector, element.text)
            if signature in seen:
                continue
            seen.add(signature)
            selector_lower = (element.selector or "").lower()
            text_lower = (element.text or "").lower()
            matches_priority = any(token in selector_lower for token in keyword_selectors) or any(
                token in text_lower for token in keyword_texts
            )

            if len(head) < max(6, limit // 3):
                head.append(element)
                if matches_priority:
                    priority.append(element)
                else:
                    others.append(element)
                continue

            if matches_priority:
                priority.append(element)
            else:
                others.append(element)

        sample: list[InteractiveElement] = []
        for element in head:
            if element not in sample:
                sample.append(element)
            if len(sample) >= limit:
                return sample[:limit]

        for element in priority:
            if element not in sample:
                sample.append(element)
            if len(sample) >= limit:
                return sample[:limit]

        for element in others:
            if element not in sample:
                sample.append(element)
            if len(sample) >= limit:
                return sample[:limit]

        return sample[:limit]


class AgentStep(BaseModel):
    step_index: int
    observation: Observation
    action: Action
    result_summary: str
    artifacts: list[str]


class PlanDraftItem(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    success_criteria: str | None = Field(default=None, max_length=400)


class PlanDraft(BaseModel):
    plan: list[PlanDraftItem]


PlanStatus = Literal["pending", "in_progress", "done"]


class PlanItem(BaseModel):
    index: int
    title: str
    status: PlanStatus = "pending"
    notes: list[str] = Field(default_factory=list)

    def mark_in_progress(self) -> None:
        if self.status == "pending":
            self.status = "in_progress"

    def mark_done(self, note: str | None = None) -> None:
        self.status = "done"
        if note:
            self.notes.append(note)


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
    def sanitize_keys(data: Any) -> Any:
        if isinstance(data, dict):
            sanitized: dict[str, Any] = {}
            for key, value in data.items():
                new_key = key
                if isinstance(key, str):
                    new_key = key.strip()
                    while new_key.endswith(":") or new_key.endswith("="):
                        new_key = new_key[:-1]
                    new_key = new_key.replace(" ", "_")
                sanitized[new_key] = sanitize_keys(value)
            return sanitized
        if isinstance(data, list):
            return [sanitize_keys(item) for item in data]
        return data

    def ensure_reason_field(data: Any) -> Any:
        if not isinstance(data, dict) or "reason" in data:
            return data

        patched = dict(data)
        fallback: str | None = None
        action_type = patched.get("type")
        if action_type == "finish" and patched.get("text"):
            fallback = str(patched.get("text"))
        elif action_type == "await_user_login" and patched.get("message"):
            fallback = str(patched.get("message"))
        elif action_type in {"abort", "wait"} and patched.get("text"):
            fallback = str(patched.get("text"))
        elif action_type == "scroll":
            fallback = "Adjusting scroll position to surface new content."
        elif patched.get("message"):
            fallback = str(patched.get("message"))
        elif patched.get("text"):
            fallback = str(patched.get("text"))
        if not fallback:
            fallback = "Auto-generated rationale to satisfy schema requirements."
        patched["reason"] = fallback
        return patched

    try:
        parsed = orjson.loads(payload)
        parsed = sanitize_keys(parsed)
        parsed = ensure_reason_field(parsed)
        return model.model_validate(parsed)
    except (orjson.JSONDecodeError, ValidationError, TypeError):
        pass

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
            parsed = sanitize_keys(parsed)
            parsed = ensure_reason_field(parsed)
            return model.model_validate(parsed)
        except (orjson.JSONDecodeError, ValidationError):  # pragma: no cover - try fallback
            continue
        except TypeError as exc:
            raise ParsingError(f"Invalid JSON structure: {exc}") from exc

    try:
        parsed = json.loads(cleaned)
        parsed = sanitize_keys(parsed)
        parsed = ensure_reason_field(parsed)
        return model.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:  # pragma: no cover - diagnostics
        raise ParsingError(f"Failed to repair JSON payload: {payload}") from exc

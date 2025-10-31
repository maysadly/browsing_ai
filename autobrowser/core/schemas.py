from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class RiskLevel(str, Enum):
    """Simple risk classification for browser actions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


ActionType = Literal[
    "navigate",
    "query",
    "click",
    "type",
    "select",
    "wait",
    "scroll",
    "extract",
    "confirm_gate",
]


class Action(BaseModel):
    """Structured action understood by the browser tools layer."""

    type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class Affordance(BaseModel):
    """High level, human readable interactive element description."""

    role: Optional[str] = None
    name: Optional[str] = None
    locator: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Outcome of executing an action on the page."""

    ok: bool
    note: str
    affordances: List[Affordance] = Field(default_factory=list)
    text_chunks: List[str] = Field(default_factory=list)
    screenshot_path: Optional[str] = None
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    error_category: Optional[str] = None


class PageFormField(BaseModel):
    """Structured representation for a form input."""

    label: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None
    type: Optional[str] = None
    locator: Optional[str] = None


class PageEntity(BaseModel):
    """Entity extracted from the page (e.g. search results, table rows)."""

    title: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[HttpUrl] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class PageStatePack(BaseModel):
    """Compact snapshot of the browser page to feed into the LLM."""

    url: str
    title: str
    goals: List[str] = Field(default_factory=list)
    affordances: List[Affordance] = Field(default_factory=list)
    text_snippets: List[str] = Field(default_factory=list)
    forms: List[List[PageFormField]] = Field(default_factory=list)
    entities: List[PageEntity] = Field(default_factory=list)
    screenshot_path: Optional[str] = None


class StepLog(BaseModel):
    """Record of a single planning/execution step."""

    id: str
    task_id: str
    timestamp: datetime
    role: Literal["planner", "browser", "reader", "critic", "reporter", "safety"]
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    note: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskStatus(str, Enum):
    """Lifecycle state of orchestration tasks."""

    NEW = "New"
    RUNNING = "Running"
    NEEDS_CONFIRM = "NeedsConfirm"
    DONE = "Done"
    ERROR = "Error"


class TaskLogEvent(BaseModel):
    """Event pushed to streaming clients."""

    id: str
    task_id: str
    timestamp: datetime
    kind: Literal[
        "status",
        "step",
        "confirm_request",
        "artifact",
        "error",
        "info",
        "plan",
        "summary",
        "recovery",
        "heartbeat",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Persistent representation of a long running autonomous task."""

    id: str
    goal: str
    created_at: datetime
    status: TaskStatus = TaskStatus.NEW
    risk_level: RiskLevel = RiskLevel.MEDIUM
    allow_high_risk: bool = False
    logs: List[TaskLogEvent] = Field(default_factory=list)
    completed_at: Optional[datetime] = None
    result_summary: Optional[str] = None
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def append_event(self, event: TaskLogEvent) -> None:
        self.logs.append(event)


class ConfirmPayload(BaseModel):
    """Description of a high risk action waiting for approval."""

    task_id: str
    step_id: str
    action: Action
    summary: str
    screenshot_path: Optional[str] = None
    form_preview: Optional[Dict[str, Any]] = None


class SafetyDecision(BaseModel):
    """Safety layer response for a candidate action."""

    risk_level: RiskLevel
    requires_confirmation: bool
    reason: str


class ToolCall(BaseModel):
    """LLM tool invocation description."""

    name: str
    arguments: Dict[str, Any]

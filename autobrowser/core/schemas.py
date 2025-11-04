from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional

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
    "submit",
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
    extra: Dict[str, Any] = Field(default_factory=dict)


class PageEntity(BaseModel):
    """Entity extracted from the page (e.g. search results, table rows)."""

    title: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[HttpUrl] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class BoundingBox(BaseModel):
    """Simple rectangle representation for on-screen element geometry."""

    x: float
    y: float
    width: float
    height: float


class DistilledNode(BaseModel):
    """Lightweight DOM element description used for planning."""

    node_id: int
    path: str
    tag: str
    role: Optional[str] = None
    label: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    locator: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    visible: bool = True
    hash: str


class NavigationHint(BaseModel):
    """Score assigned to an interactive element considered by the navigator."""

    label: str
    locator: Optional[str] = None
    score: float = 0.0
    source: Literal["affordance", "distilled"]
    bbox: Optional[BoundingBox] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class DomChangeDetail(BaseModel):
    """Describes how a single DOM node changed between snapshots."""

    path: str
    locator: Optional[str] = None
    summary: str
    fields_changed: List[str] = Field(default_factory=list)


class DomSnapshotDiff(BaseModel):
    """Delta between two consecutive distilled DOM snapshots."""

    current_hash: str
    previous_hash: Optional[str] = None
    added: List[DomChangeDetail] = Field(default_factory=list)
    removed: List[DomChangeDetail] = Field(default_factory=list)
    changed: List[DomChangeDetail] = Field(default_factory=list)


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
    annotated_screenshot_path: Optional[str] = None
    distilled_elements: List[DistilledNode] = Field(default_factory=list)
    navigation_hints: List[NavigationHint] = Field(default_factory=list)
    dom_hash: Optional[str] = None
    dom_diff: Optional[DomSnapshotDiff] = None


@dataclass
class TaskContext:
    """Lightweight representation of the current task state for agents."""

    goal: str
    plan: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    steps_remaining: int = 0
    seconds_remaining: float = 0.0
    token_budget_remaining: int = 0

    def update_plan(self, steps: Iterable[str], current: Optional[str]) -> None:
        self.plan = list(steps)
        self.current_step = current

    def consume_tokens(self, amount: int) -> None:
        if amount <= 0:
            return
        remainder = self.token_budget_remaining - amount
        self.token_budget_remaining = remainder if remainder > 0 else 0


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


class TaskContract(BaseModel):
    """Defines goal success criteria, metrics, and limits for a task."""

    success_condition: str = Field(default="Complete the assigned goal without errors.")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)


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
    contract: TaskContract = Field(default_factory=TaskContract)

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

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings
else:
    Settings = Any  # type: ignore[misc]

try:
    from .config import get_settings
except ModuleNotFoundError:  # pragma: no cover
    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")
from .schemas import Action, RiskLevel, SafetyDecision, Task


HIGH_RISK_TYPES = {"confirm_gate"}
MEDIUM_RISK_TYPES = {"type", "select", "extract", "wait"}
LOW_RISK_TYPES = {"navigate", "query", "scroll", "click"}


@dataclass
class SafetyContext:
    """Lightweight structure to expose policy decisions."""

    settings: "Settings"


class SafetyEngine:
    """Determines whether actions can proceed immediately or require approval."""

    def __init__(self, settings: Optional["Settings"] = None) -> None:
        self._settings = settings or get_settings()

    def classify_action(self, action: Action) -> RiskLevel:
        action_type = action.type

        intent = str(action.params.get("intent", "")).lower()
        if action_type in HIGH_RISK_TYPES or "submit" in intent or "delete" in intent:
            return RiskLevel.HIGH
        if action_type in MEDIUM_RISK_TYPES:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def assess(self, task: Task, action: Action, *,
               description: str = "") -> SafetyDecision:
        """Return safety decision on an action for a task."""

        risk_level = self.classify_action(action)
        requires_confirmation = bool(action.params.get("force_confirm", False))
        reason = "Low risk navigation."

        if risk_level is RiskLevel.MEDIUM:
            reason = "Medium risk interaction; continuing automatically."
        elif risk_level is RiskLevel.HIGH:
            reason = description or "High risk action requires confirmation."
            requires_confirmation = not task.allow_high_risk

        if (
            self._settings.allow_domains
            and action.type == "navigate"
            and "url" in action.params
        ):
            hostname = urlparse(action.params["url"]).hostname or ""
            if hostname and hostname not in self._settings.allow_domains:
                requires_confirmation = True
                reason = (
                    f"Domain '{hostname}' outside of allow list; require confirmation."
                )

        decision = SafetyDecision(
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            reason=reason,
        )
        try:
            from loguru import logger

            logger.info(
                "Safety assessment",
                task_id=task.id,
                action=action.type,
                risk=risk_level.value,
                requires_confirmation=requires_confirmation,
            )
        except Exception:  # pragma: no cover
            pass
        return decision

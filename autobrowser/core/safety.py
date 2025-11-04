from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Any, TYPE_CHECKING, Deque, Dict
from urllib.parse import urlparse
import time

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
        self._click_timestamps: Dict[str, Deque[float]] = defaultdict(deque)

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
        force_flag = bool(action.params.get("force_confirm", False))
        requires_confirmation = force_flag
        reason = "Low risk navigation."

        if action.type == "submit":
            risk_level = RiskLevel.HIGH
            if not task.allow_high_risk:
                requires_confirmation = True
                reason = "Form submission requires human confirmation."
            else:
                reason = "Form submission allowed by policy."

        if risk_level is RiskLevel.MEDIUM:
            reason = "Medium risk interaction; continuing automatically."
        elif risk_level is RiskLevel.HIGH:
            reason = description or "High risk action requires confirmation."
            requires_confirmation = requires_confirmation or not task.allow_high_risk

        if action.type == "navigate" and "url" in action.params:
            parsed = urlparse(action.params["url"])
            if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
                requires_confirmation = True
                risk_level = RiskLevel.HIGH
                reason = "Blocked navigation to unsupported URL scheme."

            if getattr(self._settings, "allow_domains", []):
                hostname = parsed.hostname or ""
                if hostname and hostname not in self._settings.allow_domains:
                    requires_confirmation = True
                    reason = (
                        f"Domain '{hostname}' outside of allow list; require confirmation."
                    )

        if action.type == "click":
            limit = int(getattr(self._settings, "max_clicks_per_second", 0) or 0)
            if limit > 0:
                now = time.time()
                timestamps = self._click_timestamps[task.id]
                while timestamps and now - timestamps[0] > 1.0:
                    timestamps.popleft()
                timestamps.append(now)
                if len(timestamps) > limit:
                    requires_confirmation = True
                    reason = (
                        f"Click rate exceeded {limit}/s; wait before retrying."
                    )

        if force_flag and not description and risk_level is not RiskLevel.HIGH:
            reason = "Confirmation required by caller."

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

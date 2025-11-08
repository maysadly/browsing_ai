from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Action, Observation, RunStatus


@dataclass(slots=True)
class EvaluationResult:
    continue_run: bool
    status: RunStatus
    final_answer: str | None = None
    error: str | None = None
    replan: bool = False


class StepEvaluator:
    """Simple heuristics for detecting completion, loops, or blockers."""

    CAPTCHA_PATTERNS = (
        re.compile(r"captcha", re.IGNORECASE),
        re.compile(r"verify you are", re.IGNORECASE),
    )

    def evaluate(
        self,
        action: Action,
        observation_before: Observation,
        observation_after: Observation,
        previous_excerpt: str | None,
    ) -> EvaluationResult:
        if action.type == "finish":
            return EvaluationResult(continue_run=False, status="ok", final_answer=action.text)
        if action.type == "abort":
            return EvaluationResult(continue_run=False, status="aborted", error=action.text)

        text = observation_after.visible_text
        if any(pattern.search(text) for pattern in self.CAPTCHA_PATTERNS):
            return EvaluationResult(
                continue_run=False,
                status="needs_human",
                final_answer="CAPTCHA detected",
            )

        excerpt = text[:200]
        if previous_excerpt is not None and previous_excerpt == excerpt:
            return EvaluationResult(continue_run=True, status="ok", replan=True)

        return EvaluationResult(continue_run=True, status="ok")

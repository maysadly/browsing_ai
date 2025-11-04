from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from autobrowser.core.schemas import (
    Action,
    Observation,
    PageStatePack,
    Task,
    TaskContext,
)

from .navigator import Navigator
from .strategy import StrategyAgent


class ExecutorAgent:
    """Bridges high-level plan steps with concrete browser actions."""

    def __init__(
        self,
        strategy: StrategyAgent,
        *,
        max_failures: int = 3,
        affordance_limit: int = 18,
        snippet_limit: int = 6,
        navigator: Optional[Navigator] = None,
    ) -> None:
        self._strategy = strategy
        self._max_failures = max_failures
        self._affordance_limit = affordance_limit
        self._snippet_limit = snippet_limit
        self._failures: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._failure_reasons: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
        self._navigator = navigator or Navigator()

    def decide_action(
        self,
        task: Task,
        page_state: PageStatePack,
        short_history: List[str],
        rag_snippets: List[str],
        *,
        plan_step: Optional[str],
        context: Optional[TaskContext] = None,
    ):
        plan_key = self._plan_key(plan_step)
        failure_count = self._failures[task.id].get(plan_key, 0)
        failure_reason = self._failure_reasons.get(task.id, {}).get(plan_key)
        user_inputs: Dict[str, Any] = task.metadata.get("user_inputs", {})  # type: ignore[assignment]

        if failure_reason == "hidden_input" and failure_count >= 1 and not user_inputs:
            logger.info(
                "Executor: gating due to hidden input",
                task_id=task.id,
                plan_step=plan_step,
            )
            intent = "Provide verification code or credentials"
            note = "Hidden input encountered; requesting user input [step-blocked]"
            return Action(type="confirm_gate", params={"intent": intent}), note

        filtered_state = self._filter_page_state(task, page_state, plan_step)

        action, note = self._strategy.decide_next_action(
            task,
            filtered_state,
            short_history,
            rag_snippets,
            plan_step=plan_step,
            failure_count=failure_count,
            failure_reason=failure_reason,
            user_inputs=user_inputs or None,
            context=context,
        )
        return action, note

    def record_failure(
        self,
        task_id: str,
        plan_step: Optional[str],
        observation: Optional[Observation] = None,
    ) -> None:
        key = self._plan_key(plan_step)
        self._failures[task_id][key] = self._failures[task_id].get(key, 0) + 1
        self._failure_reasons[task_id][key] = self._classify_failure(observation)
        failures = self._failures[task_id][key]
        if failures >= self._max_failures:
            logger.warning(
                "Executor: repeated failures for plan step",
                task_id=task_id,
                step=plan_step,
                failures=failures,
            )

    def record_success(self, task_id: str, plan_step: Optional[str]) -> None:
        key = self._plan_key(plan_step)
        if task_id in self._failures and key in self._failures[task_id]:
            del self._failures[task_id][key]
            if not self._failures[task_id]:
                del self._failures[task_id]
        if task_id in self._failure_reasons and key in self._failure_reasons[task_id]:
            del self._failure_reasons[task_id][key]
            if not self._failure_reasons[task_id]:
                del self._failure_reasons[task_id]

    def clear_step(self, task_id: str, plan_step: Optional[str]) -> None:
        self.record_success(task_id, plan_step)

    def reset_task(self, task_id: str) -> None:
        self._failures.pop(task_id, None)
        self._failure_reasons.pop(task_id, None)

    # ------------------------------------------------------------------ utils

    def _filter_page_state(
        self,
        task: Task,
        page_state: PageStatePack,
        plan_step: Optional[str],
    ) -> PageStatePack:
        keywords = self._extract_keywords(task.goal, plan_step)

        affordances = self._navigator.rank_affordances(
            page_state.affordances,
            keywords=keywords,
        )
        snippets = self._filter_snippets(page_state.text_snippets, keywords)
        navigation_hints = self._navigator.collect_candidates(page_state, keywords=keywords)

        filtered = page_state.model_copy(deep=True)
        filtered.affordances = affordances[: self._affordance_limit]
        filtered.text_snippets = snippets[: self._snippet_limit]
        filtered.navigation_hints = navigation_hints[: self._navigator.max_candidates]
        return filtered

    def _filter_snippets(self, snippets: Iterable[str], keywords: List[str]) -> List[str]:
        if not keywords:
            return list(snippets)
        filtered: List[str] = []
        for snippet in snippets:
            text = snippet.lower()
            if any(keyword in text for keyword in keywords):
                filtered.append(snippet)
        return filtered or list(snippets)

    def _extract_keywords(self, *segments: Optional[str]) -> List[str]:
        tokens: List[str] = []
        for segment in segments:
            if not segment:
                continue
            words = re.findall(r"[a-zA-Zа-яА-Я0-9]+", segment.lower())
            tokens.extend(word for word in words if len(word) > 2)
        seen = set()
        unique: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique

    def _plan_key(self, plan_step: Optional[str]) -> str:
        return (plan_step or "__default__").strip().lower()

    def _classify_failure(self, observation: Optional[Observation]) -> Optional[str]:
        if observation is None:
            return None
        note = (observation.note or "").lower()
        if observation.error_category in {"HandlerException", "Exception"}:
            if "input of type \"radio\"" in note:
                return "radio_input"
            if "type=\"hidden\"" in note or "hidden" in note:
                return "hidden_input"
        if observation.error_category == "HiddenInput":
            return "hidden_input"
        if observation.error_category == "Timeout" and "code" in note:
            return "otp_timeout"
        return observation.error_category or None

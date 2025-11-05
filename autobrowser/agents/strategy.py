from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from loguru import logger
from pydantic import ValidationError

from autobrowser.core.schemas import (
    Action,
    Affordance,
    PageEntity,
    PageFormField,
    PageStatePack,
    Task,
    TaskContext,
)
from autobrowser.core.heuristics import HeuristicPattern, HeuristicStore
from autobrowser.llm.provider import OpenAIProvider
from .navigator import Navigator


@dataclass
class WorkflowState:
    authenticated: bool = False
    auth_prompt_visible: bool = False
    profile_reviewed: bool = False
    search_started: bool = False
    applications_submitted: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "authenticated": self.authenticated,
            "auth_prompt_visible": self.auth_prompt_visible,
            "profile_reviewed": self.profile_reviewed,
            "search_started": self.search_started,
            "applications_submitted": self.applications_submitted,
        }


class StrategyAgent:
    """Unified high-level agent that plans actions, repairs tool calls, and generates recoveries."""

    _REQUIRED_FIELDS: Dict[str, List[str]] = {
        "navigate": ["url"],
        "query": ["selector_or_locator"],
        "extract": ["locator"],
        "click": ["locator_or_text"],
        "type": ["locator"],
        "confirm_gate": ["intent"],
    }

    def __init__(
        self,
        llm: OpenAIProvider,
        *,
        navigator: Optional[Navigator] = None,
        heuristics: Optional[HeuristicStore] = None,
    ) -> None:
        self._llm = llm
        self._navigator = navigator or Navigator()
        registry = llm.prompt_registry
        self._plan_prompt = registry.load("planner.system.txt")
        self._recovery_prompt = registry.load("critic.system.txt")
        self._report_prompt = registry.load("writer.system.txt")
        self._review_prompt = registry.load("reviewer.system.txt")
        self._missing_action_prompt = (
            "You are fixing a planner output that omitted a tool call. "
            "Return exactly one JSON object with keys `type` and `params` suitable for dispatching to browser tools. "
            "If no action is possible, respond with `{\"type\": \"wait\", \"params\": {\"reason\": \"<short note>\"}}`."
        )
        self._repair_prompt = (
            "You repair incomplete tool calls for a browser automation agent. "
            "Always return a JSON object with keys `type` and `params`. "
            "Include every mandatory parameter (url, locator, selector, etc.). "
            "Prefer durable selectors (data attributes, role+name) and fully qualified URLs. "
            "Do not add explanation outside the JSON object."
        )
        self._max_repair_attempts = 2
        self._workflow: Dict[str, WorkflowState] = {}
        if heuristics is not None:
            self._heuristics = heuristics
        else:
            root = Path(__file__).resolve().parents[2]
            heuristics_path = root / "storage" / "heuristics.json"
            self._heuristics = HeuristicStore(heuristics_path)

    def reset_task(self, task_id: str) -> None:
        """Clear cached workflow state for a completed task."""
        self._workflow.pop(task_id, None)

    # ------------------------------------------------------------------ planning

    def build_initial_plan(
        self,
        task: Task,
        *,
        context: Optional[TaskContext] = None,
    ) -> List[str]:
        user_message = (
            "Goal: {goal}\n"
            "Provide a numbered list (3-6 bullet points) of major steps to solve the goal."
        ).format(goal=task.goal)
        if context:
            user_message += (
                f"\nMax steps remaining: {context.steps_remaining}"
                f"\nToken budget remaining: {context.token_budget_remaining}"
            )

        logger.info("Strategy: building initial plan", task_id=task.id)
        response = self._llm.chat(
            system=self._plan_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.1,
        )
        self._consume_tokens(context, user_message)
        content = response["choices"][0]["message"].get("content", "")
        lines = [line.strip(" -") for line in content.splitlines() if line.strip()]
        plan = [line for line in lines if line]
        logger.info("Strategy: initial plan parsed", task_id=task.id, steps=len(plan))
        return plan

    def decide_next_action(
        self,
        task: Task,
        page_state: PageStatePack,
        short_history: List[str],
        rag_snippets: List[str],
        *,
        plan_step: Optional[str] = None,
        failure_count: int = 0,
        failure_reason: Optional[str] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> Tuple[Action, str]:
        workflow_state = self._update_workflow_state(task, page_state, short_history)
        content = self._compose_planner_prompt(
            task,
            page_state,
            short_history,
            rag_snippets,
            plan_step=plan_step,
            failure_count=failure_count,
            failure_reason=failure_reason,
            user_inputs=user_inputs,
            context=context,
            workflow_state=workflow_state,
        )
        self._consume_tokens(context, content)

        logger.info(
            "Strategy: deciding next action",
            task_id=task.id,
            history=len(short_history),
            rag=len(rag_snippets),
        )
        action, planner_note = self._run_planner_dialogue(
            task,
            page_state,
            short_history,
            rag_snippets,
            plan_step=plan_step,
            failure_count=failure_count,
            failure_reason=failure_reason,
            user_inputs=user_inputs,
            context=context,
            workflow_state=workflow_state,
            planner_payload=content,
        )

        review_comment, revised_action, approved = self._review_action(
            task,
            page_state,
            action,
            planner_note,
            plan_step=plan_step,
            context=context,
        )

        final_action = revised_action or action
        combined_note = self._merge_notes(planner_note, review_comment, approved)

        logger.info(
            "Strategy: multi-agent dialogue",
            task_id=task.id,
            planner_note=planner_note,
            reviewer_comment=review_comment,
            reviewer_approved=approved,
            action=final_action.type,
        )

        final_action = self._postprocess_action(
            task,
            final_action,
            page_state,
            short_history,
            rag_snippets,
            plan_step=plan_step,
            failure_count=failure_count,
            failure_reason=failure_reason,
            user_inputs=user_inputs,
            context=context,
            workflow_state=workflow_state,
        )
        return final_action, combined_note

    # ------------------------------------------------------------------ recovery / reporting

    def _run_planner_dialogue(
        self,
        task: Task,
        page_state: PageStatePack,
        short_history: List[str],
        rag_snippets: List[str],
        *,
        plan_step: Optional[str],
        failure_count: int,
        failure_reason: Optional[str],
        user_inputs: Optional[Dict[str, Any]],
        context: Optional[TaskContext],
        workflow_state: WorkflowState,
        planner_payload: str,
    ) -> Tuple[Action, str]:
        response = self._llm.chat(
            system=self._plan_prompt,
            messages=[{"role": "user", "content": planner_payload}],
            tools=[self._browser_action_tool_schema()],
            temperature=0.1,
        )

        choice = response["choices"][0]["message"]
        note = choice.get("content") or ""
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            logger.warning(
                "Strategy: planner returned no tool call; invoking repair helper",
                task_id=task.id,
                finish_reason=response["choices"][0].get("finish_reason"),
            )
            action = self._repair_missing_action(
                task,
                page_state,
                short_history,
                rag_snippets,
                planner_message=choice,
                plan_step=plan_step,
                failure_count=failure_count,
                failure_reason=failure_reason,
                user_inputs=user_inputs,
                context=context,
                workflow_state=workflow_state,
            )
            return action, note

        arguments = tool_calls[0]["function"].get("arguments", "{}")
        payload = json.loads(arguments)
        action = Action(type=payload["type"], params=payload.get("params", {}))
        logger.info("Strategy: action selected", task_id=task.id, action=action.type, note=note)
        return action, note

    def _review_action(
        self,
        task: Task,
        page_state: PageStatePack,
        action: Action,
        planner_note: str,
        *,
        plan_step: Optional[str],
        context: Optional[TaskContext],
    ) -> Tuple[str, Optional[Action], bool]:
        payload: Dict[str, Any] = {
            "goal": task.goal,
            "current_url": page_state.url,
            "page_title": page_state.title,
            "action": action.model_dump(),
            "planner_note": planner_note,
        }
        if plan_step:
            payload["plan_step"] = plan_step
        message = json.dumps(payload, ensure_ascii=False)
        self._consume_tokens(context, message)

        response = self._llm.chat(
            system=self._review_prompt,
            messages=[{"role": "user", "content": message}],
            temperature=0.1,
        )
        reviewer_message = response["choices"][0]["message"].get("content", "") or ""
        comment, revision, approved = self._parse_reviewer_response(reviewer_message)
        return comment, revision, approved

    def _parse_reviewer_response(self, raw_message: str) -> Tuple[str, Optional[Action], bool]:
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.debug("Strategy: reviewer response was not JSON", message=raw_message)
            return raw_message.strip(), None, False

        comment = str(data.get("comment") or "").strip()
        approved = bool(data.get("approval", False))
        revision_payload = data.get("revision")
        revision_action: Optional[Action] = None
        if isinstance(revision_payload, dict):
            action_type = revision_payload.get("type")
            params = revision_payload.get("params")
            if isinstance(action_type, str) and isinstance(params, dict):
                try:
                    revision_action = Action(type=action_type, params=params)
                except ValidationError:
                    logger.warning("Strategy: reviewer proposed invalid revision", revision=revision_payload)

        return comment or raw_message.strip(), revision_action, approved

    def _merge_notes(self, planner_note: str, reviewer_comment: str, approved: bool) -> str:
        sections: List[str] = []
        if planner_note and planner_note.strip():
            sections.append(planner_note.strip())
        if reviewer_comment and reviewer_comment.strip():
            prefix = "[reviewer-ok]" if approved else "[reviewer]"
            sections.append(f"{prefix} {reviewer_comment.strip()}")
        return "\n".join(sections)

    def suggest_recovery(
        self,
        task: Task,
        history: List[str],
        observation: Optional[str],
    ) -> str:
        message = (
            f"Goal: {task.goal}\n"
            f"Observation: {observation or 'No observation captured.'}\n"
            "Recent history:\n"
            + "\n".join(f"- {entry}" for entry in history[-5:])
        )
        response = self._llm.chat(
            system=self._recovery_prompt,
            messages=[{"role": "user", "content": message}],
            temperature=0.3,
        )
        suggestion = response["choices"][0]["message"].get("content", "").strip()
        logger.info("Strategy: recovery suggestion produced", task_id=task.id, suggestion=suggestion)
        return suggestion

    def summarize_entities(
        self,
        task: Task,
        entities: List[PageEntity],
        *,
        words: int = 120,
    ) -> str:
        bullet_points = "\n".join(
            f"- {entity.title}: {entity.summary}" for entity in entities if entity.summary
        )
        payload = (
            f"Goal: {task.goal}\n"
            f"Compose a concise summary (<= {words} words) using only the supplied findings.\n\n"
            f"Findings:\n{bullet_points or 'No extracted entities.'}"
        )
        response = self._llm.chat(
            system=self._report_prompt,
            messages=[{"role": "user", "content": payload}],
            temperature=0.25,
        )
        summary = response["choices"][0]["message"].get("content", "").strip()
        logger.info("Strategy: report generated", task_id=task.id, length=len(summary))
        return summary

    # ------------------------------------------------------------------ planner prompt composition

    def _compose_planner_prompt(
        self,
        task: Task,
        page_state: PageStatePack,
        short_history: List[str],
        rag_snippets: List[str],
        *,
        plan_step: Optional[str],
        failure_count: int,
        failure_reason: Optional[str],
        user_inputs: Optional[Dict[str, Any]],
        context: Optional[TaskContext],
        workflow_state: WorkflowState,
    ) -> str:
        chunks: List[str] = [
            f"Goal: {task.goal}",
            f"Current URL: {page_state.url}",
            f"Page title: {page_state.title}",
            "Recent steps:",
            *[f"- {entry}" for entry in short_history[-5:]],
        ]
        if plan_step:
            chunks.extend(
                [
                    "Plan focus:",
                    plan_step,
                    "Stay aligned with this step; request a new plan only if it is complete or impossible.",
                ]
            )
        if failure_count:
            chunks.append(f"Failures on this step: {failure_count}")
        if failure_reason:
            chunks.append(f"Last failure reason: {failure_reason}")
        if user_inputs:
            chunks.append("User-provided inputs:")
            chunks.extend(f"- {key}: {value}" for key, value in user_inputs.items())

        chunks.append("Workflow status:")
        for key, value in workflow_state.as_dict().items():
            chunks.append(f"- {key}: {value}")
        if workflow_state.auth_prompt_visible and not workflow_state.authenticated:
            chunks.append(
                "Login prompt detected: pause and issue `confirm_gate` with intent describing required credentials before proceeding."
            )
        if not workflow_state.profile_reviewed:
            chunks.append("Profile has not been reviewed yet; gather context before executing irreversible actions.")
        if workflow_state.applications_submitted >= 3:
            chunks.append("Three submissions recorded; consider producing a summary or finalising the task.")

        shortlist = self.shortlist_affordances(page_state, task.goal, limit=8, workflow_state=workflow_state)
        if shortlist:
            chunks.append("Priority elements:")
            chunks.extend(
                f"- {aff.role or 'element'} • {aff.name or 'unnamed'} • locator={aff.locator or 'n/a'}"
                for aff in shortlist
            )

        form_lines = self._format_forms(page_state.forms)
        if form_lines:
            chunks.append("Forms detected:")
            chunks.extend(form_lines)

        snippet_lines = self._format_snippets(page_state.text_snippets)
        if snippet_lines:
            chunks.append("On-page snippets:")
            chunks.extend(snippet_lines)

        diff_lines = self._format_dom_diff(page_state.dom_diff)
        if diff_lines:
            chunks.append("Recent DOM changes:")
            chunks.extend(diff_lines)

        if context:
            chunks.append("Task context:")
            chunks.append(f"- Steps remaining: {context.steps_remaining}")
            chunks.append(f"- Seconds remaining: {int(context.seconds_remaining)}")
            chunks.append(f"- Token budget remaining: {context.token_budget_remaining}")
            if context.plan:
                chunks.append("Plan outline:")
                chunks.extend(f"* {step}" for step in context.plan[:6])

        if rag_snippets:
            chunks.append("Relevant memory:")
            chunks.extend(f"- {item}" for item in rag_snippets[:5])

        return "\n".join(chunks)

    # ------------------------------------------------------------------ workflow state tracking

    def _update_workflow_state(
        self,
        task: Task,
        page_state: PageStatePack,
        history: List[str],
    ) -> WorkflowState:
        state = self._workflow.setdefault(task.id, WorkflowState())
        domain = self._domain_from_url(page_state.url)
        if self._detect_authenticated(page_state, domain):
            state.authenticated = True
        state.auth_prompt_visible = self._detect_auth_prompt(page_state, domain) and not state.authenticated
        if self._detect_profile_section(page_state):
            state.profile_reviewed = True
        if self._detect_search_activity(page_state, history):
            state.search_started = True
        state.applications_submitted = self._estimate_submissions(history, state.applications_submitted)
        return state

    def _detect_authenticated(self, page_state: PageStatePack, domain: Optional[str]) -> bool:
        if self._patterns_match("auth_indicator", page_state, domain):
            return True
        for snippet in page_state.text_snippets[:10]:
            lower = snippet.lower()
            if any(token in lower for token in ("logout", "log out", "выйти", "выйти из аккаунта")):
                return True
        title = (page_state.title or "").lower()
        return any(token in title for token in ("dashboard", "profile", "личный кабинет", "резюме"))

    def _detect_auth_prompt(self, page_state: PageStatePack, domain: Optional[str]) -> bool:
        if self._patterns_match("auth_prompt", page_state, domain):
            return True
        candidates = page_state.affordances[:20]
        for affordance in candidates:
            label = (affordance.name or "").lower()
            if any(token in label for token in ("login", "log in", "sign in", "войти", "авторизация")):
                return True
        return False

    def _detect_profile_section(self, page_state: PageStatePack) -> bool:
        for snippet in page_state.text_snippets[:20]:
            lower = snippet.lower()
            if any(token in lower for token in ("резюме", "профиль", "profile", "cv", "account settings")):
                return True
        entities = page_state.entities or []
        for entity in entities:
            title = (entity.title or "").lower()
            if any(token in title for token in ("profile", "resume", "резюме")):
                return True
        return False

    def _detect_search_activity(self, page_state: PageStatePack, history: List[str]) -> bool:
        entities = page_state.entities or []
        if entities and len(entities) >= 3:
            return True
        for snippet in page_state.text_snippets[:10]:
            lower = snippet.lower()
            if any(token in lower for token in ("results", "найдено", "ваканс", "results for")):
                return True
        for entry in history[-6:]:
            lower = entry.lower()
            if "typed" in lower or "search" in lower or "поиск" in lower:
                return True
        return False

    def _estimate_submissions(self, history: List[str], previous: int) -> int:
        keywords = ("applied", "responded", "отклик", "подал заявку", "submitted")
        count = 0
        for entry in history:
            lower = entry.lower()
            if any(keyword in lower for keyword in keywords):
                count += 1
        return max(previous, count)

    # ------------------------------------------------------------------ repair helpers

    def _postprocess_action(
        self,
        task: Task,
        action: Action,
        page_state: PageStatePack,
        history: List[str],
        rag: List[str],
        *,
        plan_step: Optional[str] = None,
        failure_count: int = 0,
        failure_reason: Optional[str] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
        workflow_state: Optional[WorkflowState] = None,
    ) -> Action:
        current = action
        attempts = 0
        while attempts <= self._max_repair_attempts:
            missing = self._detect_missing_fields(current)
            if not missing:
                return current
            logger.warning(
                "Strategy: repairing incomplete action",
                action=current.type,
                missing=missing,
                attempt=attempts + 1,
            )
            current = self._repair_action(
                task,
                page_state,
                history,
                rag,
                current,
                missing,
                plan_step=plan_step,
                failure_count=failure_count,
                failure_reason=failure_reason,
                user_inputs=user_inputs,
                context=context,
                workflow_state=workflow_state,
            )
            attempts += 1

        params = self._apply_heuristics(
            task,
            page_state,
            current,
            plan_step=plan_step,
            failure_count=failure_count,
            failure_reason=failure_reason,
            user_inputs=user_inputs,
            workflow_state=workflow_state,
        )
        final = Action(type=current.type, params=params)
        missing = self._detect_missing_fields(final)
        if missing:
            logger.error(
                "Strategy: unable to repair action after heuristics",
                action=final.type,
                missing=missing,
            )
            raise ValueError(f"Unable to repair action {final.type}: missing {missing}")
        logger.info("Strategy: action repaired via heuristics", action=final.type, params=final.params)
        logger.debug("Strategy: final action", action=final.model_dump())
        if workflow_state and not workflow_state.authenticated and workflow_state.auth_prompt_visible:
            if final.type != "confirm_gate":
                logger.info("Strategy: switching to confirm_gate due to authentication requirement")
                return Action(
                    type="confirm_gate",
                    params={
                        "intent": "Authenticate user to continue automation",
                    },
                )
        return final

    def _detect_missing_fields(self, action: Action) -> List[str]:
        requirements = self._REQUIRED_FIELDS.get(action.type, [])
        params = action.params or {}
        missing: List[str] = []
        for requirement in requirements:
            if requirement == "url" and not params.get("url"):
                missing.append("url")
            elif requirement == "selector_or_locator" and not (params.get("selector") or params.get("locator")):
                missing.append("selector/locator")
            elif requirement == "locator" and not params.get("locator"):
                missing.append("locator")
            elif requirement == "locator_or_text" and not (params.get("locator") or params.get("text")):
                missing.append("locator/text")
            elif requirement == "intent" and not params.get("intent"):
                missing.append("intent")
        return missing

    def _repair_action(
        self,
        task: Task,
        page_state: PageStatePack,
        history: List[str],
        rag: List[str],
        action: Action,
        missing: List[str],
        *,
        plan_step: Optional[str] = None,
        failure_count: int = 0,
        failure_reason: Optional[str] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
        workflow_state: Optional[WorkflowState] = None,
    ) -> Action:
        payload_ctx = {
            "goal": task.goal,
            "current_url": page_state.url,
            "page_title": page_state.title,
            "affordances": [
                {"role": a.role, "name": a.name, "locator": a.locator}
                for a in page_state.affordances[:20]
            ],
            "recent_steps": history[-5:],
            "rag_snippets": rag[:5],
            "original_action": action.model_dump(),
            "missing_fields": missing,
        }
        if plan_step:
            payload_ctx["plan_step"] = plan_step
        if failure_count:
            payload_ctx["failures"] = failure_count
        if failure_reason:
            payload_ctx["failure_reason"] = failure_reason
        if user_inputs:
            payload_ctx["user_inputs"] = user_inputs
        if context:
            payload_ctx["token_budget_remaining"] = context.token_budget_remaining
            payload_ctx["steps_remaining"] = context.steps_remaining
        if workflow_state:
            payload_ctx["workflow_state"] = workflow_state.as_dict()

        serialized = json.dumps(payload_ctx, ensure_ascii=False)
        self._consume_tokens(context, serialized)

        response = self._llm.chat(
            system=self._repair_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Complete the tool call so that all mandatory parameters are present.\n"
                        f"Context: {serialized}"
                    ),
                }
            ],
            temperature=0,
        )

        content = response["choices"][0]["message"].get("content", "")
        logger.debug("Strategy repair response", content=content)
        data = self._extract_json_object(content)
        repaired = Action(type=data["type"], params=data.get("params", {}))
        logger.info("Strategy: action repaired", action=repaired.type, params=repaired.params)
        return repaired

    def _repair_missing_action(
        self,
        task: Task,
        page_state: PageStatePack,
        history: List[str],
        rag: List[str],
        planner_message: Dict[str, object],
        *,
        plan_step: Optional[str] = None,
        failure_count: int = 0,
        failure_reason: Optional[str] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
        workflow_state: Optional[WorkflowState] = None,
    ) -> Action:
        payload_ctx: Dict[str, Any] = {
            "goal": task.goal,
            "current_url": page_state.url,
            "page_title": page_state.title,
            "affordances": [
                {"role": a.role, "name": a.name, "locator": a.locator}
                for a in page_state.affordances[:20]
            ],
            "forms": [
                [
                    {
                        "label": field.label,
                        "name": field.name,
                        "locator": field.locator,
                        "type": field.type,
                    }
                    for field in form[:6]
                ]
                for form in page_state.forms[:3]
            ],
            "recent_steps": history[-5:],
            "rag_snippets": rag[:5],
            "planner_output": planner_message,
        }
        if plan_step:
            payload_ctx["plan_step"] = plan_step
        if failure_count:
            payload_ctx["failures"] = failure_count
        if failure_reason:
            payload_ctx["failure_reason"] = failure_reason
        if user_inputs:
            payload_ctx["user_inputs"] = user_inputs
        if context:
            payload_ctx["token_budget_remaining"] = context.token_budget_remaining
            payload_ctx["steps_remaining"] = context.steps_remaining
        if workflow_state:
            payload_ctx["workflow_state"] = workflow_state.as_dict()

        serialized = json.dumps(payload_ctx, ensure_ascii=False)
        self._consume_tokens(context, serialized)
        response = self._llm.chat(
            system=self._missing_action_prompt,
            messages=[{"role": "user", "content": serialized}],
            temperature=0,
        )
        content = response["choices"][0]["message"].get("content", "")
        logger.debug("Strategy missing-action repair response", content=content)
        data = self._extract_json_object(content)
        action = Action(type=data["type"], params=data.get("params", {}))
        logger.info("Strategy: missing action repaired", action=action.type, params=action.params)
        return action

    # ------------------------------------------------------------------ heuristics / formatting

    def shortlist_affordances(
        self,
        page_state: PageStatePack,
        goal: str,
        limit: int = 6,
        workflow_state: Optional[WorkflowState] = None,
    ) -> List[Affordance]:
        state = workflow_state or WorkflowState()
        keyword = self._extract_primary_keyword(goal)
        shortlist = self._affordances_from_hints(
            hints=page_state.navigation_hints,
            page_state=page_state,
            limit=limit,
        )

        domain = self._domain_from_url(page_state.url)
        heuristic_affordances = self._affordances_from_heuristics(domain, page_state)
        for candidate in heuristic_affordances:
            if len(shortlist) >= limit:
                break
            if candidate not in shortlist:
                shortlist.append(candidate)

        if len(shortlist) < limit:
            scored: List[tuple[int, Affordance]] = []
            for affordance in page_state.affordances:
                score = 0
                label = " ".join(
                    part for part in [affordance.role or "", affordance.name or ""] if part
                ).lower()
                if keyword and keyword in label:
                    score += 3
                if not state.profile_reviewed and any(token in label for token in ["profile", "account", "resume", "резюме"]):
                    score += 3
                if affordance.role in {"link", "button"}:
                    score += 2
                if affordance.role in {"textbox", "combobox"}:
                    score += 1
                    if not state.search_started:
                        score += 2
                scored.append((score, affordance))

            ranked = sorted(scored, key=lambda item: item[0], reverse=True)
            for _, affordance in ranked:
                if len(shortlist) >= limit:
                    break
                if affordance not in shortlist:
                    shortlist.append(affordance)

        return shortlist[:limit]

    def _extract_primary_keyword(self, goal: str) -> Optional[str]:
        tokens = [token.strip().lower() for token in goal.split() if len(token) > 4]
        return tokens[0] if tokens else None

    def _affordances_from_hints(
        self,
        *,
        hints: List,
        page_state: PageStatePack,
        limit: int,
    ) -> List[Affordance]:
        affordance_lookup = {id(aff): aff for aff in page_state.affordances}
        shortlisted: List[Affordance] = []
        for hint in hints:
            if len(shortlisted) >= limit:
                break
            if hint.source == "affordance":
                affordance = affordance_lookup.get(hint.extra.get("affordance_id"))
                if affordance and affordance not in shortlisted:
                    shortlisted.append(affordance)
                    continue
            if hint.locator:
                synthetic = Affordance(
                    role=hint.extra.get("role"),
                    name=hint.label,
                    locator=hint.locator,
                    extra={"source": hint.source},
                )
                shortlisted.append(synthetic)
        return shortlisted

    def _affordances_from_heuristics(
        self,
        domain: Optional[str],
        page_state: PageStatePack,
    ) -> List[Affordance]:
        categories = ["apply_button", "search_input"]
        results: List[Affordance] = []
        seen: set[Tuple[Optional[str], str]] = set()
        for category in categories:
            try:
                patterns = list(self._heuristics.find(category, domain=domain))
            except Exception:  # noqa: BLE001
                logger.debug("Strategy: heuristic affordance lookup failed", category=category, domain=domain)
                continue
            if not patterns:
                continue
            ordered = sorted(patterns, key=lambda pattern: 0 if pattern.domain else 1)
            for pattern in ordered:
                if not pattern.locator:
                    continue
                for role, locator, text in self._iter_locator_candidates(page_state, pattern.locator):
                    resolved_role = role
                    if not resolved_role:
                        resolved_role = "textbox" if category == "search_input" else "button"
                    name = text or pattern.text or category.replace("_", " ").title()
                    signature = (resolved_role, locator)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    results.append(
                        Affordance(
                            role=resolved_role,
                            name=name[:120],
                            locator=locator,
                            extra={
                                "source": f"heuristic:{category}",
                                "domain": pattern.domain,
                            },
                        )
                    )
        return results

    def _format_forms(self, forms: List[List[PageFormField]]) -> List[str]:
        lines: List[str] = []
        for idx, form in enumerate(forms[:5], start=1):
            lines.append(f"- Form {idx}:")
            for field in form[:8]:
                descriptor = f"  • {field.label or field.name or 'field'}"
                if field.type:
                    descriptor += f" ({field.type})"
                if field.locator:
                    descriptor += f" locator={field.locator[:80]}"
                lines.append(descriptor)
        return lines

    def _format_snippets(self, snippets: Iterable[str]) -> List[str]:
        lines: List[str] = []
        for snippet in list(snippets)[:5]:
            sanitized = re.sub(r"\s+", " ", snippet).strip()
            if len(sanitized) > 200:
                sanitized = sanitized[:197] + "..."
            lines.append(f"- {sanitized}")
        return lines

    def _format_dom_diff(self, diff) -> List[str]:
        if not diff:
            return []
        lines: List[str] = []
        for item in diff.added[:3]:
            lines.append(f"+ {item.summary}")
        for item in diff.removed[:3]:
            lines.append(f"- {item.summary}")
        for item in diff.changed[:3]:
            fields = ", ".join(item.fields_changed) if item.fields_changed else "changed"
            lines.append(f"* {item.summary} ({fields})")
        return lines

    def _browser_action_tool_schema(self) -> Dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": "dispatch_action",
                "description": "Dispatch a browser action to progress towards the task goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "navigate",
                                "query",
                                "click",
                                "type",
                                "select",
                                "wait",
                                "scroll",
                                "extract",
                                "confirm_gate",
                            ],
                        },
                        "params": {
                            "type": "object",
                            "default": {},
                            "description": "Arguments for the action. Keep optional values compact.",
                        },
                    },
                    "required": ["type"],
                },
            },
        }

    def _consume_tokens(self, context: Optional[TaskContext], prompt: str) -> None:
        if not context:
            return
        try:
            tokens = self._llm.count_tokens(prompt)
        except Exception:  # noqa: BLE001
            logger.debug("Strategy: token estimation failed")
            return
        context.consume_tokens(tokens + 16)

    def _apply_heuristics(
        self,
        task: Task,
        page_state: PageStatePack,
        action: Action,
        *,
        plan_step: Optional[str] = None,
        failure_count: int = 0,
        failure_reason: Optional[str] = None,
        user_inputs: Optional[Dict[str, Any]] = None,
        workflow_state: Optional[WorkflowState] = None,
    ) -> Dict[str, object]:
        params = dict(action.params)
        focus = (plan_step or "").lower()

        def focus_contains(*keywords: str) -> bool:
            return any(keyword in focus for keyword in keywords if keyword)

        domain = self._domain_from_url(page_state.url)

        if action.type == "navigate" and not params.get("url"):
            params["url"] = self._infer_navigation_target(task, page_state)
        elif action.type == "query" and not (params.get("selector") or params.get("locator")):
            locator = self._locator_from_patterns("search_input", page_state, domain)
            if not locator:
                locator = self._navigator.best_locator(
                    page_state,
                    keywords=self._extract_search_keywords(task.goal) or ["search", "results"],
                    roles={"textbox", "combobox"},
                )
            params["locator"] = locator or "body"
            if locator:
                self._record_heuristic("search_input", domain, locator=locator)
        elif action.type == "type" and not params.get("locator"):
            locator = self._locator_from_patterns("search_input", page_state, domain)
            if not locator:
                locator = self._form_field_locator(
                    page_state,
                    plan_step,
                    fallback_goal=task.goal,
                    domain=domain,
                )
            if not locator:
                locator = self._navigator.best_locator(
                    page_state,
                    keywords=self._extract_search_keywords(plan_step or task.goal) or ["input", "form", "search"],
                    roles={"textbox", "combobox"},
                )
            if not locator:
                locator = self._affordance_textbox_locator(page_state, domain)
            if locator:
                params["locator"] = locator
                self._record_heuristic("search_input", domain, locator=locator)
            params.setdefault("clear", True)
        elif action.type == "extract" and not params.get("locator"):
            locator = self._navigator.best_locator(
                page_state,
                keywords=["result", "listing", "card"],
                roles={"link", "article", "listitem"},
            )
            params["locator"] = locator or "body"
        elif action.type == "click" and not (params.get("locator") or params.get("text")):
            preferred_keywords: List[str] = [
                "поиск",
                "search",
                "найти",
                "submit",
                "login",
                "sign",
                "примен",
                "отклик",
            ]
            if focus_contains("отклик", "apply", "respond"):
                preferred_keywords = ["отклик", "откликнуться", "apply", "respond"]
            candidate = self._navigator.best_locator(
                page_state,
                keywords=preferred_keywords,
                roles={"button", "link"},
            )
            params["locator"] = candidate or "button"
            if candidate and focus_contains("отклик", "apply", "respond"):
                self._record_heuristic("apply_button", domain, locator=candidate)
        elif action.type == "confirm_gate" and not params.get("intent"):
            intent = self._derive_confirm_intent(
                workflow_state=workflow_state,
                failure_reason=failure_reason,
                page_state=page_state,
                plan_step=plan_step,
            )
            params["intent"] = intent
        return params

    def _derive_confirm_intent(
        self,
        *,
        workflow_state: Optional[WorkflowState],
        failure_reason: Optional[str],
        page_state: PageStatePack,
        plan_step: Optional[str],
    ) -> str:
        if workflow_state and workflow_state.auth_prompt_visible and not workflow_state.authenticated:
            return "Authenticate user to continue automation"
        if failure_reason:
            normalized = failure_reason.replace("_", " ").strip()
            if normalized:
                return f"Resolve blocker: {normalized}"
        if plan_step:
            condensed = plan_step.strip()
            if condensed:
                return f"Confirm to proceed: {condensed}"
        title = (page_state.title or "").strip()
        if title:
            return f"Confirm action on {title}"
        return "User confirmation required to continue"

    def _locator_from_patterns(
        self,
        category: str,
        page_state: PageStatePack,
        domain: Optional[str],
    ) -> Optional[str]:
        try:
            patterns = list(self._heuristics.find(category, domain=domain))
        except Exception:  # noqa: BLE001
            logger.debug("Strategy: heuristic lookup failed", category=category, domain=domain)
            return None
        if not patterns:
            return None
        prioritized = sorted(patterns, key=lambda pattern: 0 if pattern.domain else 1)
        for pattern in prioritized:
            locator = pattern.locator
            if not locator:
                continue
            for role, candidate_locator, text in self._iter_locator_candidates(page_state, locator):
                if pattern.matches(role=role, locator=candidate_locator, text=text):
                    return candidate_locator
        return None

    def _iter_locator_candidates(
        self,
        page_state: PageStatePack,
        locator: str,
    ) -> Iterable[Tuple[Optional[str], str, Optional[str]]]:
        for affordance in page_state.affordances:
            if affordance.locator == locator:
                yield affordance.role, affordance.locator, affordance.name
        for node in page_state.distilled_elements:
            if node.locator == locator:
                text = node.label or node.text or node.name
                yield node.role, node.locator, text
        for form in page_state.forms:
            for field in form:
                if field.locator == locator:
                    field_type = (field.type or "").lower()
                    role: Optional[str] = None
                    if field_type in {"text", "search", "email", "url", "tel"}:
                        role = "textbox"
                    elif field_type in {"select-one", "select"}:
                        role = "combobox"
                    yield role, field.locator, field.label or field.name

    def _text_present(self, page_state: PageStatePack, needle: str) -> bool:
        if not needle:
            return False
        target = needle.lower()
        for snippet in page_state.text_snippets:
            if target in snippet.lower():
                return True
        for affordance in page_state.affordances:
            if affordance.name and target in affordance.name.lower():
                return True
        for node in page_state.distilled_elements:
            for candidate in (node.label, node.text, node.name):
                if candidate and target in candidate.lower():
                    return True
        return False

    def _patterns_match(
        self,
        category: str,
        page_state: PageStatePack,
        domain: Optional[str],
    ) -> bool:
        try:
            patterns = list(self._heuristics.find(category, domain=domain))
        except Exception:  # noqa: BLE001
            logger.debug("Strategy: heuristic match failed", category=category, domain=domain)
            return False
        if not patterns:
            return False
        prioritized = sorted(patterns, key=lambda pattern: 0 if pattern.domain else 1)
        for pattern in prioritized:
            if pattern.locator:
                for role, locator, text in self._iter_locator_candidates(page_state, pattern.locator):
                    if pattern.matches(role=role, locator=locator, text=text):
                        return True
            if pattern.text and self._text_present(page_state, pattern.text):
                return True
        return False

    def _record_heuristic(
        self,
        category: str,
        domain: Optional[str],
        *,
        locator: Optional[str] = None,
        role: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        if not locator and not text:
            return
        try:
            self._heuristics.record(
                category,
                domain=domain,
                locator=locator,
                role=role,
                text=text,
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Strategy: failed to record heuristic", category=category, domain=domain)

    def _domain_from_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            host = urlparse(url).hostname
        except Exception:  # noqa: BLE001
            return None
        if not host:
            return None
        host = host.lower()
        return host[4:] if host.startswith("www.") else host

    def _infer_navigation_target(self, task: Task, page_state: PageStatePack) -> str:
        url = page_state.url
        if "google." in url:
            query = quote_plus(task.goal[:60])
            return f"https://www.google.com/search?q={query}"
        return "https://www.google.com"

    def _extract_search_keywords(self, text: str) -> List[str]:
        words = re.findall(r"[a-zA-ZА-Яа-я0-9+-]+", text.lower())
        filtered = [w for w in words if len(w) > 2]
        return filtered[:6]

    def _form_field_locator(
        self,
        page_state: PageStatePack,
        plan_step: Optional[str],
        *,
        fallback_goal: str,
        domain: Optional[str],
    ) -> Optional[str]:
        keywords = self._extract_search_keywords(plan_step or fallback_goal)
        keyword_set = {kw for kw in keywords if len(kw) > 2}
        for form in page_state.forms:
            for field in form:
                label = " ".join(
                    part for part in [field.label, field.name, field.type] if part
                ).lower()
                if not label and not keyword_set:
                    continue
                if keyword_set and not any(token in label for token in keyword_set):
                    continue
                if field.locator:
                    if domain:
                        self._record_heuristic(
                            "search_input",
                            domain,
                            locator=field.locator,
                            role="textbox",
                            text=field.label or field.name,
                        )
                    return field.locator
        return None

    def _affordance_textbox_locator(
        self,
        page_state: PageStatePack,
        domain: Optional[str],
    ) -> Optional[str]:
        for affordance in page_state.affordances:
            if affordance.role in {"textbox", "combobox"} and affordance.locator:
                if domain:
                    self._record_heuristic(
                        "search_input",
                        domain,
                        locator=affordance.locator,
                        role=affordance.role,
                        text=affordance.name,
                    )
                return affordance.locator
        return None

    def _extract_json_object(self, text: str) -> Dict[str, object]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Repair response did not contain JSON object.")
        return json.loads(match.group(0))

    @property
    def navigator(self) -> Navigator:
        return self._navigator

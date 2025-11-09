from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

import typer

from ..browser.controller import BrowserActionResult, BrowserController
from ..browser.observation import ObservationCollector
from ..browser.sandbox import SandboxPolicy
from ..config import Settings
from ..errors import BrowserError, LLMError, ParsingError, SandboxViolation
from ..logging import set_run_context
from ..types import Action, AgentStep, InteractiveElement, Observation, PlanItem, RunResult
from .evaluator import EvaluationResult, StepEvaluator
from .memory import AgentMemory
from .planner import Planner
from .trace import TraceRecorder

logger = logging.getLogger(__name__)


class Agent:
    """Main orchestration loop coordinating planner, browser, and evaluator."""

    PLAN_STATUS_PATTERN = re.compile(r"plan_status\s*=\s*(done|ongoing|in_progress)", re.IGNORECASE)
    PLAN_TASK_PATTERN = re.compile(r"\[\s*task\s*(\d+)(?:\s*/\s*(\d+))?:", re.IGNORECASE)

    def __init__(
        self,
        planner: Planner,
        settings: Settings,
        sandbox: SandboxPolicy,
        runs_dir: Path,
        capture_screenshots: bool = True,
    ) -> None:
        self._planner = planner
        self._settings = settings
        self._sandbox = sandbox
        self._runs_dir = runs_dir
        self._capture_screenshots = capture_screenshots
        self._memory = AgentMemory()
        self._evaluator = StepEvaluator()
        self._plan: list[PlanItem] = []
        self._current_plan_index: int = 0

    async def run(self, task: str, start_url: str | None = None, headless: bool | None = None) -> RunResult:
        run_id = str(uuid.uuid4())
        run_dir = TraceRecorder.new_run_dir(self._runs_dir, prefix=f"run_{run_id}")
        set_run_context(run_id=run_id)
        headless = self._settings.headless_default if headless is None else headless

        collector = ObservationCollector(run_dir, capture_screenshots=self._capture_screenshots)
        trace = TraceRecorder(run_id=run_id, root_dir=run_dir)
        steps: list[AgentStep] = []
        previous_excerpt: str | None = None
        last_action_summary: str | None = None
        last_action: Action | None = None
        same_action_streak: int = 0
        enforce_repeat_guard = not self._settings.allow_repeat_actions
        self._plan = []
        self._current_plan_index = 0

        async with BrowserController(
            self._sandbox,
            run_dir,
            headless=headless,
            user_data_dir=self._settings.browser_profile_dir,
        ) as controller:
            if start_url:
                try:
                    await controller.open_url(
                        Action(
                            type="open_url",
                            url=start_url,
                            reason="Open initial start URL before planning",
                        )
                    )
                except (BrowserError, SandboxViolation) as exc:
                    logger.exception("Failed to open start url")
                    return RunResult(steps=[], status="failed", error=str(exc))

            observation = await collector.collect(controller.page, step_index=0)
            await self._initialise_plan(task, observation, trace)

            for step_index in range(1, self._sandbox.max_steps + 1):
                trace.record_observation(step_index, observation)

                sandbox_summary = (
                    f"Allowed hosts: {', '.join(sorted(self._sandbox.allowed_hosts))}; "
                    f"Steps remaining: {self._sandbox.max_steps - self._sandbox.steps_taken}"
                )
                memory_context = self._memory.compressed()

                assessment = self._evaluator.assess(observation)
                plan_descriptor = self._active_plan_descriptor()
                logger.info(
                    "Step %d context | active_task=%s | plan=%s | modules=evaluator>planner>browser",
                    step_index,
                    plan_descriptor or "none",
                    self._plan_status_overview() or "[no plan]",
                )
                page_context_hint = self._derive_page_context(observation)
                recent_history = self._format_recent_history(steps)
                hint_lines = [
                    "Evaluator assessment:",
                    f"- login_required: {'yes' if assessment.login_required else 'no'}",
                    f"- captcha_detected: {'yes' if assessment.captcha_detected else 'no'}",
                ]
                hint_lines.append(f"- page_context: {page_context_hint}")
                if observation.overlays:
                    overlay_notes = []
                    for overlay in observation.overlays[:3]:
                        label = overlay.title or (overlay.text[:80] if overlay.text else "")
                        if not label:
                            label = overlay.selector or "overlay"
                        overlay_notes.append(label.strip())
                    overlay_summary = "; ".join(note for note in overlay_notes if note)
                    if overlay_summary:
                        hint_lines.append(
                            f"- overlay_alert: Modal or popup blocking interactions -> {overlay_summary}"
                        )
                    else:
                        hint_lines.append("- overlay_alert: Modal or popup is blocking interactions; inspect close/confirm buttons before retrying the previous selector.")
                if recent_history:
                    hint_lines.append(f"- recent_history: {recent_history}")
                hint_lines.append(f"- task_focus: {task}")
                hint_lines.append(
                    "- planning_protocol: Break the goal into clear sub-tasks, execute them one by one, "
                    "and mention the current sub-task in the reason for each action."
                )
                plan_hint = self._format_plan_hint()
                if plan_hint:
                    hint_lines.append(f"- execution_plan: {plan_hint}")
                    hint_lines.append(
                        "- plan_status_instructions: Prefix reasons with the active task as `[Task i/n: name]` and append `plan_status=ongoing` while working it; switch to `plan_status=done` once completed and move to the next task."
                    )
                if last_action_summary:
                    last_selector = last_action.selector if last_action else "-"
                    last_type = last_action.type if last_action else "-"
                    hint_lines.append(
                        f"- last_action: {last_type} selector={last_selector} -> {last_action_summary}"
                    )
                    if last_action and last_action.selector:
                        element_state = self._find_element_by_selector(observation, last_action.selector)
                        if element_state is not None:
                            element_text = (element_state.text or "").strip()
                            if last_action.type == "type" and (last_action.text or "").strip():
                                typed_text = (last_action.text or "").strip()
                                if element_text:
                                    if element_text.lower() == typed_text.lower():
                                        hint_lines.append(
                                            f"- field_state: {last_action.selector} already contains {element_text!r}; submit or use nearby search controls instead of retyping."
                                        )
                                    else:
                                        hint_lines.append(
                                            f"- field_state: {last_action.selector} now shows {element_text!r} (previous input {typed_text!r}); adjust next step accordingly."
                                        )
                                else:
                                    hint_lines.append(
                                        f"- field_state: {last_action.selector} appears empty; confirm the selector or check if the site replaced the field after typing."
                                    )
                            elif last_action.type in {"click", "press"}:
                                descriptor = element_text or (element_state.role or element_state.tag or "control")
                                hint_lines.append(
                                    f"- field_state: {last_action.selector} is still present ({descriptor}); consider a different follow-up if the outcome did not change."
                                )
                    if last_action and last_action.type == "type":
                        submit_hint = self._summarise_submit_controls(observation)
                        if submit_hint:
                            hint_lines.append(f"- submit_controls: {submit_hint}")
                    summary_lower = last_action_summary.lower()
                    if any(keyword in summary_lower for keyword in ("already dismissed", "no visible matches")):
                        hint_lines.append(
                            "- IMPORTANT: Previous click removed the dialog; pick a different action instead of repeating the same selector."
                        )
                if enforce_repeat_guard and same_action_streak >= 2 and last_action is not None:
                    if last_action.type == "type":
                        repetition_phrase = "typed into"
                    elif last_action.type == "press":
                        repetition_phrase = "pressed"
                    else:
                        repetition_phrase = "clicked"
                    hint_lines.append(
                        f"- WARNING: That selector has been {repetition_phrase} {same_action_streak} times consecutively; analyze the page and choose a new strategy before repeating it."
                    )
                blocker_hint = "\n".join(hint_lines)

                forced_action = assessment.forced_action
                if forced_action is not None:
                    action = forced_action
                    logger.info(
                        "Step %d action forced by evaluator: type=%s reason=%s",
                        step_index,
                        action.type,
                        action.reason,
                    )
                else:
                    planner_hint = blocker_hint
                    replanned = False
                    repeat_replans = 0
                    while True:
                        try:
                            action = await self._planner.decide_action(
                                task=task,
                                observation=observation,
                                memory_context=memory_context,
                                sandbox_summary=sandbox_summary,
                                blocker_hint=planner_hint,
                            )
                        except (LLMError, ParsingError) as exc:
                            logger.exception("Planner failure")
                            return RunResult(steps=steps, status="failed", error=str(exc))

                        if (
                            action.type == "await_user_login"
                            and not assessment.login_required
                            and not assessment.captcha_detected
                        ):
                            if replanned:
                                logger.info(
                                    "Planner repeatedly requested login despite evaluator hints; "
                                    "substituting wait action for stabilization."
                                )
                                action = Action(
                                    type="wait",
                                    timeout_ms=1000,
                                    reason=(
                                        "Waiting briefly after successful login to allow the page "
                                        "to stabilize before replanning."
                                    ),
                                )
                                break

                            logger.info(
                                "Planner requested login but evaluator indicates none required; requesting replan."
                            )
                            planner_hint += (
                                "\nREPLAN: Evaluator confirms the user is already logged in; pick a different action instead of await_user_login."
                            )
                            replanned = True
                            continue

                        repeated_same_selector = (
                            last_action is not None
                            and action.type == last_action.type
                            and action.selector == last_action.selector
                        )
                        summary_lower = (last_action_summary or "").lower()
                        dismissed_repeat = repeated_same_selector and last_action_summary and any(
                            keyword in summary_lower
                            for keyword in ("already dismissed", "no visible matches", "blocked by another element")
                        )
                        stagnant_repeat = (
                            repeated_same_selector
                            and same_action_streak >= 2
                            and last_action_summary is not None
                            and any(
                                keyword in summary_lower
                                for keyword in ("clicked", "typed", "pressed", "selected", "scrolled", "blocked")
                            )
                        )

                        if enforce_repeat_guard and (dismissed_repeat or stagnant_repeat):
                            if repeat_replans < 2:
                                if dismissed_repeat:
                                    reason_hint = "That selector already disappeared"
                                elif action.type == "type":
                                    reason_hint = (
                                        f"Text already entered into {action.selector}; proceed to submit or change fields instead of typing again"
                                    )
                                elif action.type == "press":
                                    reason_hint = (
                                        f"Keys already pressed on {action.selector}; try a different control or navigation"
                                    )
                                else:
                                    reason_hint = (
                                        f"That selector was already clicked {same_action_streak} times without visible progress"
                                    )
                                planner_hint += (
                                    f"\nREPLAN: {reason_hint}; choose a different approach (scroll, inspect page text, search, or interact with new content)."
                                )
                                repeat_replans += 1
                                logger.info(
                                    "Planner proposed repeating %s on %s despite prior outcome; requesting replan",
                                    action.type,
                                    action.selector,
                                )
                                continue

                            logger.info(
                                "Substituting alternate action to break repeat loop for selector %s",
                                action.selector,
                            )
                            if action.type == "type":
                                action = Action(
                                    type="press",
                                    selector=action.selector,
                                    keys="Enter",
                                    reason=(
                                        "Submit the current input instead of retyping the same text; pressing Enter should trigger the search."
                                    ),
                                )
                            elif action.type == "click":
                                action = Action(
                                    type="scroll",
                                    scroll="down",
                                    reason=(
                                        "Scroll the page to reveal unobstructed controls since repeated clicks were blocked."
                                    ),
                                )
                            else:
                                action = Action(
                                    type="scroll",
                                    scroll="down",
                                    reason=(
                                        "Change page context after repeating the same action without progress."
                                    ),
                                )
                            repeat_replans = 0
                            same_action_streak = 0

                        break

                if action.type == "press" and action.keys is None:
                    logger.info(
                        "Press action on %s missing keys; defaulting to Enter",
                        action.selector or "-",
                    )
                    action = Action(
                        type="press",
                        selector=action.selector,
                        keys="Enter",
                        reason=action.reason,
                    )

                trace.record_action(step_index, action)
                logger.info(
                    "Step %d action decided [%s]: type=%s selector=%s reason=%s",
                    step_index,
                    plan_descriptor or "no-plan",
                    action.type,
                    action.selector or "-",
                    action.reason,
                )

                is_manual_login = action.type == "await_user_login"
                if action.type not in {"finish", "abort", "await_user_login"}:
                    self._sandbox.validate_action()

                try:
                    if is_manual_login:
                        result = await self._await_user_login(action, controller)
                    else:
                        result = await controller.perform(action)
                except (BrowserError, SandboxViolation) as exc:
                    logger.exception("Browser execution failed")
                    return RunResult(steps=steps, status="failed", error=str(exc))

                self._sandbox.record_step()

                result_payload = {
                    "summary": result.summary,
                    "artifacts": result.artifacts,
                    "url": result.url,
                    "reason": action.reason,
                }
                trace.record_result(step_index, result_payload)

                next_observation = await collector.collect(controller.page, step_index=step_index)
                agent_step = trace.to_agent_step(
                    index=step_index,
                    observation=observation,
                    action=action,
                    summary=result.summary,
                    artifacts=result.artifacts,
                )
                steps.append(agent_step)
                self._memory.add(observation, action)
                self._update_plan_progress(action, result.summary, trace)
                if last_action is not None and action.type == last_action.type and action.selector == last_action.selector:
                    same_action_streak += 1
                else:
                    same_action_streak = 1
                last_action_summary = result.summary
                last_action = action

                evaluation = self._evaluator.evaluate(
                    action=action,
                    observation_before=observation,
                    observation_after=next_observation,
                    previous_excerpt=previous_excerpt,
                )

                if not evaluation.continue_run:
                    return RunResult(
                        steps=steps,
                        status=evaluation.status,
                        final_answer=evaluation.final_answer,
                        error=evaluation.error,
                    )

                if evaluation.replan:
                    await asyncio.sleep(0.5)

                observation = next_observation
                previous_excerpt = observation.visible_text[:200]

        return RunResult(
            steps=steps,
            status="failed",
            error="Maximum steps reached without completion",
        )

    async def _await_user_login(self, action: Action, controller: BrowserController) -> BrowserActionResult:
        message = action.message or "Please complete the login or registration manually."
        typer.echo(message)
        typer.echo("Press Enter once you are ready to continue.")

        loop = asyncio.get_running_loop()

        def _wait_for_confirmation() -> None:
            input("Press Enter after finishing the sign-in process...")

        await loop.run_in_executor(None, _wait_for_confirmation)
        await controller.ensure_ready()
        return BrowserActionResult(
            summary="User login wait completed",
            artifacts=[],
            url=controller.page.url,
        )

    def _derive_page_context(self, observation: Observation) -> str:
        title = (observation.title or "-").strip()
        lines = [line.strip() for line in observation.visible_text.splitlines() if line.strip()]
        snippet = " / ".join(lines[:5]) if lines else "[no visible text]"
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        interactive_descriptors: list[str] = []
        for element in observation.interactive_elements[:5]:
            text = (element.text or element.attributes.get("aria-label") or "").strip()
            if text:
                interactive_descriptors.append(text)
        interactive_hint = ", ".join(interactive_descriptors) if interactive_descriptors else ""
        if interactive_hint:
            return f"title='{title}', key_text={snippet}, notable_elements={interactive_hint}"
        return f"title='{title}', key_text={snippet}"

    def _format_recent_history(self, history: list[AgentStep]) -> str:
        if not history:
            return ""
        recent = history[-3:]
        parts: list[str] = []
        for step in recent:
            summary = step.result_summary.replace("\n", " ")
            if len(summary) > 80:
                summary = summary[:77] + "..."
            parts.append(f"{step.action.type} -> {summary}")
        return "; ".join(parts)

    async def _initialise_plan(self, task: str, observation: Observation, trace: TraceRecorder) -> None:
        try:
            plan = await self._planner.create_plan(task, observation)
        except (LLMError, ParsingError) as exc:
            logger.warning("Failed to create initial plan: %s", exc)
            self._plan = []
            self._current_plan_index = 0
            return

        self._plan = plan
        self._current_plan_index = 0
        if self._plan:
            self._plan[0].mark_in_progress()
            plan_outline = "; ".join(f"{item.index}. {item.title}" for item in self._plan)
            logger.info("Execution plan initialised (%d tasks): %s", len(self._plan), plan_outline)
            logger.info(
                "Active task -> %s",
                self._active_plan_descriptor() or "none",
            )
        try:
            trace.record_plan(self._plan)
        except Exception:  # pragma: no cover - file system issues
            logger.exception("Unable to persist plan trace")

    def _format_plan_hint(self) -> str:
        if not self._plan:
            return ""

        parts: list[str] = []
        for item in self._plan:
            status = item.status
            prefix = f"{item.index}."
            if item.index - 1 == self._current_plan_index and status != "done":
                prefix += "*"
            descriptor = f"{prefix} {item.title} [{status}]"
            parts.append(descriptor)
        return " | ".join(parts)

    def _update_plan_progress(self, action: Action, result_summary: str, trace: TraceRecorder) -> None:
        if not self._plan:
            return

        reason = action.reason or ""
        status_match = self.PLAN_STATUS_PATTERN.search(reason)
        status = status_match.group(1).lower() if status_match else None
        if status is None and action.plan_status:
            status = action.plan_status.lower()
        target_index = self._determine_plan_index(reason)
        if target_index is None:
            target_index = self._current_plan_index

        updated = False

        if status in {"ongoing", "in_progress"}:
            if 0 <= target_index < len(self._plan):
                previous_status = self._plan[target_index].status
                self._plan[target_index].mark_in_progress()
                if self._current_plan_index != target_index:
                    self._current_plan_index = target_index
                    updated = True
                if self._plan[target_index].status != previous_status:
                    updated = True
                    self._log_plan_transition(self._plan[target_index])
        elif status == "done":
            if 0 <= target_index < len(self._plan):
                self._plan[target_index].mark_done(result_summary)
                self._log_plan_transition(self._plan[target_index])
                if target_index == self._current_plan_index:
                    self._advance_plan(trace)
                    return
                updated = True
        else:
            if 0 <= self._current_plan_index < len(self._plan):
                previous_status = self._plan[self._current_plan_index].status
                self._plan[self._current_plan_index].mark_in_progress()
                if self._plan[self._current_plan_index].status != previous_status:
                    updated = True
                    self._log_plan_transition(self._plan[self._current_plan_index])

        if updated:
            try:
                trace.record_plan(self._plan)
            except Exception:  # pragma: no cover - file system issues
                logger.exception("Unable to persist plan update")

    def _advance_plan(self, trace: TraceRecorder) -> None:
        updated = False
        while self._current_plan_index < len(self._plan) and self._plan[self._current_plan_index].status == "done":
            self._current_plan_index += 1
            updated = True

        if self._current_plan_index < len(self._plan):
            if self._plan[self._current_plan_index].status == "pending":
                self._plan[self._current_plan_index].mark_in_progress()
                updated = True
                self._log_plan_transition(self._plan[self._current_plan_index])

        if updated:
            try:
                trace.record_plan(self._plan)
            except Exception:  # pragma: no cover
                logger.exception("Unable to persist plan progression")

    def _determine_plan_index(self, reason: str) -> int | None:
        match = self.PLAN_TASK_PATTERN.search(reason)
        if not match:
            return None
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(self._plan):
            return None
        return index

    def _log_plan_transition(self, item: PlanItem) -> None:
        descriptor = f"Task {item.index}/{len(self._plan)} '{item.title}'"
        logger.info("Plan update: %s -> %s", descriptor, item.status)

    def _plan_status_overview(self) -> str:
        if not self._plan:
            return ""
        return ", ".join(f"{item.index}:{item.status}" for item in self._plan)

    def _active_plan_descriptor(self) -> str | None:
        if not self._plan:
            return None
        if self._current_plan_index < 0 or self._current_plan_index >= len(self._plan):
            return None
        item = self._plan[self._current_plan_index]
        return f"Task {item.index}/{len(self._plan)} '{item.title}' [{item.status}]"

    def _find_element_by_selector(self, observation: Observation, selector: str) -> InteractiveElement | None:
        selector = selector.strip()
        for element in observation.interactive_elements:
            if element.selector == selector:
                return element
        return None

    def _summarise_submit_controls(self, observation: Observation) -> str:
        candidates: list[str] = []
        for element in observation.interactive_elements:
            tag = (element.tag or "").lower()
            text = (element.text or "").strip()
            attrs = {k: (v or "").lower() for k, v in element.attributes.items()}
            role = (element.role or "").lower()
            selector = element.selector or tag or "control"

            is_submit = attrs.get("type") in {"submit", "search"}
            text_matches = any(keyword in text.lower() for keyword in ("найти", "искать", "search", "поиск"))
            role_submit = "submit" in role

            if tag == "button" and (is_submit or text_matches or role_submit):
                label = text or attrs.get("aria_label") or selector
                candidates.append(f"{selector} ({label.strip() or 'button'})")
            elif tag == "input" and is_submit:
                label = text or attrs.get("aria_label") or selector
                candidates.append(f"{selector} ({label.strip() or 'submit input'})")

            if len(candidates) >= 3:
                break

        return "; ".join(candidates)

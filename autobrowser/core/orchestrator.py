from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Deque, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

from loguru import logger

from autobrowser.agents.executor import ExecutorAgent
from autobrowser.agents.navigator import Navigator
from autobrowser.agents.extractor import ExtractorAgent
from autobrowser.agents.strategy import StrategyAgent
from autobrowser.browser.manager import BrowserManager
from autobrowser.browser.reader import PageReader
from autobrowser.browser.tools import BrowserToolExecutor
if TYPE_CHECKING:  # pragma: no cover
    from autobrowser.core.config import Settings
else:
    Settings = Any  # type: ignore[misc]

try:
    from autobrowser.core.config import get_settings
except ModuleNotFoundError:  # pragma: no cover
    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")
from autobrowser.core.logger import log_event
from autobrowser.core.memory import LongTermMemory, ShortTermMemory
from autobrowser.core.safety import SafetyEngine
from autobrowser.core.schemas import (
    Action,
    ConfirmPayload,
    Observation,
    PageStatePack,
    StepLog,
    Task,
    TaskContext,
    TaskContract,
    TaskLogEvent,
    TaskStatus,
)
from autobrowser.llm.provider import OpenAIProvider
from autobrowser.storage.files import FileStorage


class ConfirmationGate:
    """Tracks pending confirmation requests."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.decision: Optional[bool] = None
        self.payload: Optional[Dict[str, Any]] = None


class TaskOrchestrator:
    """Coordinates planner, LLM and browser to execute autonomous tasks."""

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        *,
        llm: Optional[OpenAIProvider] = None,
        browser_manager: Optional[BrowserManager] = None,
        browser_tools: Optional[BrowserToolExecutor] = None,
        reader: Optional[PageReader] = None,
        memory: Optional[LongTermMemory] = None,
        safety: Optional[SafetyEngine] = None,
        storage: Optional[FileStorage] = None,
        strategy: Optional[StrategyAgent] = None,
        extractor: Optional[ExtractorAgent] = None,
        executor: Optional[ExecutorAgent] = None,
        navigator: Optional[Navigator] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._tasks: Dict[str, Task] = {}
        self._tasks_lock = threading.Lock()
        self._event_queues: Dict[str, "queue.Queue[TaskLogEvent]"] = {}

        self._task_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._executor = ThreadPoolExecutor(max_workers=self._settings.max_concurrent_tasks)

        # Core components
        self._llm = llm or OpenAIProvider(self._settings)
        self._browser_manager = browser_manager or BrowserManager(self._settings)
        self._browser_tools = browser_tools or BrowserToolExecutor(self._settings.artifacts_dir)
        self._reader = reader or PageReader()
        self._memory = memory or LongTermMemory(self._settings)
        self._short_memory: Dict[str, ShortTermMemory] = {}
        self._task_plans: Dict[str, deque[str]] = {}
        self._safety = safety or SafetyEngine(self._settings)
        self._storage = storage or FileStorage(self._settings)
        self._artifacts_root = Path(self._settings.artifacts_dir).resolve()
        self._plan_failures: Dict[str, Dict[str, int]] = {}
        self._task_token_budgets: Dict[str, int] = {}
        threshold_value = getattr(self._settings, "plan_failure_rotate_threshold", 3)
        try:
            threshold_int = int(threshold_value)
        except (TypeError, ValueError):
            threshold_int = 3
        self._plan_failure_threshold: int = max(2, threshold_int or 3)

        # Agents
        existing_strategy = strategy
        candidate_navigator = getattr(existing_strategy, "navigator", None) if existing_strategy else None
        self._navigator = navigator or candidate_navigator or Navigator()
        self._strategy = existing_strategy or StrategyAgent(self._llm, navigator=self._navigator)
        self._executor_agent = executor or ExecutorAgent(self._strategy, navigator=self._navigator)
        self._extractor = extractor or ExtractorAgent()

        self._confirmations: Dict[str, ConfirmationGate] = {}
        self._task_deadlines: Dict[str, float] = {}

        if hasattr(self._browser_manager, "start"):
            self._browser_manager.start()
        self._dispatcher.start()

    # Task lifecycle -----------------------------------------------------

    def create_task(self, goal: str, *, allow_high_risk: bool = False) -> Task:
        task_id = uuid.uuid4().hex
        contract = TaskContract(
            success_condition=f"Achieve goal: {goal}",
            metrics={"task_success": False, "completed_steps": 0},
            limits={
                "max_steps": self._settings.max_steps,
                "max_seconds": self._settings.task_timeout_seconds,
                "token_budget": self._settings.token_budget,
            },
        )
        task = Task(
            id=task_id,
            goal=goal,
            created_at=datetime.now(UTC),
            status=TaskStatus.NEW,
            allow_high_risk=allow_high_risk,
            contract=contract,
        )
        logger.info("Task created", task_id=task_id, goal=goal, allow_high_risk=allow_high_risk)
        with self._tasks_lock:
            self._tasks[task_id] = task
            self._event_queues[task_id] = queue.Queue()
            self._short_memory[task_id] = ShortTermMemory(window=12)
            self._task_deadlines[task_id] = time.time() + self._settings.task_timeout_seconds
            self._task_token_budgets[task_id] = self._settings.token_budget

        self._publish_status(task, TaskStatus.NEW)
        self._publish_info(task, "info", {"type": "contract", "contract": contract.model_dump()})
        self._task_queue.put(task_id)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        with self._tasks_lock:
            return list(self._tasks.values())

    def confirm_task_action(
        self,
        task_id: str,
        approve: bool,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        gate = self._confirmations.get(task_id)
        if not gate:
            return False
        gate.decision = approve
        gate.payload = payload
        gate.event.set()
        return True

    def stream_events(self, task_id: str) -> Generator[TaskLogEvent, None, None]:
        queue_ref = self._event_queues.get(task_id)
        if queue_ref is None:
            return
        heartbeat = self._settings.sse_heartbeat_seconds
        while True:
            try:
                event = queue_ref.get(timeout=heartbeat)
                yield event
            except queue.Empty:
                yield self._make_event(task_id, "heartbeat", {})
            if self._stop_event.is_set():
                break

    # Internal loops -----------------------------------------------------

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task_id = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._executor.submit(self._run_task, task_id)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._browser_manager.shutdown()

    # Execution ----------------------------------------------------------

    def _run_task(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if not task:
            logger.error("Task missing during run", task_id=task_id)
            return
        self._update_status(task, TaskStatus.RUNNING)

        logger.info("Starting task orchestration", task_id=task_id)
        page = self._browser_manager.new_page(task_id)
        short_memory = self._short_memory[task_id]

        logger.info("Requesting initial plan", task_id=task_id)
        try:
            initial_context = self._make_context(
                task,
                step_index=0,
                plan=None,
            )
            plan_steps = self._strategy.build_initial_plan(task, context=initial_context)
            logger.info("Initial plan ready", task_id=task_id, steps=plan_steps)
            self._publish_info(task, "plan", {"steps": plan_steps})
            if plan_steps:
                self._task_plans[task_id] = deque(plan_steps)
                self._plan_failures[task_id] = {}
            self._task_token_budgets[task_id] = initial_context.token_budget_remaining
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planner failed during initial plan", task_id=task_id)
            self._publish_error(task, f"Planner failed: {exc}")
            self._browser_manager.close_task(task_id)
            self._memory.clear_task(task_id)
            return

        step_count = 0
        success = False
        last_screenshot: Optional[str] = None
        recent_action_signatures: deque[str] = deque(maxlen=5)

        persist_context = bool(getattr(self._settings, "persist_context_on_failure", True))

        while step_count < self._settings.max_steps:
            if time.time() > self._task_deadlines[task_id]:
                self._publish_error(task, "Task timed out.")
                break

            try:
                logger.info(
                    "Reading page state",
                    task_id=task_id,
                    step=step_count,
                    last_screenshot=last_screenshot,
                )
                page_state = self._reader.read(
                    page,
                    [task.goal],
                    last_screenshot,
                    task_id=task_id,
                    step_index=step_count,
                )
                logger.info(
                    "Page state captured",
                    task_id=task_id,
                    step=step_count,
                    url=page_state.url,
                    title=page_state.title,
                    affordances=len(page_state.affordances),
                    snippets=len(page_state.text_snippets),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Reader failed", task_id=task_id, step=step_count)
                self._publish_error(task, f"Reader failed: {exc}")
                break

            try:
                rag_entries = [entry.text for entry in self._memory.search(task_id, task.goal, top_k=5)]
            except Exception as exc:  # noqa: BLE001
                logger.exception("RAG search failed", task_id=task_id, step=step_count)
                self._publish_error(task, f"Memory search failed: {exc}")
                break
            logger.info(
                "RAG search complete",
                task_id=task_id,
                step=step_count,
                entries=len(rag_entries),
            )

            try:
                plan_focus = self._task_plans.get(task_id)
                focus_step = plan_focus[0] if plan_focus else None
                context = self._make_context(task, step_count, plan_focus)
                if context.steps_remaining <= 0:
                    self._publish_error(task, "Step budget exhausted.")
                    break
                if context.seconds_remaining <= 0:
                    self._publish_error(task, "Task deadline reached.")
                    break
                if context.token_budget_remaining <= 0:
                    self._publish_error(task, "Token budget exhausted.")
                    break
                logger.info(
                    "Executor deciding next action",
                    task_id=task_id,
                    step=step_count,
                    url=page_state.url,
                    plan_step=focus_step,
                )
                action, planner_note = self._executor_agent.decide_action(
                    task,
                    page_state,
                    short_memory.as_list(),
                    rag_entries,
                    plan_step=focus_step,
                    context=context,
                )
                self._task_token_budgets[task_id] = context.token_budget_remaining
                logger.info(
                    "Executor issued action",
                    task_id=task_id,
                    step=step_count,
                    action=action.type,
                    note=planner_note,
                )
                self._log_strategy_thought(task, short_memory, step_count, planner_note, focus_step)

                if action.type == "confirm_gate" and not action.params.get("intent"):
                    logger.warning(
                        "confirm_gate missing intent, proceeding anyway",
                        task_id=task_id,
                        step=step_count,
                    )
                    action.params["intent"] = "User confirmation required to continue"

                signature = f"{action.type}:{json.dumps(action.params, sort_keys=True)}"
                if recent_action_signatures.count(signature) >= 2:
                    logger.warning(
                        "Detected repetitive action",
                        task_id=task_id,
                        step=step_count,
                        action=action.type,
                    )
                    recovery = self._strategy.suggest_recovery(
                        task,
                        short_memory.as_list(),
                        None,
                    )
                    self._publish_info(
                        task,
                        "recovery",
                        {
                            "reason": "repetitive_action",
                            "action": action.type,
                            "suggestion": recovery,
                        },
                    )
                    skip_log = StepLog(
                        id=uuid.uuid4().hex,
                        task_id=task_id,
                        timestamp=datetime.now(UTC),
                        role="critic",
                        action=None,
                        observation=None,
                        note=f"Skipped repetitive action {action.type}.",
                    )
                    short_memory.add_step(skip_log)
                    self._emit_step(task, skip_log)
                    step_count += 1
                    continue
                recent_action_signatures.append(signature)
            except Exception as exc:  # noqa: BLE001
                self._publish_error(task, f"Executor failed: {exc}")
                logger.exception("Executor next action failed", task_id=task_id)
                break

            safety = self._safety.assess(task, action, description=planner_note)
            if safety.requires_confirmation:
                logger.warning(
                    "Action requires confirmation",
                    task_id=task_id,
                    step=step_count,
                    action=action.type,
                    reason=safety.reason,
                )
                approved, confirm_payload = self._handle_confirmation(
                    task,
                    action,
                    page_state,
                    planner_note,
                )
                if not approved:
                    self._publish_error(task, "High risk action was rejected.")
                    logger.info("High risk action rejected by user", task_id=task_id)
                    break
                if isinstance(confirm_payload, dict) and confirm_payload:
                    task.metadata.setdefault("user_inputs", {}).update(confirm_payload)

            logger.info(
                "Executing action",
                task_id=task_id,
                step=step_count,
                action=action.type,
            )
            observation = self._browser_tools.execute(
                page=page,
                task_id=task_id,
                step_number=step_count,
                action=action,
            )
            logger.info(
                "Action executed",
                task_id=task_id,
                step=step_count,
                ok=observation.ok,
                note=observation.note,
                error=observation.error_category,
            )
            last_screenshot = observation.screenshot_path

            action_note = self._describe_action(action, planner_note, focus_step)
            step_log = StepLog(
                id=uuid.uuid4().hex,
                task_id=task_id,
                timestamp=datetime.now(UTC),
                role="browser",
                action=action,
                observation=observation,
                note=action_note,
                metadata={
                    "step_index": step_count,
                    "planner_note": planner_note,
                    "plan_step": focus_step,
                },
            )
            short_memory.add_step(step_log)
            updated_state = self._reader.read(
                page,
                [task.goal],
                observation.screenshot_path,
                task_id=task_id,
                step_index=step_count,
            )
            self._memory.add_page_state(task_id, updated_state, observation=observation)
            self._emit_step(task, step_log)
            self._log_observation(task, short_memory, updated_state, observation, step_count)
            self._advance_plan(task_id, action, observation, planner_note)

            if not observation.ok:
                self._executor_agent.record_failure(task_id, focus_step, observation)
                recovery = self._strategy.suggest_recovery(
                    task,
                    short_memory.as_list(),
                    observation.note if observation else None,
                )
                self._publish_info(task, "recovery", {"suggestion": recovery})
                logger.warning(
                    "Observation indicates failure",
                    task_id=task_id,
                    step=step_count,
                    action_summary=action_note,
                    error_note=observation.note,
                    error_category=observation.error_category,
                )
                self._record_plan_failure(task, focus_step)
                step_count += 1
                continue

            self._executor_agent.record_success(task_id, focus_step)
            self._clear_plan_failure(task_id, focus_step)

            # Optionally perform extraction/reporting
            plan_remaining = self._task_plans.get(task_id)

            if action.type == "extract":
                logger.info("Extraction action completed", task_id=task_id)
                entities = self._extractor.extract_entities(updated_state)
                summary = self._strategy.summarize_entities(task, entities, words=120)
                task.result_summary = summary
                self._storage.write_json(
                    task_id,
                    "entities",
                    {"entities": [entity.model_dump() for entity in entities]},
                )
                self._publish_info(task, "summary", {"text": summary})
                if plan_remaining and len(plan_remaining) > 0:
                    step_count += 1
                    continue

            if plan_remaining is not None and len(plan_remaining) == 0:
                success = True
                logger.info("All plan steps completed", task_id=task_id)
                break

            step_count += 1

        if success:
            self._update_status(task, TaskStatus.DONE)
            task.contract.metrics["task_success"] = True
            logger.info("Task completed successfully", task_id=task_id)
            self._release_resources(task_id)
            return

        if persist_context and not self._stop_event.is_set():
            self._plan_failures.pop(task_id, None)
            self._task_plans.pop(task_id, None)
            self._task_token_budgets[task_id] = self._settings.token_budget
            task.contract.metrics["retry_count"] = task.contract.metrics.get("retry_count", 0) + 1
            self._task_deadlines[task_id] = time.time() + self._settings.task_timeout_seconds
            self._executor_agent.reset_task(task_id)
            self._update_status(task, TaskStatus.RUNNING)
            logger.info(
                "Task scheduled for retry with persistent context",
                task_id=task_id,
                retry=task.contract.metrics["retry_count"],
            )
            self._task_queue.put(task_id)
            return

        if task.status != TaskStatus.ERROR:
            self._update_status(task, TaskStatus.ERROR)
            logger.error("Task ended with error status", task_id=task_id)

        self._release_resources(task_id)

    def _release_resources(self, task_id: str) -> None:
        self._browser_manager.close_task(task_id)
        self._memory.clear_task(task_id)
        self._short_memory.pop(task_id, None)
        self._task_deadlines.pop(task_id, None)
        self._task_plans.pop(task_id, None)
        self._plan_failures.pop(task_id, None)
        self._task_token_budgets.pop(task_id, None)
        self._executor_agent.reset_task(task_id)
        self._strategy.reset_task(task_id)

    # Confirmation -------------------------------------------------------

    def _handle_confirmation(
        self,
        task: Task,
        action: Action,
        page_state: PageStatePack,
        planner_note: str,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        task.status = TaskStatus.NEEDS_CONFIRM
        payload = ConfirmPayload(
            task_id=task.id,
            step_id=uuid.uuid4().hex,
            action=action,
            summary=planner_note or f"Action {action.type}",
            screenshot_path=self._to_artifact_url(page_state.screenshot_path),
            form_preview={"goal": task.goal, "url": page_state.url},
        )
        event = self._make_event(task.id, "confirm_request", payload.model_dump())
        self._push_event(task, event)
        log_event(event)

        gate = ConfirmationGate()
        self._confirmations[task.id] = gate

        event_set = gate.event.wait(timeout=self._settings.step_timeout_seconds)
        self._confirmations.pop(task.id, None)
        task.status = TaskStatus.RUNNING
        self._publish_status(task, TaskStatus.RUNNING)

        if not event_set or gate.decision is None:
            return False, None
        return bool(gate.decision), gate.payload or None

    # Event helpers ------------------------------------------------------

    def _advance_plan(
        self,
        task_id: str,
        action: Action,
        observation: Observation,
        planner_note: Optional[str],
    ) -> None:
        plan = self._task_plans.get(task_id)
        if not plan:
            return
        if action.type in {"confirm_gate"}:
            return
        if not observation.ok:
            return

        marker = (planner_note or "").lower()
        if "[step-complete]" not in marker:
            return

        completed = plan.popleft()
        self._clear_plan_failure(task_id, completed)
        logger.info("Plan step completed", task_id=task_id, step=completed, action=action.type)
        self._executor_agent.clear_step(task_id, completed)
        task = self.get_task(task_id)
        if task:
            metrics = task.contract.metrics
            metrics["completed_steps"] = metrics.get("completed_steps", 0) + 1
            task.metadata.pop("user_inputs", None)

    def _publish_status(self, task: Task, status: TaskStatus) -> None:
        event = self._make_event(task.id, "status", {"status": status.value})
        self._push_event(task, event)
        log_event(event)

    def _publish_error(self, task: Task, message: str) -> None:
        task.status = TaskStatus.ERROR
        task.last_error = message
        event = self._make_event(task.id, "error", {"message": message})
        self._push_event(task, event)
        log_event(event)

    def _publish_info(self, task: Task, kind: str, payload: Dict[str, Any]) -> None:
        event = self._make_event(task.id, kind, payload)
        self._push_event(task, event)
        log_event(event)

    def _emit_step(self, task: Task, step: StepLog) -> None:
        observation = step.observation.model_dump() if step.observation else None
        if observation and observation.get("screenshot_path"):
            observation["screenshot_path"] = self._to_artifact_url(
                observation["screenshot_path"]
            )

        event = self._make_event(
            task.id,
            "step",
            {
                "step_id": step.id,
                "role": step.role,
                "action": step.action.model_dump() if step.action else None,
                "observation": observation,
                "note": step.note,
            },
        )
        self._push_event(task, event)
        log_event(event)

    def _update_status(self, task: Task, status: TaskStatus) -> None:
        task.status = status
        if status is TaskStatus.DONE:
            task.completed_at = datetime.now(UTC)
        self._publish_status(task, status)

    def _make_event(self, task_id: str, kind: str, payload: Dict[str, Any]) -> TaskLogEvent:
        return TaskLogEvent(
            id=uuid.uuid4().hex,
            task_id=task_id,
            timestamp=datetime.now(UTC),
            kind=kind,
            payload=payload,
        )

    def _push_event(self, task: Task, event: TaskLogEvent) -> None:
        task.append_event(event)
        queue_ref = self._event_queues.get(task.id)
        if queue_ref:
            queue_ref.put(event)

    def _to_artifact_url(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            rel = Path(path).resolve().relative_to(self._artifacts_root)
            return f"/artifacts/{rel.as_posix()}"
        except ValueError:
            return None

    def _log_strategy_thought(
        self,
        task: Task,
        short_memory: ShortTermMemory,
        step_index: int,
        planner_note: Optional[str],
        plan_step: Optional[str],
    ) -> None:
        thought = (planner_note or "").strip() or "(planner provided no note)"
        if plan_step:
            thought = f"{thought} [plan={plan_step}]"
        thought_log = StepLog(
            id=uuid.uuid4().hex,
            task_id=task.id,
            timestamp=datetime.now(UTC),
            role="planner",
            action=None,
            observation=None,
            note=f"Thought: {thought}",
            metadata={
                "step_index": step_index,
                "plan_step": plan_step,
            },
        )
        short_memory.add_step(thought_log)
        self._emit_step(task, thought_log)

    def _describe_action(
        self,
        action: Action,
        planner_note: Optional[str],
        plan_step: Optional[str],
    ) -> str:
        target = (
            action.params.get("locator")
            or action.params.get("text")
            or action.params.get("url")
            or ""
        )
        parts = [f"Action: {action.type}"]
        if target:
            parts.append(str(target)[:120])
        if plan_step:
            parts.append(f"plan={plan_step}")
        if planner_note:
            parts.append(f"intent={planner_note[:160]}")
        return " | ".join(part for part in parts if part)

    def _log_observation(
        self,
        task: Task,
        short_memory: ShortTermMemory,
        page_state: PageStatePack,
        observation: Observation,
        step_index: int,
    ) -> None:
        summary_parts = []
        if observation.note:
            summary_parts.append(observation.note)
        summary_parts.append(f"url={page_state.url}")
        if page_state.dom_diff:
            diff = page_state.dom_diff
            summary_parts.append(
                f"domΔ +{len(diff.added)}/-{len(diff.removed)}/~{len(diff.changed)}"
            )
        summary_parts.append(f"affordances={len(page_state.affordances)}")
        note = "Observation: " + " | ".join(summary_parts)
        observation_log = StepLog(
            id=uuid.uuid4().hex,
            task_id=task.id,
            timestamp=datetime.now(UTC),
            role="reader",
            action=None,
            observation=None,
            note=note[:400],
            metadata={
                "step_index": step_index,
                "dom_hash": page_state.dom_hash,
            },
        )
        short_memory.add_step(observation_log)
        self._emit_step(task, observation_log)

    def _make_context(
        self,
        task: Task,
        step_index: int,
        plan: Optional[Deque[str]],
    ) -> TaskContext:
        steps_remaining = max(0, self._settings.max_steps - step_index)
        deadline = self._task_deadlines.get(task.id, time.time())
        seconds_remaining = max(0.0, deadline - time.time())
        token_budget = self._task_token_budgets.get(task.id, self._settings.token_budget)
        plan_list = list(plan) if plan else []
        current_step = plan[0] if plan else None
        return TaskContext(
            goal=task.goal,
            plan=plan_list,
            current_step=current_step,
            steps_remaining=steps_remaining,
            seconds_remaining=seconds_remaining,
            token_budget_remaining=token_budget,
        )

    def _record_plan_failure(self, task: Task, plan_step: Optional[str]) -> None:
        if not plan_step:
            return
        tracker = self._plan_failures.setdefault(task.id, {})
        key = plan_step.strip().lower()
        tracker[key] = tracker.get(key, 0) + 1
        if tracker[key] < self._plan_failure_threshold:
            return

        tracker[key] = 0
        plan = self._task_plans.get(task.id)
        if not plan or len(plan) <= 1 or plan[0] != plan_step:
            return
        plan.rotate(-1)
        logger.warning(
            "Plan rotated after repeated failure",
            task_id=task.id,
            plan=list(plan),
            failed_step=plan_step,
        )
        self._publish_info(task, "plan", {"steps": list(plan), "source": "failure_reorder"})

    def _clear_plan_failure(self, task_id: str, plan_step: Optional[str]) -> None:
        if not plan_step:
            return
        tracker = self._plan_failures.get(task_id)
        if not tracker:
            return
        tracker.pop(plan_step.strip().lower(), None)

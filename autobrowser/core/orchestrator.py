from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from loguru import logger

from autobrowser.agents.applier import ApplierAgent
from autobrowser.agents.content import ContentAgent
from autobrowser.agents.critic import CriticAgent
from autobrowser.agents.extractor import ExtractorAgent
from autobrowser.agents.planner import PlannerAgent
from autobrowser.agents.researcher import ResearcherAgent
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
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        extractor: Optional[ExtractorAgent] = None,
        content: Optional[ContentAgent] = None,
        critic: Optional[CriticAgent] = None,
        applier: Optional[ApplierAgent] = None,
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
        self._safety = safety or SafetyEngine(self._settings)
        self._storage = storage or FileStorage(self._settings)
        self._artifacts_root = Path(self._settings.artifacts_dir).resolve()

        # Agents
        self._planner = planner or PlannerAgent(self._llm)
        self._researcher = researcher or ResearcherAgent()
        self._extractor = extractor or ExtractorAgent()
        self._content = content or ContentAgent(self._llm)
        self._critic = critic or CriticAgent(self._llm)
        self._applier = applier or ApplierAgent()

        self._confirmations: Dict[str, ConfirmationGate] = {}
        self._task_deadlines: Dict[str, float] = {}

        if hasattr(self._browser_manager, "start"):
            self._browser_manager.start()
        self._dispatcher.start()

    # Task lifecycle -----------------------------------------------------

    def create_task(self, goal: str, *, allow_high_risk: bool = False) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(
            id=task_id,
            goal=goal,
            created_at=datetime.utcnow(),
            status=TaskStatus.NEW,
            allow_high_risk=allow_high_risk,
        )
        logger.info("Task created", task_id=task_id, goal=goal, allow_high_risk=allow_high_risk)
        with self._tasks_lock:
            self._tasks[task_id] = task
            self._event_queues[task_id] = queue.Queue()
            self._short_memory[task_id] = ShortTermMemory()
            self._task_deadlines[task_id] = time.time() + self._settings.task_timeout_seconds

        self._publish_status(task, TaskStatus.NEW)
        self._task_queue.put(task_id)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        with self._tasks_lock:
            return list(self._tasks.values())

    def confirm_task_action(self, task_id: str, approve: bool) -> bool:
        gate = self._confirmations.get(task_id)
        if not gate:
            return False
        gate.decision = approve
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
            plan_steps = self._planner.build_initial_plan(task)
            logger.info("Initial plan ready", task_id=task_id, steps=plan_steps)
            self._publish_info(task, "plan", {"steps": plan_steps})
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
                page_state = self._reader.read(page, [task.goal], last_screenshot)
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
                logger.info(
                    "Planner deciding next action",
                    task_id=task_id,
                    step=step_count,
                    url=page_state.url,
                )
                action, planner_note = self._planner.decide_next_action(
                    task,
                    page_state,
                    short_memory.as_list(),
                    rag_entries,
                )
                logger.info(
                    "Planner issued action",
                    task_id=task_id,
                    step=step_count,
                    action=action.type,
                    note=planner_note,
                )

                if action.type == "confirm_gate":
                    if step_count < 5:
                        logger.warning(
                            "Skipping premature confirm_gate",
                            task_id=task_id,
                            step=step_count,
                        )
                        step_count += 1
                        continue
                    if not action.params.get("intent"):
                        logger.warning(
                            "confirm_gate missing intent, rejecting",
                            task_id=task_id,
                            step=step_count,
                        )
                        step_count += 1
                        continue

                signature = f"{action.type}:{json.dumps(action.params, sort_keys=True)}"
                if recent_action_signatures.count(signature) >= 2:
                    logger.warning(
                        "Detected repetitive action",
                        task_id=task_id,
                        step=step_count,
                        action=action.type,
                    )
                    recovery = self._critic.suggest_recovery(
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
                        timestamp=datetime.utcnow(),
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
                self._publish_error(task, f"Planner failed: {exc}")
                logger.exception("Planner next action failed", task_id=task_id)
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
                if not self._handle_confirmation(task, action, page_state, planner_note):
                    self._publish_error(task, "High risk action was rejected.")
                    logger.info("High risk action rejected by user", task_id=task_id)
                    break

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

            step_log = StepLog(
                id=uuid.uuid4().hex,
                task_id=task_id,
                timestamp=datetime.utcnow(),
                role="browser",
                action=action,
                observation=observation,
                note=planner_note,
            )
            short_memory.add_step(step_log)
            updated_state = self._reader.read(
                page,
                [task.goal],
                observation.screenshot_path,
            )
            self._memory.add_page_state(task_id, updated_state, observation=observation)
            self._emit_step(task, step_log)

            if not observation.ok:
                recovery = self._critic.suggest_recovery(task, short_memory.as_list(), observation)
                self._publish_info(task, "recovery", {"suggestion": recovery})
                logger.warning(
                    "Observation indicates failure",
                    task_id=task_id,
                    step=step_count,
                    note=observation.note,
                    error=observation.error_category,
                )
                step_count += 1
                continue

            # Optionally perform extraction/reporting
            if action.type == "extract":
                logger.info("Extraction action completed, building summary", task_id=task_id)
                entities = self._extractor.extract_entities(updated_state)
                summary = self._content.summarize_entities(task, entities, words=120)
                task.result_summary = summary
                self._storage.write_json(
                    task_id,
                    "entities",
                    {"entities": [entity.model_dump() for entity in entities]},
                )
                self._publish_info(task, "summary", {"text": summary})
                success = True
                break

            step_count += 1

        if success:
            self._update_status(task, TaskStatus.DONE)
            logger.info("Task completed successfully", task_id=task_id)
        else:
            if task.status != TaskStatus.ERROR:
                self._update_status(task, TaskStatus.ERROR)
                logger.error("Task ended with error status", task_id=task_id)

        self._browser_manager.close_task(task_id)
        self._memory.clear_task(task_id)
        logger.info("Task cleanup finished", task_id=task_id)

    # Confirmation -------------------------------------------------------

    def _handle_confirmation(
        self,
        task: Task,
        action: Action,
        page_state: PageStatePack,
        planner_note: str,
    ) -> bool:
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

        approved = gate.event.wait(timeout=self._settings.step_timeout_seconds)
        self._confirmations.pop(task.id, None)
        task.status = TaskStatus.RUNNING
        self._publish_status(task, TaskStatus.RUNNING)

        if not approved or gate.decision is None:
            return False
        return gate.decision

    # Event helpers ------------------------------------------------------

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
            task.completed_at = datetime.utcnow()
        self._publish_status(task, status)

    def _make_event(self, task_id: str, kind: str, payload: Dict[str, Any]) -> TaskLogEvent:
        return TaskLogEvent(
            id=uuid.uuid4().hex,
            task_id=task_id,
            timestamp=datetime.utcnow(),
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

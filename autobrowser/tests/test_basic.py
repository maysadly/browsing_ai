from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

if "loguru" not in sys.modules:
    class _StubLogger:
        def bind(self, **kwargs):
            return self

        def __getattr__(self, name: str):
            def _noop(*args, **kwargs):
                return None

            return _noop

        def opt(self, **kwargs):
            return self

    sys.modules["loguru"] = SimpleNamespace(logger=_StubLogger())

from autobrowser.agents.navigator import Navigator
from autobrowser.browser.manager import BrowserManager
from autobrowser.browser.reader import PageReader
from autobrowser.browser.tools import BrowserToolExecutor
from autobrowser.core.memory import ShortTermMemory
from autobrowser.core.orchestrator import TaskOrchestrator
from autobrowser.core.safety import SafetyEngine
from autobrowser.core.schemas import (
    Action,
    BoundingBox,
    DistilledNode,
    Observation,
    PageEntity,
    PageStatePack,
    RiskLevel,
    StepLog,
    Task,
    TaskContext,
    TaskStatus,
)
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


def test_safety_requires_confirmation_for_high_risk():
    settings = SimpleNamespace(allow_domains=[])
    engine = SafetyEngine(settings)
    task = Task(
        id="t1",
        goal="Submit a form",
        created_at=datetime.now(UTC),
        status=TaskStatus.NEW,
        risk_level=RiskLevel.MEDIUM,
        allow_high_risk=False,
    )
    action = Action(type="confirm_gate", params={"intent": "submit important form"})
    decision = engine.assess(task, action, description="Submitting a form")

    assert decision.risk_level is RiskLevel.HIGH
    assert decision.requires_confirmation is True


def test_safety_force_confirm_respected_with_high_risk():
    settings = SimpleNamespace(allow_domains=[])
    engine = SafetyEngine(settings)
    task = Task(
        id="t2",
        goal="Perform risky action",
        created_at=datetime.now(UTC),
        status=TaskStatus.NEW,
        risk_level=RiskLevel.MEDIUM,
        allow_high_risk=True,
    )
    action = Action(type="confirm_gate", params={"force_confirm": True})

    decision = engine.assess(task, action)

    assert decision.risk_level is RiskLevel.HIGH
    assert decision.requires_confirmation is True


def test_safety_enforces_click_throttling():
    settings = SimpleNamespace(allow_domains=[], max_clicks_per_second=1)
    engine = SafetyEngine(settings)
    task = Task(
        id="t_click",
        goal="Rapid clicking",
        created_at=datetime.now(UTC),
        status=TaskStatus.NEW,
        risk_level=RiskLevel.LOW,
        allow_high_risk=False,
    )
    action = Action(type="click", params={})

    first = engine.assess(task, action)
    assert first.requires_confirmation is False

    second = engine.assess(task, action)
    assert second.requires_confirmation is True


def test_safety_blocks_non_http_navigation():
    settings = SimpleNamespace(allow_domains=[], max_clicks_per_second=5)
    engine = SafetyEngine(settings)
    task = Task(
        id="t_nav",
        goal="Navigate safely",
        created_at=datetime.now(UTC),
        status=TaskStatus.NEW,
        risk_level=RiskLevel.LOW,
        allow_high_risk=False,
    )
    action = Action(type="navigate", params={"url": "javascript:alert(1)"})

    decision = engine.assess(task, action)
    assert decision.requires_confirmation is True
    assert "scheme" in decision.reason.lower()


def test_safety_submit_requires_confirmation_without_opt_in():
    settings = SimpleNamespace(allow_domains=[], max_clicks_per_second=5)
    engine = SafetyEngine(settings)
    task = Task(
        id="t_submit",
        goal="Submit form",
        created_at=datetime.now(UTC),
        status=TaskStatus.NEW,
        risk_level=RiskLevel.LOW,
        allow_high_risk=False,
    )
    action = Action(type="submit", params={"intent": "submit payment"})

    decision = engine.assess(task, action)
    assert decision.requires_confirmation is True
    assert decision.risk_level is RiskLevel.HIGH

    task.allow_high_risk = True
    decision_allowed = engine.assess(task, action)
    assert decision_allowed.requires_confirmation is False

def test_short_term_memory_window():
    memory = ShortTermMemory(window=3)
    for idx in range(5):
        step = StepLog(
            id=f"s{idx}",
            task_id="task",
            timestamp=datetime.now(UTC),
            role="planner",
            note=f"Step {idx}",
            action=Action(type="wait", params={}),
        )
        memory.add_step(step)

    history = memory.as_list()
    assert len(history) == 3
    assert history[0].startswith("[planner]") and "Step 2" in history[0]
    assert history[-1].startswith("[planner]") and "Step 4" in history[-1]


class FakeStrategy:
    def __init__(self) -> None:
        self.calls = 0
        self._navigator = Navigator()

    def build_initial_plan(self, task: Task, context: Optional[TaskContext] = None) -> List[str]:
        if context:
            context.consume_tokens(5)
        return [f"Work on {task.goal}"]

    def decide_next_action(
        self,
        task,
        page_state,
        history,
        rag,
        plan_step=None,
        failure_count=0,
        failure_reason=None,
        user_inputs=None,
        context: Optional[TaskContext] = None,
    ):
        if context:
            context.consume_tokens(3)
        self.calls += 1
        if self.calls <= 6:
            return Action(type="confirm_gate", params={"intent": "submit form"}), "Requesting confirmation"
        return Action(type="extract", params={"locator": "main"}), "Extracting results [step-complete]"

    def suggest_recovery(self, task: Task, history: List[str], observation: Optional[str]) -> str:
        return "Retry with adjusted selector."

    def summarize_entities(self, task: Task, entities: List[PageEntity], words: int = 120) -> str:
        return "Summary: Playwright article summary."

    @property
    def navigator(self) -> Navigator:
        return self._navigator


class FakeBrowserManager:
    class Page:
        url = "https://example.org"

        def title(self):
            return "Example"

    def start(self):
        return None

    def new_page(self, task_id: str):
        return self.Page()

    def close_task(self, task_id: str):
        return None

    def shutdown(self):
        return None


class FakeBrowserTools:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    def execute(self, page, task_id, step_number, action):
        path = self.artifacts_dir / task_id
        path.mkdir(parents=True, exist_ok=True)
        screenshot = path / f"step_{step_number:03d}.png"
        screenshot.write_bytes(b"\x89PNG\r\n")
        return Observation(
            ok=True,
            note=f"Executed {action.type}",
            affordances=[],
            text_chunks=["Example snippet"],
            screenshot_path=str(screenshot),
            page_url="https://example.org",
            page_title="Example",
        )


class FakeReader:
    def read(self, page, goals, screenshot_path, **kwargs):
        return PageStatePack(
            url="https://example.org",
            title="Example",
            goals=goals,
            affordances=[],
            text_snippets=["Playwright is a browser automation library."],
            forms=[],
            entities=[],
            screenshot_path=screenshot_path,
            annotated_screenshot_path=screenshot_path,
        )


class FakeMemory:
    def add_page_state(self, task_id, state, observation=None):
        return None

    def search(self, task_id, query, top_k=5):
        return []

    def clear_task(self, task_id):
        return None


class FakeExtractor:
    def extract_entities(self, page_state):
        return [
            PageEntity(title="Playwright Article", summary="An article about Playwright."),
        ]


class FakeStorage:
    def __init__(self, settings):
        self._settings = settings
        Path(settings.artifacts_dir).mkdir(parents=True, exist_ok=True)

    def write_json(self, task_id, name, payload):
        path = Path(self._settings.artifacts_dir) / task_id
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{name}.json"
        file_path.write_text("{}", encoding="utf-8")
        return file_path


def test_browser_manager_close_task_releases_resources(monkeypatch, tmp_path):
    closed = []
    stopped = []

    class DummyContext:
        def __init__(self) -> None:
            self.pages: List[object] = []

        def close(self) -> None:
            closed.append(True)

        def new_page(self) -> object:
            page = object()
            self.pages.append(page)
            return page

    class DummyChromium:
        def launch_persistent_context(self, **kwargs):
            return DummyContext()

    class DummyPlaywright:
        def __init__(self) -> None:
            self.chromium = DummyChromium()

        def stop(self) -> None:
            stopped.append(True)

    class DummyRunner:
        def start(self):
            return DummyPlaywright()

    monkeypatch.setattr(
        "autobrowser.browser.manager.sync_playwright",
        lambda: DummyRunner(),
    )

    settings = SimpleNamespace(
        session_dir=tmp_path / "sessions",
        max_concurrent_tasks=1,
    )

    manager = BrowserManager(settings)
    page = manager.new_page("task-1")
    assert page is not None
    assert "task-1" in manager._handles

    manager.close_task("task-1")
    assert closed, "Context should be closed"
    assert stopped, "Playwright runner should be stopped"
    assert "task-1" not in manager._handles

    # Should be a no-op when closing again.
    manager.close_task("task-1")


def test_extract_action_reports_failure_when_empty(tmp_path):
    class DummyLocator:
        def all_text_contents(self):
            return []

        def scroll_into_view_if_needed(self, timeout: int = 0) -> None:
            return None

    class DummyPage:
        url = "https://example.org"

        def locator(self, selector: str):
            return DummyLocator()

        def screenshot(self, path: str):
            Path(path).write_bytes(b"")

    executor = BrowserToolExecutor(tmp_path)
    action = Action(type="extract", params={"locator": "main"})
    observation = executor.execute(DummyPage(), "task-x", 1, action)

    assert observation.ok is False
    assert observation.error_category == "HandlerException"
    assert "No elements" in observation.note


def test_click_respects_force_and_options(tmp_path):
    class RecordingLocator:
        def __init__(self) -> None:
            self.kwargs: Dict[str, Any] | None = None

        def count(self) -> int:
            return 1

        def click(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def scroll_into_view_if_needed(self, timeout: int = 0) -> None:
            return None

    class RecordingPage:
        def __init__(self, locator: RecordingLocator) -> None:
            self._locator = locator
            self.url = "https://example.org"

        def locator(self, selector: str):
            assert selector == "[data-qa='search-button']"
            return self._locator

        def title(self) -> str:
            return "Example"

        def screenshot(self, path: str) -> None:
            Path(path).write_bytes(b"")

    locator = RecordingLocator()
    page = RecordingPage(locator)
    executor = BrowserToolExecutor(tmp_path)
    action = Action(
        type="click",
        params={
            "locator": "[data-qa='search-button']",
            "force": True,
            "no_wait_after": True,
            "click_count": 2,
            "delay": 50,
            "button": "left",
        },
    )

    observation = executor.execute(page, "task-force", 3, action)

    assert observation.ok is True
    assert locator.kwargs is not None
    assert locator.kwargs.get("force") is True
    assert locator.kwargs.get("no_wait_after") is True
    assert locator.kwargs.get("click_count") == 2
    assert locator.kwargs.get("delay") == 50
    assert locator.kwargs.get("button") == "left"


def test_click_retries_with_force_on_timeout(tmp_path):
    class FlakyLocator:
        def __init__(self) -> None:
            self.calls = 0
            self.kwargs: Dict[str, Any] | None = None

        def count(self) -> int:
            return 1

        def click(self, **kwargs: Any) -> None:
            self.calls += 1
            if self.calls == 1:
                raise PlaywrightTimeoutError("timeout")
            self.kwargs = kwargs

        def scroll_into_view_if_needed(self, timeout: int = 0) -> None:
            return None

    class FlakyPage:
        def __init__(self, locator: FlakyLocator) -> None:
            self._locator = locator
            self.url = "https://example.org"

        def locator(self, selector: str):
            return self._locator

        def title(self) -> str:
            return "Example"

        def screenshot(self, path: str) -> None:
            Path(path).write_bytes(b"")

    locator = FlakyLocator()
    page = FlakyPage(locator)
    executor = BrowserToolExecutor(tmp_path)
    action = Action(type="click", params={"locator": "#login"})

    observation = executor.execute(page, "task-retry", 4, action)

    assert observation.ok is True
    assert locator.calls == 2
    assert locator.kwargs is not None
    assert locator.kwargs.get("force") is True
    assert "[force]" in observation.note


def test_type_handles_radio_inputs(tmp_path):
    class RadioLocator:
        def __init__(self) -> None:
            self.checked = False

        def count(self) -> int:
            return 1

        def get_attribute(self, name: str):
            if name == "type":
                return "radio"
            if name == "value":
                return "APPLICANT"
            return None

        def scroll_into_view_if_needed(self, timeout: int = 0) -> None:
            return None

        def check(self, timeout: int = 0, force: bool = False) -> None:
            self.checked = True

    class RadioPage:
        def __init__(self, locator: RadioLocator) -> None:
            self.locator_obj = locator
            self.url = "https://example.org"

        def locator(self, selector: str):
            return self.locator_obj

        def screenshot(self, path: str):
            Path(path).write_bytes(b"")

        def title(self):
            return "Example"

    radio = RadioLocator()
    page = RadioPage(radio)
    executor = BrowserToolExecutor(tmp_path)

    action = Action(
        type="type",
        params={
            "locator": "[name='account-type']",
            "value": "APPLICANT",
        },
    )

    observation = executor.execute(page, "task-radio", 5, action)

    assert observation.ok is True
    assert radio.checked is True


def test_orchestrator_confirmation_flow(tmp_path):
    artifacts = tmp_path / "artifacts"
    sessions = tmp_path / "sessions"
    logs = tmp_path / "logs"
    settings = SimpleNamespace(
        artifacts_dir=artifacts,
        session_dir=sessions,
        logs_dir=logs,
        max_concurrent_tasks=1,
        token_budget=2000,
        max_steps=8,
        task_timeout_seconds=5,
        step_timeout_seconds=2,
        sse_heartbeat_seconds=1,
        openai_model="test-model",
        embedding_model="test-emb",
        allow_domains=[],
    )

    orchestrator = TaskOrchestrator(
        settings=settings,
        llm=object(),
        browser_manager=FakeBrowserManager(),
        browser_tools=FakeBrowserTools(artifacts),
        reader=FakeReader(),
        memory=FakeMemory(),
        safety=SafetyEngine(settings),
        storage=FakeStorage(settings),
        strategy=FakeStrategy(),
        extractor=FakeExtractor(),
    )

    task = orchestrator.create_task(
        "Open browser and summarise playwright docs", allow_high_risk=False
    )

    confirmed = False
    for _ in range(30):
        if orchestrator.confirm_task_action(task.id, True):
            confirmed = True
            break
        time.sleep(0.1)
    assert confirmed

    deadline = time.time() + 5
    final_task = task
    while time.time() < deadline:
        final_task = orchestrator.get_task(task.id)
        if final_task and final_task.status in (TaskStatus.DONE, TaskStatus.ERROR):
            break
        time.sleep(0.1)

    assert final_task.status is TaskStatus.DONE
    assert "Summary" in (final_task.result_summary or "")

    orchestrator.shutdown()


def test_page_reader_dom_diff_detects_changes():
    reader = PageReader(annotation_top_k=0)
    bbox = BoundingBox(x=10, y=10, width=40, height=12)

    def build_node(text: str) -> DistilledNode:
        node_hash = reader._hash_node(
            path="button#submit",
            tag="button",
            role="button",
            label=text,
            name=None,
            text=text,
            locator="#submit",
            bbox=bbox,
            visible=True,
        )
        return DistilledNode(
            node_id=1,
            path="button#submit",
            tag="button",
            role="button",
            label=text,
            name=None,
            text=text,
            locator="#submit",
            bbox=bbox,
            visible=True,
            hash=node_hash,
        )

    first = build_node("Submit")
    first_hash = hashlib.sha1(f"{first.path}:{first.hash}".encode("utf-8")).hexdigest()
    added_diff = reader._update_dom_snapshot("task:page", [first], first_hash)
    assert added_diff is not None
    assert added_diff.added and added_diff.added[0].summary.startswith("button")

    updated = build_node("Submit now")
    updated_hash = hashlib.sha1(f"{updated.path}:{updated.hash}".encode("utf-8")).hexdigest()
    changed_diff = reader._update_dom_snapshot("task:page", [updated], updated_hash)
    assert changed_diff is not None
    assert changed_diff.changed
    assert "text" in changed_diff.changed[0].fields_changed


def test_page_reader_annotation_draws_labels(tmp_path):
    reader = PageReader(annotation_top_k=3)
    screenshot_path = tmp_path / "s.png"
    Image.new("RGB", (200, 200), color="white").save(screenshot_path)

    bbox = BoundingBox(x=20, y=30, width=60, height=40)
    node_hash = reader._hash_node(
        path="button#annotate",
        tag="button",
        role="button",
        label="Annotate",
        name=None,
        text="Annotate",
        locator="#annotate",
        bbox=bbox,
        visible=True,
    )
    node = DistilledNode(
        node_id=1,
        path="button#annotate",
        tag="button",
        role="button",
        label="Annotate",
        name=None,
        text="Annotate",
        locator="#annotate",
        bbox=bbox,
        visible=True,
        hash=node_hash,
    )

    annotated_path = reader._annotate_screenshot(str(screenshot_path), [node])
    assert annotated_path is not None
    assert Path(annotated_path).exists()

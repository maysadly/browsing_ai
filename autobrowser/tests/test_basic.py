from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from autobrowser.core.memory import ShortTermMemory
from autobrowser.core.orchestrator import TaskOrchestrator
from autobrowser.core.safety import SafetyEngine
from autobrowser.core.schemas import (
    Action,
    Observation,
    PageEntity,
    PageStatePack,
    RiskLevel,
    StepLog,
    Task,
    TaskStatus,
)


def test_safety_requires_confirmation_for_high_risk():
    settings = SimpleNamespace(allow_domains=[])
    engine = SafetyEngine(settings)
    task = Task(
        id="t1",
        goal="Submit a form",
        created_at=datetime.utcnow(),
        status=TaskStatus.NEW,
        risk_level=RiskLevel.MEDIUM,
        allow_high_risk=False,
    )
    action = Action(type="confirm_gate", params={"intent": "submit important form"})
    decision = engine.assess(task, action, description="Submitting a form")

    assert decision.risk_level is RiskLevel.HIGH
    assert decision.requires_confirmation is True


def test_short_term_memory_window():
    memory = ShortTermMemory(window=3)
    for idx in range(5):
        step = StepLog(
            id=f"s{idx}",
            task_id="task",
            timestamp=datetime.utcnow(),
            role="planner",
            note=f"Step {idx}",
            action=Action(type="wait", params={}),
        )
        memory.add_step(step)

    history = memory.as_list()
    assert len(history) == 3
    assert history[0].startswith("[planner]") and "Step 2" in history[0]
    assert history[-1].startswith("[planner]") and "Step 4" in history[-1]


class FakePlanner:
    def __init__(self) -> None:
        self.calls = 0

    def build_initial_plan(self, task: Task) -> List[str]:
        return [f"Work on {task.goal}"]

    def decide_next_action(self, task, page_state, history, rag):
        if self.calls == 0:
            self.calls += 1
            return Action(type="confirm_gate", params={"intent": "submit form"}), "Requesting confirmation"
        self.calls += 1
        return Action(type="extract", params={"locator": "main"}), "Extracting results"


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
    def read(self, page, goals, screenshot_path):
        return PageStatePack(
            url="https://example.org",
            title="Example",
            goals=goals,
            affordances=[],
            text_snippets=["Playwright is a browser automation library."],
            forms=[],
            entities=[],
            screenshot_path=screenshot_path,
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


class FakeContent:
    def summarize_entities(self, task, entities, words=120):
        return "Summary: Playwright article summary."


class FakeCritic:
    def suggest_recovery(self, task, history, observation):
        return "Retry with adjusted selector."


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
        max_steps=4,
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
        planner=FakePlanner(),
        extractor=FakeExtractor(),
        content=FakeContent(),
        critic=FakeCritic(),
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

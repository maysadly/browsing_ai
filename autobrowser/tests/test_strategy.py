from __future__ import annotations

import sys
import json
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List
from types import SimpleNamespace

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

from autobrowser.agents.strategy import StrategyAgent
from autobrowser.agents.navigator import Navigator
from autobrowser.core.schemas import (
    Action,
    Affordance,
    PageStatePack,
    Task,
    TaskContext,
)
from autobrowser.core.heuristics import HeuristicStore


class _StubPromptRegistry:
    def load(self, name: str) -> str:
        return name


class _StubLLM:
    def __init__(self) -> None:
        self.prompt_registry = _StubPromptRegistry()
        self.requests: List[Dict[str, Any]] = []

    def chat(self, system: str, messages: List[Dict[str, Any]], **kwargs: Any):
        self.requests.append({"system": system, "messages": messages, **kwargs})
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"type": "click", "params": {"locator": "#cta"}}'
                    }
                }
            ]
        }

    def count_tokens(self, text: str) -> int:
        return min(len(text), 8)


def test_repair_missing_action_preserves_context_and_payload():
    llm = _StubLLM()
    planner = StrategyAgent(llm, navigator=Navigator())

    task = Task(id="task-1", goal="Click the CTA", created_at=datetime.now(UTC))
    page_state = PageStatePack(
        url="https://example.org",
        title="Example",
        affordances=[Affordance(role="button", name="CTA", locator="#cta")],
        text_snippets=[],
        forms=[],
        entities=[],
    )
    history = ["[planner] Observed CTA button"]
    rag = ["CTA button is primary action"]
    context = TaskContext(
        goal=task.goal,
        steps_remaining=5,
        seconds_remaining=120.0,
        token_budget_remaining=64,
    )

    remaining_before = context.token_budget_remaining
    action = planner._repair_missing_action(  # type: ignore[attr-defined]
        task,
        page_state,
        history,
        rag,
        planner_message={"role": "assistant", "content": "No tool call emitted."},
        plan_step="Click CTA button",
        failure_count=2,
        failure_reason="timeout",
        user_inputs={"otp": "***"},
        context=context,
    )

    assert action.type == "click"
    assert action.params["locator"] == "#cta"
    assert context.token_budget_remaining < remaining_before

    assert llm.requests, "LLM should receive a repair request"
    payload = json.loads(llm.requests[0]["messages"][0]["content"])
    assert payload["goal"] == task.goal
    assert payload["planner_output"]["content"] == "No tool call emitted."
    assert payload["plan_step"] == "Click CTA button"
    assert payload["token_budget_remaining"] == remaining_before


def test_type_action_heuristic_fills_locator_from_affordances():
    llm = _StubLLM()
    agent = StrategyAgent(llm, navigator=Navigator())

    task = Task(id="task-2", goal="Enter email", created_at=datetime.now(UTC))
    page_state = PageStatePack(
        url="https://example.org/form",
        title="Example Form",
        affordances=[
            Affordance(role="textbox", name="Email", locator="[name='email']"),
        ],
        text_snippets=[],
        forms=[],
        entities=[],
    )

    action = Action(type="type", params={"text": "user@example.org"})
    params = agent._apply_heuristics(task, page_state, action)

    assert params["locator"] == "[name='email']"
    assert params.get("clear") is True


def test_workflow_state_tracks_authentication_and_progress():
    llm = _StubLLM()
    with tempfile.TemporaryDirectory() as tmp:
        heuristics_path = Path(tmp) / "heuristics.json"
        heuristics = HeuristicStore(heuristics_path)
        agent = StrategyAgent(llm, navigator=Navigator(), heuristics=heuristics)

        task = Task(id="task-auth", goal="Apply for jobs", created_at=datetime.now(UTC))

        login_state = PageStatePack(
            url="https://hh.ru/login",
            title="Login",
            affordances=[],
            text_snippets=["Пожалуйста, Войти в аккаунт чтобы продолжить."],
            forms=[],
            entities=[],
        )
        state = agent._update_workflow_state(task, login_state, history=[])
        assert state.auth_prompt_visible is True
        assert state.authenticated is False

        dashboard_state = PageStatePack(
            url="https://hh.ru/profile",
            title="Profile",
            affordances=[
                Affordance(role="button", name="Откликнуться", locator="[data-qa='respond-button']"),
            ],
            text_snippets=[
                "Ваше резюме готово к отправке.",
                "Выйти",
            ],
            forms=[],
            entities=[],
        )
        history = ["typed search query", "Отклик отправлен"]
        updated = agent._update_workflow_state(task, dashboard_state, history=history)

        assert updated is state
        assert updated.auth_prompt_visible is False
        assert updated.authenticated is True
        assert updated.profile_reviewed is True
        assert updated.search_started is True
        assert updated.applications_submitted == 1


def test_apply_heuristics_prefers_recorded_locator():
    llm = _StubLLM()
    with tempfile.TemporaryDirectory() as tmp:
        heuristics = HeuristicStore(Path(tmp) / "heuristics.json")
        heuristics.record(
            "search_input",
            domain="hh.ru",
            locator="[data-qa='search-input']",
            role="textbox",
        )
        agent = StrategyAgent(llm, navigator=Navigator(), heuristics=heuristics)

        task = Task(id="task-heuristic", goal="Find AI jobs", created_at=datetime.now(UTC))
        page_state = PageStatePack(
            url="https://hh.ru",
            title="Job Search",
            affordances=[
                Affordance(role="textbox", name="Поиск", locator="[data-qa='search-input']"),
            ],
            text_snippets=[],
            forms=[],
            entities=[],
        )
        action = Action(type="query", params={})
        params = agent._apply_heuristics(task, page_state, action)

        assert params["locator"] == "[data-qa='search-input']"


def test_confirm_gate_heuristic_fills_intent():
    llm = _StubLLM()
    agent = StrategyAgent(llm, navigator=Navigator())

    task = Task(id="task-confirm", goal="Authenticate", created_at=datetime.now(UTC))
    page_state = PageStatePack(
        url="https://example.org/login",
        title="Login - Example",
        affordances=[],
        text_snippets=[],
        forms=[],
        entities=[],
    )
    action = Action(type="confirm_gate", params={})
    params = agent._apply_heuristics(
        task,
        page_state,
        action,
        plan_step="Ask user to login",
        failure_reason="auth_required",
    )

    assert params["intent"] == "Resolve blocker: auth required"


def test_postprocess_confirms_gate_intent():
    llm = _StubLLM()
    agent = StrategyAgent(llm, navigator=Navigator())
    task = Task(id="task-confirm-2", goal="Authenticate", created_at=datetime.now(UTC))
    page_state = PageStatePack(
        url="https://example.org/login",
        title="Login",
        affordances=[],
        text_snippets=["Войти, чтобы продолжить."],
        forms=[],
        entities=[],
    )
    workflow = agent._update_workflow_state(task, page_state, history=[])
    workflow.auth_prompt_visible = True
    workflow.authenticated = False

    action = Action(type="confirm_gate", params={})
    agent._max_repair_attempts = -1
    repaired = agent._postprocess_action(
        task,
        action,
        page_state,
        history=[],
        rag=[],
        workflow_state=workflow,
    )
    assert repaired.params["intent"] == "Authenticate user to continue automation"

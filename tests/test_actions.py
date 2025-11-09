from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent.browser.controller import BrowserController
from agent.browser.sandbox import SandboxPolicy
from agent.types import Action


@dataclass
class DummyMouse:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def wheel(self, x: int, y: int) -> None:
        self.calls.append(("wheel", (x, y), {}))


@dataclass
class DummyKeyboard:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    async def press(self, keys: str, timeout: int) -> None:
        self.calls.append(("press", (keys,), {"timeout": timeout}))


@dataclass
class DummyLocator:
    page: "DummyPage"
    selector: str

    @property
    def first(self) -> "DummyLocator":
        return self

    def locator(self, nested: str) -> "DummyLocator":
        self.page.calls.append(("locator.locator", (self.selector, nested), {}))
        return self

    async def count(self) -> int:
        self.page.calls.append(("locator.count", (self.selector,), {}))
        return 1

    async def wait_for(self, state: str, timeout: int) -> None:
        self.page.calls.append(("locator.wait_for", (self.selector,), {"state": state, "timeout": timeout}))

    async def click(self, timeout: int) -> None:
        self.page.calls.append(("locator.click", (self.selector,), {"timeout": timeout}))

    async def press(self, key: str, timeout: int) -> None:
        self.page.calls.append(("locator.press", (self.selector, key), {"timeout": timeout}))

    async def evaluate(self, script: str):  # type: ignore[override]
        self.page.calls.append(("locator.evaluate", (self.selector, script), {}))
        return self.page.element_metadata.get(
            self.selector,
            {
                "tag": "input",
                "type": "text",
                "name": "",
                "placeholder": "",
                "ariaLabel": "",
                "dataQa": "",
                "role": "",
            },
        )


class DummyPage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.url = "https://example.com"
        self.mouse = DummyMouse()
        self.keyboard = DummyKeyboard()
        self.element_metadata: dict[str, dict[str, Any]] = {}

    def locator(self, selector: str) -> DummyLocator:  # type: ignore[override]
        self.calls.append(("locator", (selector,), {}))
        return DummyLocator(self, selector)

    async def wait_for_selector(self, selector: str, state: str, timeout: int) -> None:
        self.calls.append(("wait_for_selector", (selector,), {"state": state, "timeout": timeout}))

    async def click(self, selector: str, timeout: int) -> None:
        self.calls.append(("click", (selector,), {"timeout": timeout}))

    async def query_selector(self, selector: str):  # type: ignore[override]
        self.calls.append(("query_selector", (selector,), {}))
        return self

    async def scroll_into_view_if_needed(self, timeout: int) -> None:
        self.calls.append(("scroll_into_view_if_needed", tuple(), {"timeout": timeout}))

    async def fill(self, selector: str, text: str, timeout: int) -> None:
        self.calls.append(("fill", (selector, text), {"timeout": timeout}))

    async def inner_text(self, selector: str) -> str:
        self.calls.append(("inner_text", (selector,), {}))
        return "example text"

    async def evaluate(self, script: str):  # type: ignore[override]
        self.calls.append(("evaluate", (script,), {}))
        return ""

    async def focus(self, selector: str) -> None:
        self.calls.append(("focus", (selector,), {}))

    async def select_option(self, selector: str, label: str) -> None:
        self.calls.append(("select_option", (selector, label), {}))

    async def screenshot(self, path: str, full_page: bool) -> None:
        self.calls.append(("screenshot", (path,), {"full_page": full_page}))

    async def wait_for_load_state(self, state: str, timeout: int) -> None:
        self.calls.append(("wait_for_load_state", (state,), {"timeout": timeout}))


@pytest.mark.asyncio
async def test_click_waits_and_clicks(tmp_path: Path) -> None:
    sandbox = SandboxPolicy(
        allowed_hosts={"example.com"},
        max_steps=5,
        step_timeout_s=5,
        run_timeout_s=60,
        allow_js_eval=False,
        allow_file_upload=False,
    )
    controller = BrowserController(sandbox=sandbox, artifact_dir=tmp_path, headless=True)
    dummy_page = DummyPage()
    controller._page = dummy_page  # type: ignore[attr-defined]

    action = Action(type="click", selector="#login", reason="Simulate clicking login button in test")
    result = await controller.click(action)

    assert "Clicked" in result.summary
    call_names = [call[0] for call in dummy_page.calls]
    assert "locator" in call_names
    assert "locator.count" in call_names
    assert "locator.click" in call_names


@pytest.mark.asyncio
async def test_type_fills_input(tmp_path: Path) -> None:
    sandbox = SandboxPolicy(
        allowed_hosts={"example.com"},
        max_steps=5,
        step_timeout_s=5,
        run_timeout_s=60,
        allow_js_eval=False,
        allow_file_upload=False,
    )
    controller = BrowserController(sandbox=sandbox, artifact_dir=tmp_path, headless=True)
    dummy_page = DummyPage()
    controller._page = dummy_page  # type: ignore[attr-defined]

    action = Action(
        type="type",
        selector="#search",
        text="query",
        reason="Simulate entering a query in test input",
    )
    result = await controller.type_text(action)

    assert "Typed" in result.summary
    assert any(call[0] == "fill" for call in dummy_page.calls)


@pytest.mark.asyncio
async def test_type_auto_submits_search_input(tmp_path: Path) -> None:
    sandbox = SandboxPolicy(
        allowed_hosts={"example.com"},
        max_steps=5,
        step_timeout_s=5,
        run_timeout_s=60,
        allow_js_eval=False,
        allow_file_upload=False,
    )
    controller = BrowserController(sandbox=sandbox, artifact_dir=tmp_path, headless=True)
    dummy_page = DummyPage()
    dummy_page.element_metadata["#search"] = {
        "tag": "input",
        "type": "search",
        "name": "search",
        "placeholder": "Поиск",
        "ariaLabel": "Поиск вакансий",
        "dataQa": "search-input",
        "role": "textbox",
    }
    controller._page = dummy_page  # type: ignore[attr-defined]

    action = Action(
        type="type",
        selector="#search",
        text="AI инженер",
        reason="Simulate entering search query",
    )
    result = await controller.type_text(action)

    assert "auto-submitted" in result.summary
    assert any(call[0] == "locator.press" for call in dummy_page.calls)

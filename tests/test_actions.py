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


class DummyPage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.url = "https://example.com"
        self.mouse = DummyMouse()
        self.keyboard = DummyKeyboard()

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

    action = Action(type="click", selector="#login")
    result = await controller.click(action)

    assert "Clicked" in result.summary
    assert dummy_page.calls[0][0] == "wait_for_selector"
    assert dummy_page.calls[1][0] == "click"


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

    action = Action(type="type", selector="#search", text="query")
    result = await controller.type_text(action)

    assert "Typed" in result.summary
    assert any(call[0] == "fill" for call in dummy_page.calls)

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    async_playwright,
)

from ..errors import BrowserError, SandboxViolation
from ..types import Action
from .sandbox import SandboxPolicy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BrowserActionResult:
    """Structured summary of an executed action."""

    summary: str
    artifacts: list[str]
    url: str | None


class BrowserController:
    """High-level Playwright wrapper with sandbox enforcement."""

    def __init__(self, sandbox: SandboxPolicy, artifact_dir: Path, headless: bool = True) -> None:
        self._sandbox = sandbox
        self._artifact_dir = artifact_dir
        self._headless = headless
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.stop()

    async def start(self) -> None:
        if self._page is not None:
            return
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=self._headless)
    context = await browser.new_context()
    page = await context.new_page()

    self._playwright = playwright
    self._browser = browser
    self._context = context
    self._page = page

    async def stop(self) -> None:
        if self._page is not None:
            await self._page.close()
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise BrowserError("Browser not started")
        return self._page

    async def ensure_ready(self) -> None:
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=self._sandbox.step_timeout_ms)
        except PlaywrightError:  # pragma: no cover - tolerance for short waits
            logger.debug("Page load state wait timed out")

    async def perform(self, action: Action) -> BrowserActionResult:
        handlers = {
            "open_url": self.open_url,
            "click": self.click,
            "type": self.type_text,
            "press": self.press,
            "select": self.select,
            "scroll": self.scroll,
            "wait": self.wait,
            "extract_text": self.extract_text,
            "screenshot": self.screenshot,
            "evaluate_js": self.evaluate_js,
            "upload_file": self.upload_file,
            "finish": self._noop,
            "abort": self._noop,
        }
        handler = handlers.get(action.type)
        if handler is None:  # pragma: no cover - schema ensures coverage
            raise BrowserError(f"Unsupported action type {action.type}")
        return await handler(action)

    async def _noop(self, action: Action) -> BrowserActionResult:
        return BrowserActionResult(summary=f"No-op action {action.type}", artifacts=[], url=self.page.url)

    async def open_url(self, action: Action) -> BrowserActionResult:
        if not action.url:
            raise BrowserError("open_url requires url field")
        self._sandbox.validate_navigation(action.url)
        await self.page.goto(action.url, wait_until="domcontentloaded", timeout=self._sandbox.step_timeout_ms)
        await self.ensure_ready()
        return BrowserActionResult(summary=f"Opened {action.url}", artifacts=[], url=self.page.url)

    async def click(self, action: Action) -> BrowserActionResult:
        if not action.selector:
            raise BrowserError("click requires selector")
        selector = action.selector
        await self.page.wait_for_selector(selector, state="visible", timeout=self._sandbox.step_timeout_ms)
        try:
            await self.page.click(selector, timeout=self._sandbox.step_timeout_ms)
        except PlaywrightError:
            element = await self.page.query_selector(selector)
            if element is None:
                raise BrowserError(f"Element not found for selector {selector}")
            await element.scroll_into_view_if_needed(timeout=self._sandbox.step_timeout_ms)
            await self.page.click(selector, timeout=self._sandbox.step_timeout_ms)
        await self.ensure_ready()
        return BrowserActionResult(summary=f"Clicked {selector}", artifacts=[], url=self.page.url)

    async def type_text(self, action: Action) -> BrowserActionResult:
        if not action.selector or action.text is None:
            raise BrowserError("type requires selector and text")
        await self.page.wait_for_selector(action.selector, state="visible", timeout=self._sandbox.step_timeout_ms)
        await self.page.fill(action.selector, action.text, timeout=self._sandbox.step_timeout_ms)
        return BrowserActionResult(summary=f"Typed in {action.selector}", artifacts=[], url=self.page.url)

    async def press(self, action: Action) -> BrowserActionResult:
        if not action.selector or not action.keys:
            raise BrowserError("press requires selector and keys")
        await self.page.focus(action.selector)
        await self.page.keyboard.press(action.keys, timeout=self._sandbox.step_timeout_ms)
        return BrowserActionResult(summary=f"Pressed {action.keys}", artifacts=[], url=self.page.url)

    async def select(self, action: Action) -> BrowserActionResult:
        if not action.selector or action.text is None:
            raise BrowserError("select requires selector and text")
        await self.page.select_option(action.selector, label=action.text)
        return BrowserActionResult(summary=f"Selected {action.text}", artifacts=[], url=self.page.url)

    async def scroll(self, action: Action) -> BrowserActionResult:
        direction = action.scroll or "down"
        delta = {"up": -400, "down": 400, "to_top": -10_000, "to_bottom": 10_000}[direction]
        await self.page.mouse.wheel(0, delta)
        await asyncio.sleep(0.3)
        return BrowserActionResult(summary=f"Scrolled {direction}", artifacts=[], url=self.page.url)

    async def wait(self, action: Action) -> BrowserActionResult:
        timeout_s = (action.timeout_ms or self._sandbox.step_timeout_ms) / 1000
        await asyncio.sleep(timeout_s)
        return BrowserActionResult(summary=f"Waited {timeout_s:.2f}s", artifacts=[], url=self.page.url)

    async def extract_text(self, action: Action) -> BrowserActionResult:
        if action.selector:
            await self.page.wait_for_selector(action.selector, state="visible", timeout=self._sandbox.step_timeout_ms)
            text = await self.page.inner_text(action.selector)
        else:
            text = await self.page.evaluate("() => document.body.innerText.slice(0, 2000)")
        excerpt = text[:200]
        return BrowserActionResult(summary=f"Extracted text: {excerpt}", artifacts=[], url=self.page.url)

    async def screenshot(self, action: Action) -> BrowserActionResult:
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self._artifact_dir / "screenshot.png"
        await self.page.screenshot(path=str(path), full_page=True)
        return BrowserActionResult(summary="Captured screenshot", artifacts=[str(path)], url=self.page.url)

    async def evaluate_js(self, action: Action) -> BrowserActionResult:
        self._sandbox.validate_js()
        if not action.js:
            raise BrowserError("evaluate_js requires js")
        result = await self.page.evaluate(action.js)
        return BrowserActionResult(summary=f"Evaluated JS: {result!r}", artifacts=[], url=self.page.url)

    async def upload_file(self, action: Action) -> BrowserActionResult:
        self._sandbox.validate_file_upload()
        if not action.selector or not action.text:
            raise BrowserError("upload_file requires selector and text path")
        file_path = Path(action.text)
        if not file_path.exists():
            raise BrowserError(f"File not found: {file_path}")
        await self.page.set_input_files(action.selector, str(file_path))
        return BrowserActionResult(summary=f"Uploaded {file_path.name}", artifacts=[], url=self.page.url)

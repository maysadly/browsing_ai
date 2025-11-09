from __future__ import annotations

import asyncio
import logging
import re
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


_ROLE_PATTERN = re.compile(r"\[role=(?P<quote>['\"])(?P<value>.+?)(?P=quote)\]")
_TEXT_PATTERN = re.compile(r"\[text=(?P<quote>['\"])(?P<value>.+?)(?P=quote)\]")


@dataclass(slots=True)
class BrowserActionResult:
    """Structured summary of an executed action."""

    summary: str
    artifacts: list[str]
    url: str | None


class BrowserController:
    """High-level Playwright wrapper with sandbox enforcement."""

    def __init__(
        self,
        sandbox: SandboxPolicy,
        artifact_dir: Path,
        headless: bool = True,
        user_data_dir: Path | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._artifact_dir = artifact_dir
        self._headless = headless
        self._user_data_dir = user_data_dir.expanduser() if user_data_dir else None
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
        self._playwright = playwright

        if self._user_data_dir is not None:
            self._user_data_dir.mkdir(parents=True, exist_ok=True)
            context = await playwright.chromium.launch_persistent_context(
                str(self._user_data_dir),
                headless=self._headless,
            )
            page = context.pages[0] if context.pages else await context.new_page()
            browser = context.browser
        else:
            browser = await playwright.chromium.launch(headless=self._headless)
            context = await browser.new_context()
            page = await context.new_page()

        self._browser = browser
        self._context = context
        self._page = page
        self._attach_page_listeners(page)
        context.on("page", self._handle_new_page)

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
            "await_user_login": self._noop,
            "switch_tab": self.switch_tab,
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
        try:
            self._sandbox.validate_navigation(action.url)
        except SandboxViolation as exc:
            logger.info("Navigation blocked by sandbox for %s: %s", action.url, exc)
            return BrowserActionResult(
                summary=f"Navigation blocked by sandbox: {exc}",
                artifacts=[],
                url=self.page.url,
            )
        await self.page.goto(action.url, wait_until="domcontentloaded", timeout=self._sandbox.step_timeout_ms)
        await self.ensure_ready()
        return BrowserActionResult(summary=f"Opened {action.url}", artifacts=[], url=self.page.url)

    async def click(self, action: Action) -> BrowserActionResult:
        if not action.selector:
            raise BrowserError("click requires selector")
        candidates = self._selector_variants(action.selector)
        last_error: PlaywrightError | None = None
        for candidate in candidates:
            logger.debug("Attempting click selector %s", candidate)
            try:
                base_locator = self.page.locator(candidate)
                # Restrict to visible matches so we skip hidden duplicates (e.g. cookie/push buttons).
                locator = base_locator.locator(":visible").first
                if await locator.count() == 0:
                    raise PlaywrightError(f"No visible matches for {candidate}")
                await locator.wait_for(state="visible", timeout=self._sandbox.step_timeout_ms)
                await locator.click(timeout=self._sandbox.step_timeout_ms)
                await self.ensure_ready()
                return BrowserActionResult(summary=f"Clicked {candidate}", artifacts=[], url=self.page.url)
            except PlaywrightError as exc:
                last_error = exc
                continue

        if last_error is None:
            raise BrowserError(f"No valid selector variants generated for {action.selector}")

        message = str(last_error)
        if "No visible matches" in message:
            logger.info("Click target %s not found; treating as already dismissed", action.selector)
            return BrowserActionResult(
                summary=f"No visible matches for {action.selector}; assuming it is already dismissed",
                artifacts=[],
                url=self.page.url,
            )
        if "intercepts pointer events" in message:
            logger.info("Click target %s blocked by overlay", action.selector)
            overlay_summary = await self._report_modal_state()
            details = "no click performed"
            if overlay_summary:
                details += f"; detected overlay {overlay_summary}"
            return BrowserActionResult(
                summary=(
                    f"Click on {action.selector} blocked by another element intercepting pointer events; "
                    f"{details}"
                ),
                artifacts=[],
                url=self.page.url,
            )

        raise BrowserError(f"Failed to click {action.selector}: {last_error}")

    async def type_text(self, action: Action) -> BrowserActionResult:
        if not action.selector or action.text is None:
            raise BrowserError("type requires selector and text")

        locator = self.page.locator(action.selector).first
        await locator.wait_for(state="visible", timeout=self._sandbox.step_timeout_ms)
        await self.page.fill(action.selector, action.text, timeout=self._sandbox.step_timeout_ms)

        summary = f"Typed in {action.selector}"
        if await self._should_auto_submit(locator):
            try:
                await locator.press("Enter", timeout=self._sandbox.step_timeout_ms)
                summary += " and auto-submitted with Enter"
                logger.info("Auto-submitted search field %s with Enter", action.selector)
            except PlaywrightError:
                logger.debug("Auto-submit Enter failed for %s", action.selector, exc_info=True)

        return BrowserActionResult(summary=summary, artifacts=[], url=self.page.url)

    async def switch_tab(self, action: Action) -> BrowserActionResult:
        if self._context is None:
            raise BrowserError("Browser context not initialised")

        pages = [p for p in self._context.pages if not p.is_closed()]
        if not pages:
            raise BrowserError("No open tabs available")

        target: Page | None = None

        if action.tab_index is not None:
            if 0 <= action.tab_index < len(pages):
                target = pages[action.tab_index]
            else:
                raise BrowserError(f"Tab index {action.tab_index} out of range (available: {len(pages)})")

        if target is None and action.tab_url_contains:
            hint = action.tab_url_contains.lower()
            for candidate in reversed(pages):
                try:
                    url = candidate.url or ""
                except PlaywrightError:
                    url = ""
                if hint in url.lower():
                    target = candidate
                    break

        if target is None and action.tab_title_contains:
            hint = action.tab_title_contains.lower()
            for candidate in reversed(pages):
                try:
                    title = await candidate.title()
                except PlaywrightError:
                    title = ""
                if hint in title.lower():
                    target = candidate
                    break

        if target is None:
            raise BrowserError("Unable to identify target tab")

        await target.bring_to_front()
        self._page = target
        await self.ensure_ready()

        index = pages.index(target)
        try:
            current_url = target.url
        except PlaywrightError:
            current_url = None
        summary = f"Switched to tab {index}"
        if current_url:
            summary += f" ({current_url})"
        return BrowserActionResult(summary=summary, artifacts=[], url=current_url)

    async def press(self, action: Action) -> BrowserActionResult:
        if not action.selector or not action.keys:
            raise BrowserError("press requires selector and keys")
        await self.page.focus(action.selector)
        await self.page.keyboard.press(action.keys)
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

    def _selector_variants(self, selector: str) -> list[str]:
        variants: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            if candidate and candidate not in seen:
                variants.append(candidate)
                seen.add(candidate)

        add(selector)
        normalised = self._normalise_selector(selector)
        add(normalised)

        text_value = self._extract_attribute(normalised, _TEXT_PATTERN)
        if text_value:
            escaped = text_value.replace('"', '\\"')
            add(f'text="{escaped}"')
            add(f'a:has-text("{escaped}")')

        return variants

    def _normalise_selector(self, selector: str) -> str:
        cleaned = selector.strip()
        cleaned = _ROLE_PATTERN.sub(lambda m: self._rewrite_attribute(m, "role"), cleaned)
        cleaned = _TEXT_PATTERN.sub(lambda m: self._rewrite_attribute(m, "text"), cleaned)
        cleaned = cleaned.replace("\u00a0", " ")
        return cleaned

    async def _should_auto_submit(self, locator) -> bool:
        keywords = (
            "search",
            "query",
            "find",
            "lookup",
            "поиск",
            "найти",
            "искать",
        )

        try:
            metadata = await locator.evaluate(
                "(node) => ({\n"
                "  tag: node.tagName ? node.tagName.toLowerCase() : '',\n"
                "  type: node.type || '',\n"
                "  name: node.name || '',\n"
                "  placeholder: node.placeholder || '',\n"
                "  ariaLabel: node.getAttribute('aria-label') || '',\n"
                "  dataQa: node.getAttribute('data-qa') || '',\n"
                "  role: node.getAttribute('role') || '',\n"
                "})"
            )
        except PlaywrightError:
            logger.debug("Unable to inspect input metadata for auto-submit check", exc_info=True)
            return False

        if not isinstance(metadata, dict):
            return False

        tag = str(metadata.get("tag", "")).lower()
        if tag not in {"input", "textarea"}:
            return False

        input_type = str(metadata.get("type", "")).lower()
        if input_type in {"password", "email", "tel", "number"}:
            return False

        values = [
            str(metadata.get("name", "")),
            str(metadata.get("placeholder", "")),
            str(metadata.get("ariaLabel", "")),
            str(metadata.get("dataQa", "")),
            str(metadata.get("role", "")),
            input_type,
        ]
        lowered = " ".join(values).lower()
        return any(keyword in lowered for keyword in keywords)

    async def _report_modal_state(self) -> str | None:
        try:
            result = await self.page.evaluate(
                """
                () => {
                    const overlaySelectors = [
                        '[role=dialog]',
                        '.modal',
                        '.popup',
                        '.ReactModal__Content',
                        '.overlay',
                        '.modal-dialog',
                        '.dialog',
                    ];
                    const candidates = overlaySelectors.flatMap((selector) =>
                        Array.from(document.querySelectorAll(selector))
                    );
                    const visible = candidates.filter((node) => {
                        const style = window.getComputedStyle(node);
                        const rect = node.getBoundingClientRect();
                        if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) === 0) {
                            return false;
                        }
                        return rect.width > 0 && rect.height > 0;
                    });
                    if (visible.length === 0) {
                        return null;
                    }
                    const overlay = visible[0];
                    const rect = overlay.getBoundingClientRect();
                    const text = (overlay.innerText || overlay.textContent || '').trim().slice(0, 160);
                    const titleAttr = overlay.getAttribute('aria-label') || overlay.getAttribute('data-qa') || '';
                    return {
                        selector: overlay.className || overlay.id || overlay.tagName,
                        text,
                        title: titleAttr,
                        rect: { width: rect.width, height: rect.height },
                    };
                }
                """
            )
        except PlaywrightError:
            logger.debug("Overlay inspection failed", exc_info=True)
            return None

        if not isinstance(result, dict):
            return None

        selector = str(result.get("selector", "overlay")).strip() or "overlay"
        text = str(result.get("text", "")).strip()
        title = str(result.get("title", "")).strip()
        rect = result.get("rect") or {}
        width = rect.get("width")
        height = rect.get("height")
        size_part = ""
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            size_part = f" size={width:.0f}x{height:.0f}"
        label_parts = [selector]
        if title:
            label_parts.append(f'title="{title}"')
        if text:
            label_parts.append(f'text="{text}"')
        label = "; ".join(label_parts)
        return label + size_part

    def _handle_new_page(self, page: Page) -> None:
        logger.info("New tab opened; switching context")
        self._attach_page_listeners(page)
        self._page = page
        loop = asyncio.get_running_loop()
        loop.create_task(self._prepare_new_page(page))

    async def _prepare_new_page(self, page: Page) -> None:
        try:
            await page.bring_to_front()
            await page.wait_for_load_state("domcontentloaded", timeout=self._sandbox.step_timeout_ms)
        except PlaywrightError:
            logger.debug("Failed to prepare new tab", exc_info=True)

    def _attach_page_listeners(self, page: Page) -> None:
        page.on("close", lambda _: self._handle_page_close(page))

    def _handle_page_close(self, page: Page) -> None:
        if self._page is not None and self._page == page:
            fallback = self._select_fallback_page()
            if fallback is not None:
                self._page = fallback

    def _select_fallback_page(self) -> Page | None:
        if self._context is None:
            return None
        for candidate in reversed(self._context.pages):
            if not candidate.is_closed():
                return candidate
        return None

    @staticmethod
    def _rewrite_attribute(match: re.Match[str], attribute: str) -> str:
        value = match.group("value").replace('"', '\\"')
        return f"[{attribute}=\"{value}\"]"

    @staticmethod
    def _extract_attribute(selector: str, pattern: re.Pattern[str]) -> str | None:
        match = pattern.search(selector)
        if match is None:
            return None
        return match.group("value")

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger("autobrowser.tools")
from playwright.sync_api import Page, TimeoutError, Locator

from autobrowser.core.schemas import Action, Observation


class BrowserToolExecutor:
    """Executes high level actions on a Playwright page."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    def execute(
        self,
        page: Page,
        task_id: str,
        step_number: int,
        action: Action,
    ) -> Observation:
        logger.info(
            "Browser tools: execute",
            action=action.type,
            params=action.params,
            task_id=task_id,
            step=step_number,
        )
        try:
            result = self._invoke_handler(page, action)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Browser tools: handler raised",
                action=action.type,
                params=action.params,
            )
            return Observation(
                ok=False,
                note=str(exc),
                affordances=[],
                text_chunks=[],
                screenshot_path=self._capture_screenshot(
                    page, task_id, step_number, suffix="handler-error"
                ),
                error_category="HandlerException",
            )
        if result is None:
            return Observation(
                ok=False,
                note=f"Unsupported action: {action.type}",
                affordances=[],
                text_chunks=[],
                screenshot_path=None,
                error_category="UnsupportedAction",
            )

        try:
            note, extra_chunks = result
            screenshot_path = self._capture_screenshot(page, task_id, step_number)
            try:
                page_title = page.title()
            except Exception:  # noqa: BLE001
                page_title = None
            logger.info(
                "Browser tools: action complete",
                action=action.type,
                note=note,
                screenshot=screenshot_path,
            )
            return Observation(
                ok=True,
                note=note,
                affordances=[],
                text_chunks=extra_chunks,
                screenshot_path=screenshot_path,
                page_url=page.url,
                page_title=page_title,
            )
        except TimeoutError as exc:
            return Observation(
                ok=False,
                note=str(exc),
                affordances=[],
                text_chunks=[],
                screenshot_path=self._capture_screenshot(page, task_id, step_number, suffix="timeout"),
                error_category="Timeout",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Browser action failed: %s", action.type)
            return Observation(
                ok=False,
                note=str(exc),
                affordances=[],
                text_chunks=[],
                screenshot_path=self._capture_screenshot(page, task_id, step_number, suffix="error"),
                error_category="Exception",
            )

    def _invoke_handler(self, page: Page, action: Action) -> Optional[tuple[str, List[str]]]:
        handler = getattr(self, f"_handle_{action.type}", None)
        if handler is None:
            return None
        logger.debug("Browser tools: invoking handler", action=action.type)
        note = handler(page, action.params)
        if isinstance(note, tuple):
            return note
        return note, []

    def _handle_navigate(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        url = params.get("url")
        if not url:
            raise ValueError("Navigate action requires 'url' parameter.")
        wait_until = params.get("wait_until", "load")
        logger.info("Browser tools: navigating", url=url, wait_until=wait_until)
        page.goto(url, wait_until=wait_until, timeout=params.get("timeout", 45000))
        return f"Navigated to {url}", []

    def _handle_query(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        selector = params.get("locator") or params.get("selector")
        if not selector:
            raise ValueError("Query action requires 'selector' or 'locator' parameter.")
        locator = page.locator(selector)
        results = locator.all_text_contents()
        return (
            f"Found {len(results)} elements for selector '{selector}'.",
            results[:10],
        )

    def _handle_click(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        locator = self._resolve_locator(page, params)
        locator.click(timeout=params.get("timeout", 30000))
        return f"Clicked {params.get('locator', params.get('text', 'unknown'))}", []

    def _handle_type(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        text = params.get("text", "")
        locator = self._resolve_locator(page, params)
        timeout = params.get("timeout", 30000)
        if params.get("clear", True):
            locator.fill(text, timeout=timeout)
        else:
            locator.type(text, delay=params.get("delay", 20), timeout=timeout)
        return f"Typed '{text[:20]}'", []

    def _handle_select(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        locator = self._resolve_locator(page, params)
        option = params.get("value") or params.get("label")
        locator.select_option(value=option)
        return f"Selected option {option}", []

    def _handle_wait(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        wait_for = params.get("for", "networkidle")
        timeout = params.get("timeout", 30000)
        if wait_for == "networkidle":
            try:
                page.wait_for_load_state("networkidle", timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "Browser tools: networkidle wait timed out, falling back to fixed delay",
                    timeout=timeout,
                )
                page.wait_for_timeout(min(timeout, 3000))
        elif wait_for == "load":
            page.wait_for_load_state("load", timeout=timeout)
        elif wait_for == "selector":
            selector = params["selector"]
            page.wait_for_selector(selector, timeout=timeout)
        elif wait_for == "timeout":
            time.sleep(params.get("seconds", 1))
        return f"Waited for {wait_for}", []

    def _handle_scroll(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        delta = params.get("delta", 1000)
        page.mouse.wheel(0, int(delta))
        return f"Scrolled by {delta}", []

    def _handle_extract(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        selector = params.get("locator") or params.get("selector")
        if not selector:
            raise ValueError("Extract action requires 'locator' or 'selector' parameter.")
        locator = page.locator(selector)
        texts = locator.all_text_contents()
        return f"Extracted {len(texts)} elements from {selector}", texts[:50]

    def _handle_confirm_gate(self, page: Page, params: Dict[str, Any]) -> tuple[str, List[str]]:
        return "Awaiting confirmation.", []

    def _resolve_locator(self, page: Page, params: Dict[str, Any]) -> Locator:
        nth = params.get("nth")
        if "locator" in params:
            locator = page.locator(params["locator"])
            if nth is not None:
                return locator.nth(int(nth))
            count = locator.count()
            if count > 1:
                logger.warning(
                    "Locator matched multiple elements, defaulting to first",
                    selector=params["locator"],
                    matches=count,
                )
                return locator.first
            return locator
        if "text" in params:
            locator = page.get_by_text(params["text"])
            if nth is not None:
                return locator.nth(int(nth))
            count = locator.count()
            if count > 1:
                logger.warning(
                    "Text locator matched multiple elements, defaulting to first",
                    text=params["text"],
                    matches=count,
                )
                return locator.first
            return locator
        raise ValueError("Locator or text required for this action.")

    def _capture_screenshot(
        self,
        page: Page,
        task_id: str,
        step_number: int,
        *,
        suffix: str | None = None,
    ) -> Optional[str]:
        try:
            artifact_dir = self.artifacts_dir / task_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            filename = f"step_{step_number:03d}"
            if suffix:
                filename += f"_{suffix}"
            filename += ".png"
            path = artifact_dir / filename
            page.screenshot(path=path)
            return str(path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to capture screenshot.")
            return None

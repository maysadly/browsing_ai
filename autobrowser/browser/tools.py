from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger("autobrowser.tools")
from playwright.sync_api import Page, TimeoutError, Locator

from autobrowser.core.schemas import Action, Observation


@dataclass(slots=True)
class ToolResult:
    """Normalized return type for browser tool handlers."""

    note: str
    text_chunks: List[str]


@dataclass(slots=True)
class ToolContext:
    """Rich context passed to each tool handler."""

    page: Page
    action: Action
    task_id: str
    step_number: int

    @property
    def params(self) -> Dict[str, Any]:
        return self.action.params or {}

    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def int_param(self, name: str, default: int) -> int:
        value = self.param(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def bool_param(self, name: str, default: bool = True) -> bool:
        value = self.param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return default


class BrowserActionError(Exception):
    """Structured error used to signal recoverable action failures."""

    def __init__(self, message: str, *, category: str = "HandlerException") -> None:
        super().__init__(message)
        self.category = category


class BrowserToolExecutor:
    """Executes high level actions on a Playwright page."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self._default_locator_timeout_ms = 2000
        self._scroll_timeout_ms = 5000
        self._handlers: Dict[str, Callable[[ToolContext], ToolResult]] = {
            "navigate": self._handle_navigate,
            "query": self._handle_query,
            "click": self._handle_click,
            "type": self._handle_type,
            "select": self._handle_select,
            "wait": self._handle_wait,
            "scroll": self._handle_scroll,
            "extract": self._handle_extract,
            "confirm_gate": self._handle_confirm_gate,
            "submit": self._handle_submit,
        }

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
        handler = self._handlers.get(action.type)
        if handler is None:
            return Observation(
                ok=False,
                note=f"Unsupported action: {action.type}",
                affordances=[],
                text_chunks=[],
                screenshot_path=None,
                error_category="UnsupportedAction",
            )

        ctx = ToolContext(page=page, action=action, task_id=task_id, step_number=step_number)
        param_summary = self._summarize_params(ctx.params)
        try:
            result = handler(ctx)
            tool_result = self._normalize_result(result)
            screenshot_path = self._capture_screenshot(page, task_id, step_number)
            try:
                page_title = page.title()
            except Exception:  # noqa: BLE001
                page_title = None
            logger.info(
                f"Browser tools: action complete [{action.type}] params={param_summary} note={tool_result.note}"
            )
            return Observation(
                ok=True,
                note=tool_result.note,
                affordances=[],
                text_chunks=tool_result.text_chunks,
                screenshot_path=screenshot_path,
                page_url=page.url,
                page_title=page_title,
            )
        except BrowserActionError as exc:
            logger.warning(
                f"Browser tools: recoverable error [{action.type}] params={param_summary} "
                f"category={exc.category} error={exc}"
            )
            return Observation(
                ok=False,
                note=f"{action.type} failed (params={param_summary}): {exc}",
                affordances=[],
                text_chunks=[],
                screenshot_path=self._capture_screenshot(page, task_id, step_number, suffix="handler-error"),
                error_category=exc.category,
            )
        except TimeoutError as exc:
            logger.warning(
                f"Browser tools: timeout [{action.type}] params={param_summary} error={exc}"
            )
            return Observation(
                ok=False,
                note=f"{action.type} timeout (params={param_summary}): {exc}",
                affordances=[],
                text_chunks=[],
                screenshot_path=self._capture_screenshot(page, task_id, step_number, suffix="timeout"),
                error_category="Timeout",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                f"Browser tools: unhandled exception [{action.type}] params={param_summary} error={exc}"
            )
            return Observation(
                ok=False,
                note=f"{action.type} error (params={param_summary}): {exc}",
                affordances=[],
                text_chunks=[],
                screenshot_path=self._capture_screenshot(page, task_id, step_number, suffix="error"),
                error_category="Exception",
            )

    def _normalize_result(self, result: ToolResult | tuple[str, List[str]] | str) -> ToolResult:
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, tuple):
            note, chunks = result
            return ToolResult(note=note, text_chunks=list(chunks))
        return ToolResult(note=str(result), text_chunks=[])

    # ------------------------------------------------------------------ Handlers

    def _handle_navigate(self, ctx: ToolContext) -> ToolResult:
        url = ctx.param("url")
        if not url:
            raise BrowserActionError("Navigate action requires 'url' parameter.", category="InvalidParameters")
        wait_until = ctx.param("wait_until", "load")
        timeout = ctx.int_param("timeout", 45000)
        logger.info("Browser tools: navigating", url=url, wait_until=wait_until)
        ctx.page.goto(url, wait_until=wait_until, timeout=timeout)
        return ToolResult(f"Navigated to {url}", [])

    def _handle_query(self, ctx: ToolContext) -> ToolResult:
        selector = ctx.param("locator") or ctx.param("selector")
        if not selector:
            raise BrowserActionError("Query action requires 'selector' or 'locator' parameter.", category="InvalidParameters")
        locator = ctx.page.locator(selector)
        results = locator.all_text_contents()
        return ToolResult(
            note=f"Found {len(results)} elements for selector '{selector}'.",
            text_chunks=results[:10],
        )

    def _handle_click(self, ctx: ToolContext) -> ToolResult:
        params = ctx.params
        target = self._describe_target(params)
        attempts = self._build_click_attempts(params)
        backoff = float(ctx.param("retry_backoff", 0.35))
        param_brief = _safe_shorten_dict(params)

        last_error: Optional[Exception] = None
        for attempt_index, (click_kwargs, suffix) in enumerate(attempts, start=1):
            try:
                locator = self._resolve_locator(ctx)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Browser tools: click failed to resolve locator",
                    target=target,
                    params=param_brief,
                    error=str(exc),
                )
                break

            wait_timeout = min(
                click_kwargs.get("timeout", 0) or self._default_locator_timeout_ms,
                self._default_locator_timeout_ms,
            )
            self._maybe_wait_locator(locator, wait_timeout)

            try:
                if hasattr(locator, "scroll_into_view_if_needed"):
                    locator.scroll_into_view_if_needed(timeout=self._scroll_timeout_ms)
            except Exception:  # noqa: BLE001
                logger.debug("Browser tools: scroll into view skipped", action="click")

            try:
                locator.click(**click_kwargs)
                return ToolResult(note=f"Clicked {target}{suffix}", text_chunks=[])
            except TimeoutError as exc:
                last_error = exc
                logger.debug(
                    f"Browser tools: click timeout target={target} attempt={attempt_index} "
                    f"force={click_kwargs.get('force')} params={param_brief} error={exc}"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.debug(
                    f"Browser tools: click error target={target} attempt={attempt_index} "
                    f"force={click_kwargs.get('force')} params={param_brief} error={exc}"
                )
            time.sleep(backoff * attempt_index)

        if last_error:
            category = "Timeout" if isinstance(last_error, TimeoutError) else "HandlerException"
            raise BrowserActionError(
                f"Click on {target or 'element'} failed after {len(attempts)} attempts (params={param_brief}): {last_error}",
                category=category,
            ) from last_error
        return ToolResult(note=f"Clicked {target}", text_chunks=[])

    def _handle_type(self, ctx: ToolContext) -> ToolResult:
        params = ctx.params
        text = params.get("text", "")
        locator = self._resolve_locator(ctx)
        timeout = ctx.int_param("timeout", 30000)
        input_type = (locator.get_attribute("type") or "").lower()
        force = ctx.bool_param("force", False)

        if not locator:
            raise BrowserActionError("Type action received empty locator after resolution.", category="InvalidParameters")

        if input_type == "hidden":
            raise BrowserActionError("Hidden input cannot be filled directly.", category="HiddenInput")

        if input_type in {"radio", "checkbox"}:
            desired_value = params.get("value")
            if desired_value:
                selector = params.get("locator") or params.get("selector")
                if selector:
                    candidate = ctx.page.locator(f"{selector}[value='{desired_value}']")
                    if candidate.count() > 0:
                        locator = candidate
            if input_type == "radio":
                locator.check(timeout=timeout, force=force)
                descriptor = desired_value or locator.first.get_attribute("value") or "option"
                return ToolResult(f"Selected radio {descriptor}", [])
            target_checked = ctx.bool_param("checked", True)
            if target_checked:
                locator.check(timeout=timeout, force=force)
                return ToolResult("Checkbox checked", [])
            locator.uncheck(timeout=timeout, force=force)
            return ToolResult("Checkbox unchecked", [])

        if ctx.bool_param("clear", True):
            locator.fill(text, timeout=timeout)
        else:
            delay = ctx.int_param("delay", 20)
            locator.type(text, delay=delay, timeout=timeout)
        return ToolResult(f"Typed '{text[:20]}'", [])

    def _handle_select(self, ctx: ToolContext) -> ToolResult:
        locator = self._resolve_locator(ctx)
        option_value = ctx.param("value")
        option_label = ctx.param("label")
        if option_value is not None:
            locator.select_option(value=option_value)
            descriptor = option_value
        elif option_label is not None:
            locator.select_option(label=option_label)
            descriptor = option_label
        else:
            raise BrowserActionError("Select action requires 'value' or 'label' parameter.", category="InvalidParameters")
        return ToolResult(f"Selected option {descriptor}", [])

    def _handle_wait(self, ctx: ToolContext) -> ToolResult:
        wait_for = ctx.param("for", "networkidle")
        timeout = ctx.int_param("timeout", 30000)
        page = ctx.page
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
            selector = ctx.param("selector")
            if not selector:
                raise BrowserActionError("Wait for selector requires 'selector' parameter.", category="InvalidParameters")
            page.wait_for_selector(selector, timeout=timeout)
        elif wait_for == "timeout":
            time.sleep(ctx.int_param("seconds", 1))
        return ToolResult(f"Waited for {wait_for}", [])

    def _handle_scroll(self, ctx: ToolContext) -> ToolResult:
        page = ctx.page
        params = ctx.params
        selector = params.get("selector")
        if selector:
            page.locator(selector).scroll_into_view_if_needed(timeout=ctx.int_param("timeout", self._scroll_timeout_ms))
            return ToolResult(f"Scrolled to {selector}", [])

        direction = params.get("direction")
        if direction in {"top", "bottom"}:
            target = "0" if direction == "top" else "document.body.scrollHeight"
            page.evaluate(f"window.scrollTo(0, {target});")
            return ToolResult(f"Scrolled to {direction}", [])
        if direction in {"up", "down"}:
            magnitude = ctx.int_param("amount", 1200)
            delta = -magnitude if direction == "up" else magnitude
            page.mouse.wheel(0, delta)
            return ToolResult(f"Scrolled {direction} by {magnitude}", [])

        delta = ctx.int_param("delta", 1000)
        page.mouse.wheel(0, delta)
        return ToolResult(f"Scrolled by {delta}", [])

    def _handle_extract(self, ctx: ToolContext) -> ToolResult:
        page = ctx.page
        params = ctx.params
        selector = params.get("locator") or params.get("selector")
        if not selector:
            raise BrowserActionError("Extract action requires 'locator' or 'selector' parameter.", category="InvalidParameters")
        locator = page.locator(selector)
        texts = locator.all_text_contents()
        if not texts:
            fallback_param = (
                params.get("fallback_locators")
                or params.get("fallback_selector")
                or params.get("fallbacks")
                or []
            )
            if isinstance(fallback_param, str):
                normalized = fallback_param.replace("||", ",")
                fallback_selectors = [part.strip() for part in normalized.split(",") if part.strip()]
            else:
                fallback_selectors = list(fallback_param)
            fallback_selectors.extend(self._guess_extract_fallbacks(page.url, selector))

            tried_selectors = {selector}
            for fallback_selector in fallback_selectors:
                if not fallback_selector or fallback_selector in tried_selectors:
                    continue
                tried_selectors.add(fallback_selector)
                fallback_locator = page.locator(fallback_selector)
                texts = fallback_locator.all_text_contents()
                if texts:
                    logger.info(
                        "Browser tools: extract fallback selector matched",
                        original_selector=selector,
                        fallback_selector=fallback_selector,
                        matches=len(texts),
                    )
                    snippets = texts[:50]
                    snippets.insert(0, f"SELECTOR_TEMPLATE::{fallback_selector}")
                    return ToolResult(
                        note=f"Extracted {len(texts)} elements from {fallback_selector}",
                        text_chunks=snippets,
                    )
            raise BrowserActionError(f"No elements matched selector '{selector}'.")
        return ToolResult(f"Extracted {len(texts)} elements from {selector}", texts[:50])

    def _handle_confirm_gate(self, ctx: ToolContext) -> ToolResult:
        return ToolResult("Awaiting confirmation.", [])

    def _handle_submit(self, ctx: ToolContext) -> ToolResult:
        if ctx.bool_param("dry_run", True):
            return ToolResult("Dry-run submit executed; no side effects triggered.", [])
        raise BrowserActionError("Form submit requires confirmation and cannot be forced.", category="HighRisk")

    # ------------------------------------------------------------------ Helpers

    def _build_click_attempts(
        self,
        params: Dict[str, Any],
    ) -> List[tuple[Dict[str, Any], str]]:
        base_timeout_raw = params.get("timeout", 8000)
        try:
            base_timeout = int(base_timeout_raw)
        except (TypeError, ValueError):
            base_timeout = 8000

        force_timeout_raw = params.get("force_timeout", 5000)
        try:
            force_timeout = int(force_timeout_raw)
        except (TypeError, ValueError):
            force_timeout = 5000
        forced_timeout = min(base_timeout, force_timeout)

        base: Dict[str, Any] = {
            "timeout": base_timeout,
            "force": self._coerce_bool(params.get("force"), default=False),
        }
        if "button" in params:
            base["button"] = params["button"]
        if "click_count" in params:
            try:
                base["click_count"] = int(params["click_count"])
            except (TypeError, ValueError):
                pass
        if "delay" in params:
            try:
                base["delay"] = int(params["delay"])
            except (TypeError, ValueError):
                pass
        if "no_wait_after" in params:
            base["no_wait_after"] = self._coerce_bool(params["no_wait_after"], default=False)

        attempts: List[tuple[Dict[str, Any], str]] = [(base, " [force]" if base.get("force") else "")]
        if not base.get("force"):
            fallback = dict(base)
            fallback["force"] = True
            fallback["timeout"] = forced_timeout
            fallback["no_wait_after"] = True
            attempts.append((fallback, " [force]"))
        return attempts

    def _describe_target(self, params: Dict[str, Any]) -> str:
        for key in ("name", "label", "text", "locator", "selector", "role", "xpath"):
            value = params.get(key)
            if value:
                return str(value)
        return "element"

    def _coerce_bool(self, value: Any, *, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value).strip().lower()
        if text in {"false", "0", "no", "off"}:
            return False
        if text in {"true", "1", "yes", "on"}:
            return True
        return default

    def _resolve_locator(self, ctx: ToolContext) -> Locator:
        params = ctx.params
        page = ctx.page
        nth_raw = params.get("nth")
        try:
            nth = int(nth_raw) if nth_raw is not None else None
        except (TypeError, ValueError):
            nth = None
        exact = self._coerce_bool(params.get("exact"), default=True)
        timeout = ctx.int_param("locator_timeout", self._default_locator_timeout_ms)

        def finalize(candidate: Locator, *, source: str, raw: Optional[str] = None) -> tuple[Locator, int]:
            try:
                candidate = candidate.set_timeout(timeout)
            except Exception:  # noqa: BLE001
                pass
            if nth is not None:
                return candidate.nth(nth), 1
            count = self._safe_count(candidate)
            if count > 1:
                log_payload: Dict[str, Any] = {"source": source, "matches": count}
                if raw:
                    log_payload["hint"] = raw
                logger.warning(
                    "Locator resolved to multiple elements, defaulting to first",
                    **log_payload,
                )
                candidate = candidate.first
            return candidate, count

        role = params.get("role")
        if role:
            name = params.get("name") or params.get("label") or params.get("text")
            role_kwargs: Dict[str, Any] = {}
            if name:
                role_kwargs["name"] = name
                role_kwargs["exact"] = exact
            locator = page.get_by_role(role=role, **role_kwargs)
            locator, count = finalize(locator, source="role", raw=name or role)
            if count == 0:
                raise BrowserActionError(f"No role='{role}' element matched the provided hints.", category="NotFound")
            return locator

        label = params.get("label")
        if label:
            locator = page.get_by_label(label, exact=exact)
            locator, count = finalize(locator, source="label", raw=str(label))
            if count == 0:
                raise BrowserActionError(f"No element with label '{label}' found.", category="NotFound")
            return locator

        placeholder = params.get("placeholder")
        if placeholder:
            locator = page.get_by_placeholder(placeholder, exact=exact)
            locator, count = finalize(locator, source="placeholder", raw=str(placeholder))
            if count == 0:
                raise BrowserActionError(f"No element with placeholder '{placeholder}' found.", category="NotFound")
            return locator

        alt_text = params.get("alt")
        if alt_text:
            locator = page.get_by_alt_text(alt_text, exact=exact)
            locator, count = finalize(locator, source="alt", raw=str(alt_text))
            if count == 0:
                raise BrowserActionError(f"No element with alt text '{alt_text}' found.", category="NotFound")
            return locator

        title = params.get("title")
        if title:
            locator = page.get_by_title(title, exact=exact)
            locator, count = finalize(locator, source="title", raw=str(title))
            if count == 0:
                raise BrowserActionError(f"No element with title '{title}' found.", category="NotFound")
            return locator

        locator_str = params.get("locator") or params.get("selector")
        if locator_str:
            selector = locator_str.strip()
            if selector.startswith("xpath=") or selector.startswith("//"):
                target = selector if selector.startswith("xpath=") else f"xpath={selector}"
                locator = page.locator(target)
                locator, count = finalize(locator, source="xpath", raw=selector)
                if count == 0:
                    raise BrowserActionError(f"No elements matched xpath '{selector}'.", category="NotFound")
                return locator
            locator = page.locator(selector)
            locator, count = finalize(locator, source="locator", raw=selector)
            if count == 0:
                raise BrowserActionError(f"No elements matched selector '{selector}'.", category="NotFound")
            return locator

        text = params.get("text")
        if text:
            locator = page.get_by_text(text, exact=exact)
            locator, count = finalize(locator, source="text", raw=str(text))
            if count == 0 and exact:
                fuzzy_locator = page.get_by_text(text, exact=False)
                fuzzy_locator, fuzzy_count = finalize(fuzzy_locator, source="text_fuzzy", raw=str(text))
                if fuzzy_count > 0:
                    logger.info(
                        "Locator text exact match failed; falling back to substring search",
                        text=text,
                        matches=fuzzy_count,
                    )
                    locator, count = fuzzy_locator, fuzzy_count
            if count == 0:
                raise BrowserActionError(f"No element containing text '{text}' found.", category="NotFound")
            return locator

        xpath = params.get("xpath")
        if xpath:
            target = xpath if str(xpath).startswith("xpath=") else f"xpath={xpath}"
            locator = page.locator(target)
            locator, count = finalize(locator, source="xpath", raw=str(xpath))
            if count == 0:
                raise BrowserActionError(f"No elements matched xpath '{xpath}'.", category="NotFound")
            return locator

        raise BrowserActionError(
            "Locator parameters required (role, label, text, locator/selector, or xpath).",
            category="InvalidParameters",
        )

    def _maybe_wait_locator(self, locator: Locator, timeout: int) -> None:
        try:
            locator.wait_for(state="attached", timeout=timeout)
        except TimeoutError:
            logger.debug("Browser tools: locator attach wait timed out; continuing")
        except Exception:  # noqa: BLE001
            logger.debug("Browser tools: locator wait skipped due to error")

    def _safe_count(self, locator: Locator) -> int:
        try:
            return locator.count()
        except Exception:  # noqa: BLE001
            return 0

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

    def _guess_extract_fallbacks(self, url: str, selector: str) -> List[str]:
        """Provide domain-specific fallbacks when primary extract selector fails."""
        fallbacks: List[str] = []
        hostname = urlparse(url).hostname or ""

        if "indeed." in hostname:
            # Indeed's markup fluctuates; job cards commonly use these selectors.
            fallbacks.extend(
                [
                    "div.job_seen_beacon",
                    "td.resultContent",
                    "a[data-jk]",
                ]
            )

        if "hh.ru" in hostname:
            # hh.ru often renders vacancies as article cards with data-qa attributes.
            fallbacks.extend(
                [
                    "article[data-qa='vacancy-serp__vacancy']",
                    "a[data-qa='serp__vacancy-title']",
                ]
            )

        # Preserve original order but remove the primary selector if present.
        return [fb for fb in fallbacks if fb and fb != selector]

    def _summarize_params(self, params: Dict[str, Any]) -> str:
        if not params:
            return "{}"
        try:
            serialized = json.dumps(params, ensure_ascii=False, default=str)
        except Exception:
            serialized = str(params)
        if len(serialized) > 200:
            serialized = serialized[:197] + "..."
        return serialized


def _safe_shorten_dict(params: Dict[str, Any]) -> str:
    try:
        data = json.dumps(params, ensure_ascii=False, default=str)
    except Exception:
        data = str(params)
    if len(data) > 120:
        data = data[:117] + "..."
    return data

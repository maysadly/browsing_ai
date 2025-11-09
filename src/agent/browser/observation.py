from __future__ import annotations

import logging
import asyncio
from typing import cast
from pathlib import Path

from playwright.async_api import Error as PlaywrightError, Page

from ..errors import ObservationError
from ..types import (
    InteractiveElement,
    Observation,
    ModalSnapshot,
    OverlaySnapshot,
    ScrollMetrics,
    TabSnapshot,
)

logger = logging.getLogger(__name__)


class ObservationCollector:
    """Collect structured state about the current page."""

    def __init__(self, artifact_dir: Path, capture_screenshots: bool = False) -> None:
        self._artifact_dir = artifact_dir
        self._capture_screenshots = capture_screenshots

    async def collect(self, page: Page, step_index: int) -> Observation:
        try:
            url = page.url
            try:
                title = await page.title()
            except PlaywrightError:
                logger.warning("Falling back to empty title due to transient Playwright error", exc_info=True)
                title = ""
        except PlaywrightError as exc:  # pragma: no cover - rare browser issue
            raise ObservationError("Unable to read page metadata") from exc

        visible_text = await self._evaluate_with_retry(
            page,
            "() => document.body.innerText.slice(0, 4000)",
            description="visible_text",
            default="",
        )
        page_state = await self._evaluate_with_retry(
            page,
            r"""
            () => {
                const doc = document.scrollingElement || document.documentElement || document.body;
                const scrollTop = doc ? (doc.scrollTop ?? window.scrollY ?? 0) : 0;
                const scrollHeight = doc ? (doc.scrollHeight ?? document.body.scrollHeight ?? 0) : 0;
                const clientHeight = window.innerHeight ?? (doc ? doc.clientHeight : 0) ?? 0;
                const maxScrollable = Math.max(scrollHeight - clientHeight, 1);
                const progress = maxScrollable <= 0 ? 1 : Math.min(Math.max(scrollTop / maxScrollable, 0), 1);

                const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
                const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;

                const isVisible = (node) => {
                    if (!node || !node.isConnected) {
                        return false;
                    }
                    if (node.hasAttribute('hidden')) {
                        return false;
                    }
                    const ariaHidden = node.getAttribute('aria-hidden');
                    if (ariaHidden && ariaHidden.toLowerCase() === 'true') {
                        return false;
                    }
                    if (node.tagName === 'INPUT' && node.type && node.type.toLowerCase() === 'hidden') {
                        return false;
                    }
                    const style = window.getComputedStyle(node);
                    if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) === 0) {
                        return false;
                    }
                    const rect = node.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) {
                        return false;
                    }
                    const tolerance = Math.max(40, viewportHeight * 0.1);
                    if (rect.bottom < -tolerance || rect.top > viewportHeight + tolerance) {
                        return false;
                    }
                    return true;
                };

                const buildSelector = (node) => {
                    const selectorParts = [];
                    if (node.id) selectorParts.push(`#${node.id}`);
                    if (node.name) selectorParts.push(`[name="${node.name}"]`);
                    if (!selectorParts.length) {
                        const dataQa = node.getAttribute('data-qa');
                        if (dataQa) {
                            const tag = node.tagName.toLowerCase();
                            selectorParts.push(`${tag}[data-qa="${dataQa}"]`);
                        }
                    }
                    if (!selectorParts.length) {
                        const classes = (node.className || '').trim().split(/\s+/).filter(Boolean);
                        if (classes.length) selectorParts.push('.' + classes.slice(0, 2).join('.'));
                    }
                    if (!selectorParts.length) {
                        const ariaLabel = node.getAttribute('aria-label');
                        if (ariaLabel) {
                            const safe = ariaLabel.trim().replace(/"/g, '\\"');
                            selectorParts.push(`[aria-label="${safe}"]`);
                        }
                    }
                    if (!selectorParts.length) {
                        const placeholder = node.getAttribute('placeholder');
                        if (placeholder) {
                            const safe = placeholder.trim().replace(/"/g, '\\"');
                            selectorParts.push(`[placeholder="${safe}"]`);
                        }
                    }
                    if (!selectorParts.length) {
                        const dataTestId = node.getAttribute('data-testid');
                        if (dataTestId) {
                            const tag = node.tagName.toLowerCase();
                            selectorParts.push(`${tag}[data-testid="${dataTestId}"]`);
                        }
                    }
                    if (!selectorParts.length) {
                        const href = node.getAttribute('href');
                        if (href) {
                            try {
                                const absolute = new URL(href, window.location.href);
                                const relative = absolute.pathname + absolute.search + absolute.hash;
                                selectorParts.push(`${node.tagName.toLowerCase()}[href="${relative}"]`);
                            } catch (err) {
                                selectorParts.push(`${node.tagName.toLowerCase()}[href="${href}"]`);
                            }
                        }
                    }
                    if (!selectorParts.length) {
                        const textContent = (node.innerText || node.value || '').trim();
                        if (textContent) {
                            const safeText = textContent.slice(0, 80).replace(/"/g, '\\"');
                            selectorParts.push(`text="${safeText}"`);
                        }
                    }
                    return selectorParts.length ? selectorParts.join('') : null;
                };

                const serialiseInteractive = (node) => {
                    const rect = node.getBoundingClientRect();
                    const textContent = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('placeholder') || '').replace(/\s+/g, ' ').trim();
                    const truncatedText = textContent.slice(0, 160);
                    const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
                    const visibilityRatio = rect.height > 0 ? Math.max(0, Math.min(visibleHeight / rect.height, 1)) : 0;
                    return {
                        tag: node.tagName.toLowerCase(),
                        role: node.getAttribute('role') || node.tagName.toLowerCase(),
                        text: truncatedText || null,
                        selector: buildSelector(node),
                        attributes: {
                            id: node.id || null,
                            name: node.name || null,
                            type: node.type || null,
                            aria_label: node.getAttribute('aria-label') || null,
                            placeholder: node.getAttribute('placeholder') || null,
                            href: node.href || null,
                            value: typeof node.value === 'string' ? node.value : null,
                        },
                        viewport_top: rect.top,
                        viewport_bottom: rect.bottom,
                        viewport_left: rect.left,
                        viewport_right: rect.right,
                        viewport_visible_ratio: visibilityRatio,
                    };
                };

                const overlaySelectors = [
                    '[role=dialog]',
                    '.modal',
                    '.popup',
                    '.ReactModal__Content',
                    '.overlay',
                    '.modal-dialog',
                    '.dialog',
                ];
                const overlays = overlaySelectors
                    .flatMap((selector) => Array.from(document.querySelectorAll(selector)))
                    .filter((node) => isVisible(node))
                    .slice(0, 3)
                    .map((node) => {
                        const rect = node.getBoundingClientRect();
                        return {
                            selector: node.id ? `#${node.id}` : node.className || node.tagName,
                            text: (node.innerText || node.textContent || '').trim().slice(0, 200) || null,
                            title: node.getAttribute('aria-label') || node.getAttribute('data-qa') || null,
                            width: rect.width,
                            height: rect.height,
                        };
                    });

                const modalSelectors = [
                    '[data-qa="relocation-warning"]',
                    '.magritte-modal',
                    '.magritte-modal___RAW6S_9-4-1',
                    '.magritte-modal___cWCe7_9-4-1',
                    '.HH-Supernova-NotificationManager-Container',
                    '.supernova-notification',
                    '.HH-Notifications-Root',
                    '.HH-Supernova-NotificationManager-Container-Portal',
                    '.ReactModal__Content',
                    '[role="dialog"]',
                    '.modal',
                    '.popup'
                ];

                const modalCandidates = [];
                const seen = new Set();
                for (const selector of modalSelectors) {
                    for (const node of Array.from(document.querySelectorAll(selector))) {
                        if (!node || seen.has(node)) continue;
                        seen.add(node);
                        if (!isVisible(node)) continue;
                        const rect = node.getBoundingClientRect();
                        if (rect.width < 200 && rect.height < 140) continue;
                        modalCandidates.push({ node, rect });
                    }
                }
                modalCandidates.sort((a, b) => (b.rect.width * b.rect.height) - (a.rect.width * a.rect.height));

                let modal = null;
                if (modalCandidates.length) {
                    const { node, rect } = modalCandidates[0];
                    const totalArea = Math.max(viewportHeight * viewportWidth, 1);
                    const areaRatio = (rect.width * rect.height) / totalArea;
                    if (areaRatio >= 0.04 || rect.height >= viewportHeight * 0.25) {
                        const actionNodes = Array.from(node.querySelectorAll('button, a[href], [role="button"], input[type="submit"], input[type="button"], input[type="reset"]'))
                            .filter((candidate) => isVisible(candidate))
                            .slice(0, 12);
                        const actions = actionNodes.map(serialiseInteractive);

                        const fieldNodes = Array.from(node.querySelectorAll('input, textarea, select'))
                            .filter((candidate) => isVisible(candidate))
                            .slice(0, 12);
                        const fields = fieldNodes.map(serialiseInteractive);

                        const titleNode = node.querySelector('[data-qa*="title"], h1, h2, h3, header, .magritte-title, .HH-Heading, .magritte-card-title');
                        const labelAttr = node.getAttribute('aria-label') || node.getAttribute('data-qa') || '';
                        const titleText = titleNode ? titleNode.innerText : labelAttr;
                        const textual = (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();

                        const selectorLabel = (() => {
                            if (node.id) return `#${node.id}`;
                            const dataQa = node.getAttribute('data-qa');
                            if (dataQa) {
                                return `${node.tagName.toLowerCase()}[data-qa="${dataQa}"]`;
                            }
                            const className = (node.className || '').trim();
                            if (className) {
                                return '.' + className.split(/\s+/).slice(0, 2).join('.');
                            }
                            return node.tagName;
                        })();

                        modal = {
                            selector: selectorLabel,
                            title: (titleText || '').trim().slice(0, 160) || null,
                            text: textual.slice(0, 600) || null,
                            width: rect.width,
                            height: rect.height,
                            top: rect.top,
                            right: rect.right,
                            bottom: rect.bottom,
                            left: rect.left,
                            actions,
                            fields,
                        };
                    }
                }

                return {
                    scrollTop,
                    scrollHeight,
                    clientHeight,
                    progress,
                    overlays,
                    modal,
                };
            }
            """,
            description="page_state",
            default=None,
        )
        interactive_elements = await self._scrape_elements(page)
        screenshot_path = await self._capture_screenshot(page, step_index)

        overlays_raw: list[dict] = []
        scroll_payload = None
        modal_snapshot: ModalSnapshot | None = None
        if isinstance(page_state, dict):
            overlays_raw = page_state.get("overlays") or []
            if {"scrollTop", "scrollHeight", "clientHeight", "progress"}.issubset(page_state.keys()):
                scroll_payload = {
                    "scrollTop": page_state.get("scrollTop"),
                    "scrollHeight": page_state.get("scrollHeight"),
                    "clientHeight": page_state.get("clientHeight"),
                    "progress": page_state.get("progress"),
                }
            modal_raw = page_state.get("modal")
            if isinstance(modal_raw, dict):
                actions_raw = modal_raw.get("actions") or []
                fields_raw = modal_raw.get("fields") or []
                actions = []
                fields = []
                for item in actions_raw:
                    if isinstance(item, dict):
                        action = InteractiveElement.model_validate(item)
                        action.under_modal = True
                        actions.append(action)
                for item in fields_raw:
                    if isinstance(item, dict):
                        field = InteractiveElement.model_validate(item)
                        field.under_modal = True
                        fields.append(field)
                selector_value = modal_raw.get("selector")
                title_value = modal_raw.get("title")
                text_value = modal_raw.get("text")
                modal_snapshot = ModalSnapshot(
                    selector=(selector_value.strip() or None)
                    if isinstance(selector_value, str)
                    else None,
                    title=(title_value.strip() or None)
                    if isinstance(title_value, str)
                    else None,
                    text=(text_value.strip() or None)
                    if isinstance(text_value, str)
                    else None,
                    width=float(modal_raw["width"]) if isinstance(modal_raw.get("width"), (int, float)) else None,
                    height=float(modal_raw["height"]) if isinstance(modal_raw.get("height"), (int, float)) else None,
                    top=float(modal_raw["top"]) if isinstance(modal_raw.get("top"), (int, float)) else None,
                    right=float(modal_raw["right"]) if isinstance(modal_raw.get("right"), (int, float)) else None,
                    bottom=float(modal_raw["bottom"]) if isinstance(modal_raw.get("bottom"), (int, float)) else None,
                    left=float(modal_raw["left"]) if isinstance(modal_raw.get("left"), (int, float)) else None,
                    actions=actions,
                    fields=fields,
                )

        modal_bounds = None
        if (
            modal_snapshot
            and modal_snapshot.left is not None
            and modal_snapshot.right is not None
            and modal_snapshot.top is not None
            and modal_snapshot.bottom is not None
        ):
            modal_bounds = (
                cast(float, modal_snapshot.left),
                cast(float, modal_snapshot.right),
                cast(float, modal_snapshot.top),
                cast(float, modal_snapshot.bottom),
            )

        if modal_bounds:
            left, right, top, bottom = modal_bounds
            for element in interactive_elements:
                if (
                    element.viewport_left is None
                    or element.viewport_right is None
                    or element.viewport_top is None
                    or element.viewport_bottom is None
                ):
                    continue
                center_x = (element.viewport_left + element.viewport_right) / 2
                center_y = (element.viewport_top + element.viewport_bottom) / 2
                if left <= center_x <= right and top <= center_y <= bottom:
                    element.under_modal = True

        scroll_metrics = ScrollMetrics.model_validate(scroll_payload) if scroll_payload else None
        overlay_snapshots = [
            OverlaySnapshot(
                selector=(str(value).strip() or None)
                if (value := item.get("selector")) is not None
                else None,
                text=(str(value).strip() or None)
                if (value := item.get("text")) is not None
                else None,
                title=(str(value).strip() or None)
                if (value := item.get("title")) is not None
                else None,
                width=float(value)
                if isinstance((value := item.get("width")), (int, float))
                else None,
                height=float(value)
                if isinstance((value := item.get("height")), (int, float))
                else None,
            )
            for item in overlays_raw
            if isinstance(item, dict)
        ]
        open_tabs = await self._collect_tabs(page)

        if modal_snapshot:
            modal_text_parts = []
            if modal_snapshot.title:
                modal_text_parts.append(modal_snapshot.title)
            if modal_snapshot.text:
                modal_text_parts.append(modal_snapshot.text)
            modal_block = "\n".join(modal_text_parts).strip()
            background_excerpt = visible_text[:600]
            if modal_block:
                visible_text = f"[MODAL]\n{modal_block}\n\n[PAGE]\n{background_excerpt}"
            else:
                visible_text = f"[MODAL]\n{modal_snapshot.selector or 'dialog'}\n\n[PAGE]\n{background_excerpt}"

        return Observation(
            url=url,
            title=title,
            visible_text=visible_text,
            interactive_elements=interactive_elements,
            errors=[],
            page_size=self._get_viewport(page),
            network_idle=False,
            screenshot_path=screenshot_path,
            overlays=overlay_snapshots,
            scroll=scroll_metrics,
            open_tabs=open_tabs,
            modal=modal_snapshot,
        )

    async def _scrape_elements(self, page: Page) -> list[InteractiveElement]:
        script = r"""
        () => {
            const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
            const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
            const viewportPadding = Math.max(80, viewportHeight * 0.15);

            const isVisible = (node) => {
                if (!node || !node.isConnected) {
                    return false;
                }
                if (node.hasAttribute('hidden')) {
                    return false;
                }
                const ariaHidden = node.getAttribute('aria-hidden');
                if (ariaHidden && ariaHidden.toLowerCase() === 'true') {
                    return false;
                }
                if (node.tagName === 'INPUT' && node.type && node.type.toLowerCase() === 'hidden') {
                    return false;
                }
                const style = window.getComputedStyle(node);
                if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) === 0) {
                    return false;
                }
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) {
                    return false;
                }
                const outOfViewport = rect.bottom < -viewportPadding || rect.top > viewportHeight + viewportPadding;
                if (outOfViewport) {
                    return false;
                }
                return true;
            };

            const shouldSkip = (node) => {
                const dataQa = (node.getAttribute('data-qa') || '').toLowerCase();
                if (dataQa && dataQa.startsWith('a11y-')) {
                    return true;
                }
                const hrefAttr = (node.getAttribute('href') || '').toLowerCase();
                if (hrefAttr.includes('#a11y-')) {
                    return true;
                }
                return false;
            };

            const rawNodes = Array.from(document.querySelectorAll('a, button, input, select, textarea, summary'))
                .map((node) => ({ node, rect: node.getBoundingClientRect() }))
                .filter(({ node }) => isVisible(node) && !shouldSkip(node));

            rawNodes.sort((a, b) => {
                const topDelta = a.rect.top - b.rect.top;
                if (Math.abs(topDelta) > 4) {
                    return topDelta;
                }
                return a.rect.left - b.rect.left;
            });

            const nodes = rawNodes.slice(0, 80);

            return nodes.map(({ node, rect }) => {
                const role = node.getAttribute('role') || node.tagName.toLowerCase();
                const selectorParts = [];
                if (node.id) selectorParts.push(`#${node.id}`);
                if (node.name) selectorParts.push(`[name="${node.name}"]`);
                if (!selectorParts.length) {
                    const dataQa = node.getAttribute('data-qa');
                    if (dataQa) {
                        const tag = node.tagName.toLowerCase();
                        selectorParts.push(`${tag}[data-qa="${dataQa}"]`);
                    }
                }
                if (!selectorParts.length) {
                    const classes = (node.className || '').trim().split(/\s+/).filter(Boolean);
                    if (classes.length) selectorParts.push('.' + classes.slice(0, 2).join('.'));
                }
                const ariaLabel = node.getAttribute('aria-label') || '';
                const placeholder = node.getAttribute('placeholder') || '';
                const fallbackText = ariaLabel || placeholder;
                const textContent = (node.innerText || node.value || fallbackText || '').replace(/\s+/g, ' ').trim();
                const truncatedText = textContent.slice(0, 160);
                let selector = selectorParts.length ? selectorParts.join('') : null;
                if (!selector) {
                    const href = node.getAttribute('href');
                    if (href) {
                        try {
                            const absolute = new URL(href, window.location.href);
                            const relative = absolute.pathname + absolute.search + absolute.hash;
                            selector = `a[href="${relative}"]`;
                        } catch (err) {
                            selector = `a[href="${href}"]`;
                        }
                    }
                }
                if (!selector && truncatedText) {
                    const safeText = truncatedText.replace(/"/g, '\\"');
                    selector = `text="${safeText}"`;
                }
                const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
                const visibilityRatio = rect.height > 0 ? Math.max(0, Math.min(visibleHeight / rect.height, 1)) : 0;

                return {
                    tag: node.tagName.toLowerCase(),
                    role,
                    text: truncatedText,
                    selector,
                    attributes: {
                        id: node.id || null,
                        name: node.name || null,
                        type: node.type || null,
                        aria_label: ariaLabel || null,
                        placeholder: placeholder || null,
                        href: node.href || null,
                        value: typeof node.value === 'string' ? node.value : null,
                    },
                    viewport_top: rect.top,
                    viewport_bottom: rect.bottom,
                    viewport_left: rect.left,
                    viewport_right: rect.right,
                    viewport_visible_ratio: visibilityRatio,
                };
            });
        }
        """
        raw_elements = await self._evaluate_with_retry(
            page,
            script,
            description="interactive_elements",
            default=[],
        )
        return [InteractiveElement.model_validate(item) for item in raw_elements]

    async def _collect_tabs(self, page: Page) -> list[TabSnapshot]:
        try:
            context = page.context
        except PlaywrightError:
            return []

        tabs: list[TabSnapshot] = []
        for index, candidate in enumerate(context.pages):
            if candidate.is_closed():
                continue
            try:
                title = await candidate.title()
            except PlaywrightError:
                title = ""
            try:
                url = candidate.url
            except PlaywrightError:
                url = ""
            tabs.append(
                TabSnapshot(
                    index=index,
                    url=url or None,
                    title=title or None,
                    is_active=candidate == page,
                )
            )
        return tabs

    async def _capture_screenshot(self, page: Page, step_index: int) -> str | None:
        if not self._capture_screenshots:
            return None
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self._artifact_dir / f"step_{step_index:03d}_observation.png"
        try:
            await page.screenshot(path=str(path), full_page=True)
        except PlaywrightError:  # pragma: no cover - screenshot failures
            logger.warning("Screenshot capture failed", exc_info=True)
            return None
        return str(path)

    def _get_viewport(self, page: Page) -> tuple[int, int] | None:
        size = page.viewport_size
        if size is None:
            return None
        return size["width"], size["height"]

    async def _evaluate_with_retry(
        self,
        page: Page,
        expression: str,
        *,
        description: str,
        default,
        attempts: int = 3,
    ):
        last_error: PlaywrightError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await page.evaluate(expression)
            except PlaywrightError as exc:
                last_error = exc
                message = str(exc)
                transient_navigation = "Execution context was destroyed" in message or "context was destroyed" in message
                if transient_navigation and attempt < attempts:
                    logger.debug(
                        "Evaluation for %s failed due to navigation (attempt %d/%d); waiting for DOMContentLoaded",
                        description,
                        attempt,
                        attempts,
                    )
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=2_000)
                    except PlaywrightError:
                        logger.debug("Load state wait after evaluation failure also failed")
                    await asyncio.sleep(0.2)
                    continue
                logger.warning("Evaluation for %s failed: %s", description, message)
                break

        if last_error is not None:
            logger.warning("Falling back to default for %s due to evaluation failure", description, exc_info=last_error)
        return default

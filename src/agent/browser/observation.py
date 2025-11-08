from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import Error as PlaywrightError, Page

from ..errors import ObservationError
from ..types import InteractiveElement, Observation

logger = logging.getLogger(__name__)


class ObservationCollector:
    """Collect structured state about the current page."""

    def __init__(self, artifact_dir: Path, capture_screenshots: bool = False) -> None:
        self._artifact_dir = artifact_dir
        self._capture_screenshots = capture_screenshots

    async def collect(self, page: Page, step_index: int) -> Observation:
        try:
            url = page.url
            title = await page.title()
        except PlaywrightError as exc:  # pragma: no cover - rare browser issue
            raise ObservationError("Unable to read page metadata") from exc

        visible_text = await page.evaluate("() => document.body.innerText.slice(0, 4000)")
        interactive_elements = await self._scrape_elements(page)
        screenshot_path = await self._capture_screenshot(page, step_index)

        return Observation(
            url=url,
            title=title,
            visible_text=visible_text,
            interactive_elements=interactive_elements,
            errors=[],
            page_size=self._get_viewport(page),
            network_idle=False,
            screenshot_path=screenshot_path,
        )

    async def _scrape_elements(self, page: Page) -> list[InteractiveElement]:
        script = """
        () => {
            const nodes = Array.from(
                document.querySelectorAll('a, button, input, select, textarea, summary')
            ).slice(0, 50);
            return nodes.map((node) => {
                const role = node.getAttribute('role') || node.tagName.toLowerCase();
                const selectorParts = [];
                if (node.id) selectorParts.push(`#${node.id}`);
                if (node.name) selectorParts.push(`[name="${node.name}"]`);
                if (!selectorParts.length) {
                    const classes = (node.className || '').trim().split(/\\s+/).filter(Boolean);
                    if (classes.length) selectorParts.push('.' + classes.slice(0, 2).join('.'));
                }
                const selector = selectorParts.length ? selectorParts.join('') : null;
                const text = (node.innerText || node.value || '').trim().slice(0, 160);
                return {
                    tag: node.tagName.toLowerCase(),
                    role,
                    text,
                    selector,
                    attributes: {
                        id: node.id || null,
                        name: node.name || null,
                        type: node.type || null,
                        aria_label: node.getAttribute('aria-label') || null,
                        href: node.href || null,
                    }
                };
            });
        }
        """
        raw_elements = await page.evaluate(script)
        return [InteractiveElement.model_validate(item) for item in raw_elements]

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

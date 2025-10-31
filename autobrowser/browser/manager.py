from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from loguru import logger
from playwright.sync_api import BrowserContext, sync_playwright

if TYPE_CHECKING:  # pragma: no cover
    from autobrowser.core.config import Settings
else:
    Settings = Any  # type: ignore[misc]

try:
    from autobrowser.core.config import get_settings
except ModuleNotFoundError:  # pragma: no cover
    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")


class BrowserManager:
    """Controls Playwright persistent contexts for tasks."""

    def __init__(self, settings: Optional["Settings"] = None) -> None:
        self._settings = settings or get_settings()
        self._handles: Dict[str, Tuple[Any, BrowserContext]] = {}
        self._lock = Lock()

    def start(self) -> None:
        logger.info("Playwright started")

    def acquire_context(self, task_id: str) -> BrowserContext:
        with self._lock:
            if task_id in self._handles:
                logger.info("Reusing existing context", task_id=task_id)
                return self._handles[task_id][1]

            if len(self._handles) >= self._settings.max_concurrent_tasks:
                raise RuntimeError("No browser slots available.")

            session_dir = Path(self._settings.session_dir) / task_id
            session_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Launching persistent context", task_id=task_id, user_data_dir=str(session_dir))
            playwright = sync_playwright().start()
            try:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(session_dir),
                    headless=False,
                    viewport={"width": 1280, "height": 900},
                    args=["--disable-dev-shm-usage"],
                )
            except Exception:
                logger.exception("Failed to launch persistent context", task_id=task_id)
                playwright.stop()
                raise
            logger.info(
                "Persistent context launched",
                task_id=task_id,
                session_dir=str(session_dir),
                pages=len(context.pages),
            )
            self._handles[task_id] = (playwright, context)
            return context

    def new_page(self, task_id: str):
        logger.info("Requesting page for task", task_id=task_id)
        context = self.acquire_context(task_id)
        pages = context.pages
        if pages:
            logger.info("Reusing existing page", task_id=task_id)
            return pages[0]
        page = context.new_page()
        logger.debug("Opened new page for task", task_id=task_id, pages=len(context.pages))
        return page

    def close_task(self, task_id: str) -> None:
        with self._lock:
            context = self._contexts.pop(task_id, None)
            if context:
                context.close()
        logger.info("Context closed", task_id=task_id)

    def shutdown(self) -> None:
        with self._lock:
            for task_id, (playwright, context) in list(self._handles.items()):
                try:
                    context.close()
                finally:
                    playwright.stop()
                logger.info("Playwright stopped", task_id=task_id)
            self._handles.clear()

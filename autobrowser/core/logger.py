from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger("autobrowser")

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings
from .schemas import TaskLogEvent

_LOGGING_CONFIGURED = False


def configure_logging(settings: "Settings") -> None:
    """Configure loguru sinks once."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(settings.logs_dir) / "autobrowser.jsonl"

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_path,
        level="DEBUG",
        rotation="10 MB",
        serialize=True,
        enqueue=True,
    )

    _LOGGING_CONFIGURED = True


def log_event(event: TaskLogEvent, extra: Dict[str, Any] | None = None) -> None:
    """Log structured task events to the JSON sink."""

    payload = {
        "task_id": event.task_id,
        "kind": event.kind,
        "timestamp": event.timestamp.isoformat(),
        "payload": event.payload,
        "event_id": event.id,
    }
    if extra:
        payload.update(extra)

    if hasattr(logger, "bind"):
        logger.bind(task_id=event.task_id).info("task_event", **payload)
    else:  # pragma: no cover
        logger.info("task_event %s", payload)

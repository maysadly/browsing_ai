from __future__ import annotations

import contextvars
import logging
import sys
from pathlib import Path
from typing import Any

import orjson

_run_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("run_context", default={})


class ORJSONFormatter(logging.Formatter):
    """Structured JSON log formatter using orjson."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - fmt
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        payload.update(_run_context.get({}))
        for key, value in record.__dict__.items():
            if key in {"msg", "args", "levelname", "levelno", "pathname", "filename", "module"}:
                continue
            if key.startswith("_"):
                continue
            if key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return orjson.dumps(payload, default=str).decode()


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logger with JSON formatter."""

    root = logging.getLogger()
    root.setLevel(log_level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = ORJSONFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return module-level logger."""

    return logging.getLogger(name)


def set_run_context(**kwargs: Any) -> None:
    """Attach contextual metadata to subsequent log records."""

    _run_context.set(dict(kwargs))

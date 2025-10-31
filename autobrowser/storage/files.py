from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from autobrowser.core.config import Settings
else:
    Settings = Any  # type: ignore[misc]

try:
    from autobrowser.core.config import get_settings
except ModuleNotFoundError:  # pragma: no cover
    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")


class FileStorage:
    """Handles file-system artefacts such as screenshots and task logs."""

    def __init__(self, settings: "Settings" | None = None) -> None:
        self._settings = settings or get_settings()
        Path(self._settings.artifacts_dir).mkdir(parents=True, exist_ok=True)

    def task_artifact_dir(self, task_id: str) -> Path:
        path = Path(self._settings.artifacts_dir) / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, task_id: str, name: str, payload: Dict[str, Any]) -> Path:
        directory = self.task_artifact_dir(task_id)
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

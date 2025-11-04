from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pytest


def _load_tasks(env_var: str, limit: int | None = None) -> List[Dict[str, str]]:
    value = os.getenv(env_var)
    if not value:
        pytest.skip(
            f"Environment variable {env_var} is not set; "
            "provide a Mind2Web scenario file (JSON/JSONL).",
            allow_module_level=True,
        )
    path = Path(value).expanduser()
    if not path.exists():
        pytest.skip(f"Mind2Web scenario file {path} not found.", allow_module_level=True)

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("tasks") or payload.get("episodes") or []
        records = list(payload)

    if limit:
        records = records[:limit]
    if not records:
        pytest.skip(f"No Mind2Web scenarios available in {path}.", allow_module_level=True)
    return records


def _scenario_ids(items: Iterable[Dict[str, str]]) -> List[str]:
    ids: List[str] = []
    for idx, item in enumerate(items, start=1):
        ids.append(str(item.get("id") or item.get("goal") or f"mind2web_{idx}"))
    return ids


SCENARIOS = _load_tasks("MIND2WEB_TASKS", limit=int(os.getenv("MIND2WEB_TASK_LIMIT", "5")))


@pytest.mark.mind2web
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_ids(SCENARIOS))
def test_mind2web_scenario_loaded(scenario: Dict[str, str]) -> None:
    """
    Smoke-test to validate Mind2Web scenarios can be enumerated.

    The full evaluation pipeline consumes these scenarios via the orchestrator.
    """

    assert scenario.get("goal"), "Scenario must include a textual goal."
    assert scenario.get("seed_url") or scenario.get("start_url"), "Scenario must define a starting URL."


from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pytest


def _load_task_payloads(env_var: str, limit: int | None = None) -> List[Dict[str, str]]:
    path_value = os.getenv(env_var)
    if not path_value:
        pytest.skip(
            f"Environment variable {env_var} is not set; "
            "set it to a JSON/JSONL file containing benchmark scenarios.",
            allow_module_level=True,
        )

    path = Path(path_value).expanduser()
    if not path.exists():
        pytest.skip(f"Benchmark file {path} does not exist.", allow_module_level=True)

    records: List[Dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("tasks") or data.get("scenarios") or []
        records = list(data)

    if limit:
        records = records[:limit]
    if not records:
        pytest.skip(f"No scenarios loaded from {path}.", allow_module_level=True)
    return records


def _scenario_ids(items: Iterable[Dict[str, str]]) -> List[str]:
    identifiers: List[str] = []
    for idx, item in enumerate(items, start=1):
        scenario_id = item.get("id") or item.get("name") or item.get("goal") or f"scenario_{idx}"
        identifiers.append(str(scenario_id))
    return identifiers


SCENARIOS = _load_task_payloads("WEBARENA_TASKS", limit=int(os.getenv("WEBARENA_TASK_LIMIT", "5")))


@pytest.mark.webarena
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_ids(SCENARIOS))
def test_webarena_scenario_loaded(scenario: Dict[str, str]) -> None:
    """
    Smoke-test to ensure WebArena scenarios can be discovered.

    End-to-end execution is handled by the dedicated evaluation harness.
    """

    assert "goal" in scenario, "Scenario must include a user goal."
    assert scenario["goal"], "Scenario goal must be non-empty."
    assert "start_url" in scenario, "Scenario must define a starting URL."


from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import Action, AgentStep, Observation, PlanItem


@dataclass(slots=True)
class TraceRecorder:
    """Persist observations, actions, and outcomes for a run."""

    run_id: str
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def step_dir(self, index: int) -> Path:
        path = self.root_dir / f"step_{index:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_observation(self, index: int, observation: Observation) -> Path:
        path = self.step_dir(index) / "observation.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(observation.model_dump(mode="json"), file, indent=2)
        return path

    def record_action(self, index: int, action: Action) -> Path:
        path = self.step_dir(index) / "action.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(action.model_dump(mode="json"), file, indent=2)
        return path

    def record_result(self, index: int, result: dict[str, Any]) -> Path:
        path = self.step_dir(index) / "result.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        return path

    def record_plan(self, plan: list[PlanItem]) -> Path:
        path = self.root_dir / "plan.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump([item.model_dump(mode="json") for item in plan], file, indent=2)
        return path

    def to_agent_step(
        self,
        index: int,
        observation: Observation,
        action: Action,
        summary: str,
        artifacts: list[str],
    ) -> AgentStep:
        return AgentStep(
            step_index=index,
            observation=observation,
            action=action,
            result_summary=summary,
            artifacts=artifacts,
        )

    @staticmethod
    def new_run_dir(base_dir: Path, prefix: str = "run") -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = base_dir / f"{prefix}_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

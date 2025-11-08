from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from ..browser.controller import BrowserController
from ..browser.observation import ObservationCollector
from ..browser.sandbox import SandboxPolicy
from ..config import Settings
from ..errors import BrowserError, LLMError, ParsingError, SandboxViolation
from ..logging import set_run_context
from ..types import Action, AgentStep, Observation, RunResult
from .evaluator import EvaluationResult, StepEvaluator
from .memory import AgentMemory
from .planner import Planner
from .trace import TraceRecorder

logger = logging.getLogger(__name__)


class Agent:
    """Main orchestration loop coordinating planner, browser, and evaluator."""

    def __init__(
        self,
        planner: Planner,
        settings: Settings,
        sandbox: SandboxPolicy,
        runs_dir: Path,
        capture_screenshots: bool = True,
    ) -> None:
        self._planner = planner
        self._settings = settings
        self._sandbox = sandbox
        self._runs_dir = runs_dir
        self._capture_screenshots = capture_screenshots
        self._memory = AgentMemory()
        self._evaluator = StepEvaluator()

    async def run(self, task: str, start_url: str | None = None, headless: bool | None = None) -> RunResult:
        run_id = str(uuid.uuid4())
        run_dir = TraceRecorder.new_run_dir(self._runs_dir, prefix=f"run_{run_id}")
        set_run_context(run_id=run_id)
        headless = self._settings.headless_default if headless is None else headless

        collector = ObservationCollector(run_dir, capture_screenshots=self._capture_screenshots)
        trace = TraceRecorder(run_id=run_id, root_dir=run_dir)
        steps: list[AgentStep] = []
        previous_excerpt: str | None = None

        async with BrowserController(self._sandbox, run_dir, headless=headless) as controller:
            if start_url:
                try:
                    await controller.open_url(Action(type="open_url", url=start_url))
                except (BrowserError, SandboxViolation) as exc:
                    logger.exception("Failed to open start url")
                    return RunResult(steps=[], status="failed", error=str(exc))

            observation = await collector.collect(controller.page, step_index=0)

            for step_index in range(1, self._sandbox.max_steps + 1):
                trace.record_observation(step_index, observation)

                sandbox_summary = (
                    f"Allowed hosts: {', '.join(sorted(self._sandbox.allowed_hosts))}; "
                    f"Steps remaining: {self._sandbox.max_steps - self._sandbox.steps_taken}"
                )
                memory_context = self._memory.compressed()

                try:
                    action = await self._planner.decide_action(
                        task=task,
                        observation=observation,
                        memory_context=memory_context,
                        sandbox_summary=sandbox_summary,
                    )
                except (LLMError, ParsingError) as exc:
                    logger.exception("Planner failure")
                    return RunResult(steps=steps, status="failed", error=str(exc))

                trace.record_action(step_index, action)

                if action.type not in {"finish", "abort"}:
                    self._sandbox.validate_action()

                try:
                    result = await controller.perform(action)
                except (BrowserError, SandboxViolation) as exc:
                    logger.exception("Browser execution failed")
                    return RunResult(steps=steps, status="failed", error=str(exc))

                self._sandbox.record_step()

                result_payload = {
                    "summary": result.summary,
                    "artifacts": result.artifacts,
                    "url": result.url,
                }
                trace.record_result(step_index, result_payload)

                next_observation = await collector.collect(controller.page, step_index=step_index)
                agent_step = trace.to_agent_step(
                    index=step_index,
                    observation=observation,
                    action=action,
                    summary=result.summary,
                    artifacts=result.artifacts,
                )
                steps.append(agent_step)
                self._memory.add(observation, action)

                evaluation = self._evaluator.evaluate(
                    action=action,
                    observation_before=observation,
                    observation_after=next_observation,
                    previous_excerpt=previous_excerpt,
                )

                if not evaluation.continue_run:
                    return RunResult(
                        steps=steps,
                        status=evaluation.status,
                        final_answer=evaluation.final_answer,
                        error=evaluation.error,
                    )

                if evaluation.replan:
                    await asyncio.sleep(0.5)

                observation = next_observation
                previous_excerpt = observation.visible_text[:200]

        return RunResult(
            steps=steps,
            status="failed",
            error="Maximum steps reached without completion",
        )

#!/usr/bin/env python3
"""
Summarise evaluation runs recorded in the JSONL trace.

Usage:

    python autobrowser/scripts/eval_summary.py --log storage/logs/autobrowser.jsonl

Outputs success@1, mean steps, and mean duration for each task suite detected.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class TaskRecord:
    suite: str
    goal: str
    success: bool
    steps: int
    duration_seconds: float


def parse_log(path: Path) -> Iterable[TaskRecord]:
    with path.open("r", encoding="utf-8") as handle:
        tasks: Dict[str, TaskRecord] = {}
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = payload.get("record", {})
            name = record.get("name")
            if name != "autobrowser.core.orchestrator":
                continue
            message = payload.get("record", {}).get("message")
            extra = record.get("extra", {})
            task_id = extra.get("task_id")
            if message == "Task created":
                goal = extra.get("goal", "")
                suite = extra.get("suite") or ("webarena" if "WebArena" in goal else "mind2web")
                tasks[task_id] = TaskRecord(
                    suite=suite or "unknown",
                    goal=goal,
                    success=False,
                    steps=0,
                    duration_seconds=0.0,
                )
            elif message == "Task completed successfully" and task_id in tasks:
                tasks[task_id].success = True
            elif message == "Task cleanup finished" and task_id in tasks:
                tasks[task_id].duration_seconds = extra.get("elapsed_seconds", tasks[task_id].duration_seconds)
            elif message == "Plan step completed" and task_id in tasks:
                tasks[task_id].steps += 1

        return tasks.values()


def summarise(tasks: Iterable[TaskRecord]) -> Dict[str, Dict[str, float]]:
    suites: Dict[str, List[TaskRecord]] = defaultdict(list)
    for task in tasks:
        suites[task.suite].append(task)

    summary: Dict[str, Dict[str, float]] = {}
    for suite, records in suites.items():
        if not records:
            continue
        success = sum(1 for r in records if r.success)
        steps = sum(r.steps for r in records)
        duration = sum(r.duration_seconds for r in records)
        summary[suite] = {
            "tasks": float(len(records)),
            "success@1": success / len(records),
            "avg_steps": steps / len(records) if records else 0.0,
            "avg_duration_s": duration / len(records) if records else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise evaluation metrics from JSONL logs.")
    parser.add_argument("--log", type=Path, required=True, help="Path to autobrowser JSONL log.")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log file {args.log} does not exist.")

    summary = summarise(parse_log(args.log))
    if not summary:
        print("No evaluation tasks found.")
        return

    print("Evaluation summary:")
    for suite, metrics in summary.items():
        print(f"- {suite}")
        print(f"  tasks: {int(metrics['tasks'])}")
        print(f"  success@1: {metrics['success@1']:.2f}")
        print(f"  avg_steps: {metrics['avg_steps']:.2f}")
        print(f"  avg_duration_s: {metrics['avg_duration_s']:.2f}")


if __name__ == "__main__":
    main()


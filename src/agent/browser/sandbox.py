from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..errors import SandboxViolation
from .tools import expand_allowlist, normalize_host


@dataclass(slots=True)
class SandboxPolicy:
    """Runtime guardrails enforcing navigation and resource limits."""

    allowed_hosts: set[str]
    max_steps: int
    step_timeout_s: int
    run_timeout_s: int
    allow_js_eval: bool
    allow_file_upload: bool
    started_at: float = field(default_factory=time.monotonic)
    steps_taken: int = 0

    @classmethod
    def from_hosts(
        cls,
        hosts: tuple[str, ...],
        max_steps: int,
        step_timeout_s: int,
        run_timeout_s: int,
        allow_js_eval: bool,
        allow_file_upload: bool,
    ) -> "SandboxPolicy":
        return cls(
            allowed_hosts=expand_allowlist(hosts),
            max_steps=max_steps,
            step_timeout_s=step_timeout_s,
            run_timeout_s=run_timeout_s,
            allow_js_eval=allow_js_eval,
            allow_file_upload=allow_file_upload,
        )

    @property
    def step_timeout_ms(self) -> int:
        return self.step_timeout_s * 1000

    def validate_navigation(self, url: str) -> None:
        host = normalize_host(url)
        if host not in self.allowed_hosts:
            raise SandboxViolation(f"Navigation to {host} is not allowed")
        self._check_run_time()

    def validate_action(self) -> None:
        self._check_run_time()

    def validate_js(self) -> None:
        if not self.allow_js_eval:
            raise SandboxViolation("JavaScript evaluation disabled by policy")

    def validate_file_upload(self) -> None:
        if not self.allow_file_upload:
            raise SandboxViolation("File upload disabled by policy")

    def record_step(self) -> None:
        self.steps_taken += 1
        if self.steps_taken > self.max_steps:
            raise SandboxViolation("Maximum steps exceeded")
        self._check_run_time()

    def _check_run_time(self) -> None:
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.run_timeout_s:
            raise SandboxViolation("Run timeout exceeded")

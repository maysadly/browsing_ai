from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Action, Observation


@dataclass(slots=True)
class MemoryItem:
    observation_summary: str
    action_summary: str


@dataclass(slots=True)
class AgentMemory:
    """Keeps a bounded history of observations and actions."""

    max_items: int = 10
    items: list[MemoryItem] = field(default_factory=list)

    def add(self, observation: Observation, action: Action) -> None:
        item = MemoryItem(
            observation_summary=observation.summary(),
            action_summary=f"{action.type} selector={action.selector} url={action.url}",
        )
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items.pop(0)

    def compressed(self) -> str:
        if not self.items:
            return ""
        segments: list[str] = []
        for idx, item in enumerate(self.items[-self.max_items :], start=1):
            segments.append(
                f"[Memory {idx}] Observation:\n{item.observation_summary}\nAction: {item.action_summary}"
            )
        return "\n".join(segments)

from __future__ import annotations

from typing import Iterable, List

from autobrowser.core.schemas import Affordance


def from_affordance(affordance: Affordance) -> List[str]:
    """Generate resilient selectors from an affordance description."""

    selectors: List[str] = []
    if affordance.locator:
        selectors.append(affordance.locator)
    if affordance.name:
        text = affordance.name.replace('"', "").strip()
        selectors.extend(
            [
                f'role={affordance.role} name="{text}"' if affordance.role else "",
                f"text={text}",
                f"button:has-text('{text}')" if affordance.role == "button" else "",
                f"a:has-text('{text}')" if affordance.role == "link" else "",
            ]
        )
    return [selector for selector in selectors if selector]


def basic_text_selectors(text: str) -> List[str]:
    text = text.replace('"', "").strip()
    if not text:
        return []
    return [
        f"text={text}",
        f"button:has-text('{text}')",
        f"[aria-label='{text}']",
        f"[placeholder='{text}']",
    ]


def combine_strategies(labels: Iterable[str]) -> List[str]:
    selectors: List[str] = []
    for label in labels:
        selectors.extend(basic_text_selectors(label))
    return selectors

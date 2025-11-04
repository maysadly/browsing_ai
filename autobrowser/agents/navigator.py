from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from loguru import logger

from autobrowser.core.schemas import (
    Affordance,
    DistilledNode,
    NavigationHint,
    PageStatePack,
)


def _normalize_keywords(keywords: Iterable[str] | None) -> List[str]:
    tokens: List[str] = []
    if not keywords:
        return tokens
    for keyword in keywords:
        if not keyword:
            continue
        parts = [part for part in keyword.lower().split() if len(part) > 2]
        tokens.extend(parts)
    return tokens


@dataclass
class NavigatorConfig:
    max_candidates: int = 18
    distilled_weight: float = 1.2
    text_weight: float = 0.8
    role_weight: float = 1.5
    keyword_weight: float = 3.0


class Navigator:
    """Ranks page elements (affordances + distilled DOM) for navigation decisions."""

    def __init__(self, config: Optional[NavigatorConfig] = None) -> None:
        self._config = config or NavigatorConfig()

    @property
    def max_candidates(self) -> int:
        return self._config.max_candidates

    # ------------------------------------------------------------------ public API

    def collect_candidates(
        self,
        page_state: PageStatePack,
        *,
        keywords: Sequence[str] | None = None,
    ) -> List[NavigationHint]:
        keyword_tokens = _normalize_keywords(keywords)
        affordance_hints = self._score_affordances(page_state.affordances, keyword_tokens)
        distilled_hints = self._score_distilled_nodes(page_state.distilled_elements, keyword_tokens)

        combined = sorted(
            affordance_hints + distilled_hints,
            key=lambda hint: hint.score,
            reverse=True,
        )
        top = combined[: self._config.max_candidates]
        logger.debug(
            "Navigator: collected candidates",
            affordances=len(affordance_hints),
            distilled=len(distilled_hints),
            selected=len(top),
        )
        return top

    def best_locator(
        self,
        page_state: PageStatePack,
        *,
        keywords: Sequence[str] | None = None,
        roles: Optional[set[str]] = None,
    ) -> Optional[str]:
        candidates = self.collect_candidates(page_state, keywords=keywords)
        for hint in candidates:
            if roles and hint.extra.get("role") not in roles:
                continue
            if hint.locator:
                return hint.locator
        return None

    def rank_affordances(
        self,
        affordances: List[Affordance],
        *,
        keywords: Sequence[str] | None = None,
    ) -> List[Affordance]:
        hints = self._score_affordances(affordances, _normalize_keywords(keywords))
        hint_locators = {hint.extra.get("affordance_id"): hint.score for hint in hints if hint.extra.get("affordance_id")}
        ranked = sorted(
            affordances,
            key=lambda aff: hint_locators.get(id(aff), 0.0),
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------------ scoring internals

    def _score_affordances(
        self,
        affordances: Iterable[Affordance],
        keywords: List[str],
    ) -> List[NavigationHint]:
        hints: List[NavigationHint] = []
        keyword_set = set(keywords)
        for affordance in affordances:
            label_parts = [
                affordance.role or "",
                affordance.name or "",
            ]
            label = " ".join(part for part in label_parts if part).strip()
            base_score = 0.0
            if affordance.role in {"button", "link"}:
                base_score += self._config.role_weight
            if affordance.role in {"textbox", "combobox"}:
                base_score += self._config.role_weight * 0.5
            label_lower = label.lower()
            if keyword_set and any(keyword in label_lower for keyword in keyword_set):
                base_score += self._config.keyword_weight
            hint = NavigationHint(
                label=label or affordance.locator or affordance.role or "element",
                locator=affordance.locator,
                score=base_score,
                source="affordance",
                bbox=None,
                extra={
                    "role": affordance.role,
                    "affordance_id": id(affordance),
                },
            )
            hints.append(hint)
        return hints

    def _score_distilled_nodes(
        self,
        nodes: Iterable[DistilledNode],
        keywords: List[str],
    ) -> List[NavigationHint]:
        hints: List[NavigationHint] = []
        keyword_set = set(keywords)
        for node in nodes:
            label_candidates = [
                node.label,
                node.text,
                node.name,
                node.locator,
                node.path,
            ]
            label = next((item for item in label_candidates if item), None)
            if not label:
                continue
            label_lower = label.lower()
            score = self._config.distilled_weight
            if keyword_set and any(keyword in label_lower for keyword in keyword_set):
                score += self._config.keyword_weight
            if node.role:
                score += self._config.role_weight * 0.5
            if node.text and len(node.text) > 40:
                score += self._config.text_weight
            hints.append(
                NavigationHint(
                    label=label[:100],
                    locator=node.locator,
                    score=score,
                    source="distilled",
                    bbox=node.bbox,
                    extra={
                        "role": node.role,
                        "path": node.path,
                        "visible": node.visible,
                    },
                )
            )
        return hints

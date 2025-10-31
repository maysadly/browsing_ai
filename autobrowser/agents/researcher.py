from __future__ import annotations

from typing import List, Optional

from loguru import logger

from autobrowser.core.schemas import Affordance, PageStatePack


class ResearcherAgent:
    """Lightweight heuristics to help the planner focus on useful page elements."""

    def __init__(self) -> None:
        pass

    def shortlist_affordances(
        self,
        page_state: PageStatePack,
        goal: str,
        limit: int = 6,
    ) -> List[Affordance]:
        """Return most promising affordances to achieve the goal."""

        keyword = self._extract_primary_keyword(goal)
        scored: List[tuple[int, Affordance]] = []
        for affordance in page_state.affordances:
            score = 0
            label = " ".join(
                part
                for part in [
                    affordance.role or "",
                    affordance.name or "",
                ]
                if part
            ).lower()
            if keyword and keyword in label:
                score += 3
            if affordance.role in {"link", "button"}:
                score += 2
            if affordance.role in {"textbox", "combobox"}:
                score += 1
            scored.append((score, affordance))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        logger.info(
            "Researcher: shortlist created",
            goal_keyword=keyword,
            total=len(page_state.affordances),
            shortlisted=len(ranked[:limit]),
        )
        return [aff for _, aff in ranked[:limit]]

    def _extract_primary_keyword(self, goal: str) -> Optional[str]:
        tokens = [token.strip().lower() for token in goal.split() if len(token) > 4]
        return tokens[0] if tokens else None

from __future__ import annotations

from typing import Dict, List

from loguru import logger

from autobrowser.core.schemas import DistilledNode, NavigationHint, PageEntity, PageStatePack


class ExtractorAgent:
    """Transforms page state into structured entities ready for downstream agents."""

    def extract_entities(self, page_state: PageStatePack) -> List[PageEntity]:
        if page_state.entities:
            logger.info(
                "Extractor: using precomputed entities",
                count=len(page_state.entities),
            )
            return page_state.entities
        entities: List[PageEntity] = []
        entities.extend(self._entities_from_snippets(page_state.text_snippets))
        if not entities:
            entities.extend(self._entities_from_distilled(page_state.distilled_elements))
        if not entities and page_state.navigation_hints:
            entities.extend(self._entities_from_hints(page_state.navigation_hints))
        logger.info(
            "Extractor: built entities",
            count=len(entities),
            annotated=bool(page_state.annotated_screenshot_path),
        )
        return entities

    def _entities_from_snippets(self, snippets: List[str]) -> List[PageEntity]:
        entities: List[PageEntity] = []
        for snippet in snippets[:5]:
            parts = snippet.split(". ")
            if not parts:
                continue
            title = parts[0].strip()
            summary = ". ".join(parts[1:]).strip()
            entities.append(
                PageEntity(
                    title=title[:120],
                    summary=summary[:500] or snippet[:500],
                )
            )
        return entities

    def _entities_from_distilled(self, nodes: List[DistilledNode]) -> List[PageEntity]:
        entities: List[PageEntity] = []
        for node in nodes[:10]:
            text = node.text or node.label or node.name
            if not text:
                continue
            entities.append(
                PageEntity(
                    title=(node.label or node.name or node.tag)[:120],
                    summary=text[:400],
                    extra={
                        "locator": node.locator,
                        "role": node.role,
                    },
                )
            )
        return entities

    def _entities_from_hints(self, hints: List[NavigationHint]) -> List[PageEntity]:
        entities: List[PageEntity] = []
        for hint in hints[:5]:
            entities.append(
                PageEntity(
                    title=hint.label[:120],
                    summary=f"Candidate locator {hint.locator or 'n/a'} (score {hint.score:.2f})",
                    extra={
                        "source": hint.source,
                        "locator": hint.locator,
                    },
                )
            )
        return entities

    def build_report(self, entities: List[PageEntity]) -> List[Dict[str, str]]:
        report: List[Dict[str, str]] = []
        for entity in entities:
            report.append(
                {
                    "title": entity.title or "Untitled",
                    "summary": (entity.summary or "")[:400],
                    "url": entity.url or "",
                }
            )
        return report

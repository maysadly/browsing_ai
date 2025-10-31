from __future__ import annotations

from typing import Dict, List

from loguru import logger

from autobrowser.core.schemas import PageEntity, PageStatePack


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
        for snippet in page_state.text_snippets[:5]:
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
        logger.info("Extractor: built entities from snippets", count=len(entities))
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

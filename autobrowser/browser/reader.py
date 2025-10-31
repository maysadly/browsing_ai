from __future__ import annotations

import re
from html import unescape
from typing import Any, Dict, List, Optional

from loguru import logger
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from autobrowser.core.schemas import (
    Affordance,
    PageEntity,
    PageFormField,
    PageStatePack,
)

import time


class PageReader:
    """Extracts structured representation of the current page."""

    def __init__(self, snippets_limit: int = 4000) -> None:
        self.snippets_limit = snippets_limit

    def read(self, page: Page, goals: List[str], screenshot_path: Optional[str]) -> PageStatePack:
        logger.info("Reader: capturing page state", url=page.url)
        affordances = self._gather_affordances(page)
        text_snippets = self._gather_text_snippets(page)
        forms = self._gather_forms(page)
        entities = self._derive_entities(text_snippets)
        logger.info(
            "Reader: page state captured",
            url=page.url,
            affordances=len(affordances),
            snippets=len(text_snippets),
            forms=len(forms),
            entities=len(entities),
        )

        return PageStatePack(
            url=page.url,
            title=self._safe_title(page),
            goals=goals,
            affordances=affordances[:40],
            text_snippets=text_snippets,
            forms=forms,
            entities=entities,
            screenshot_path=screenshot_path,
        )

    def _safe_title(self, page: Page) -> str:
        try:
            return page.title()
        except Exception:  # noqa: BLE001
            return ""

    def _gather_affordances(self, page: Page) -> List[Affordance]:
        affordances: List[Affordance] = []

        for role in ["button", "link", "textbox", "combobox"]:
            try:
                locator = page.get_by_role(role=role)
            except Exception:  # noqa: BLE001
                logger.warning("Reader: failed to query role", role=role)
                continue
            try:
                count = min(locator.count(), 20)
            except Exception:  # noqa: BLE001
                logger.warning("Reader: failed to count elements for role", role=role)
                count = 0
            for idx in range(count):
                element = locator.nth(idx)
                name = ""
                try:
                    name = element.inner_text().strip()
                except Exception:  # noqa: BLE001
                    name = ""
                if not name:
                    try:
                        name = element.get_attribute("name") or element.get_attribute("aria-label") or ""
                    except Exception:  # noqa: BLE001
                        name = ""

                affordances.append(
                    Affordance(
                        role=role,
                        name=name[:120] if name else None,
                        locator=self._build_locator_hint(element, role, name),
                        extra={
                            "href": element.get_attribute("href") if role == "link" else None,
                        },
                    )
                )

        return affordances

    def _gather_text_snippets(self, page: Page) -> List[str]:
        selectors = ["main", "[role='main']", "article", "section", "body"]
        text_content = ""
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                text_candidate = locator.inner_text(timeout=1200)
            except PlaywrightTimeoutError:
                logger.warning("Reader: inner_text timeout", selector=selector)
                text_content = ""
                break
            except Exception:  # noqa: BLE001
                continue
            if text_candidate and text_candidate.strip():
                text_content = text_candidate
                break

        if not text_content.strip():
            try:
                html = page.content(timeout=2000)
                if len(html) > 200_000:
                    html = html[:200_000]
                text_content = self._strip_html(html)
            except Exception:  # noqa: BLE001
                text_content = ""

        chunks: List[str] = []
        for paragraph in text_content.split("\n"):
            paragraph = paragraph.strip()
            if len(paragraph) < 40:
                continue
            chunks.append(paragraph[:400])
            if sum(len(chunk) for chunk in chunks) > self.snippets_limit:
                break
        return chunks

    def _gather_forms(self, page: Page) -> List[List[PageFormField]]:
        forms: List[List[PageFormField]] = []
        try:
            form_elements = page.query_selector_all("form")
        except Exception:  # noqa: BLE001
            return forms

        for form in form_elements[:5]:
            fields: List[PageFormField] = []
            inputs = form.query_selector_all("input, textarea, select")
            for element in inputs:
                label = element.get_attribute("aria-label") or element.get_attribute("placeholder")
                fields.append(
                    PageFormField(
                        label=label,
                        name=element.get_attribute("name"),
                        value=element.get_attribute("value"),
                        type=element.get_attribute("type"),
                        locator=self._build_form_locator(element),
                    )
                )
            if fields:
                forms.append(fields)
        return forms

    def _derive_entities(self, text_snippets: List[str]) -> List[PageEntity]:
        entities: List[PageEntity] = []
        for snippet in text_snippets[:5]:
            sentence = snippet.split(". ")[0]
            entities.append(
                PageEntity(
                    title=sentence[:120],
                    summary=snippet[:300],
                )
            )
        return entities

    def _build_locator_hint(self, element, role: str, name: str) -> Optional[str]:
        identifier = element.get_attribute("id")
        if identifier:
            return f"#{identifier}"
        data_attrs = ["data-testid", "data-test", "data-qa", "data-role"]
        for attr in data_attrs:
            value = element.get_attribute(attr)
            if value:
                return f"[{attr}='{value}']"
        name_attr = element.get_attribute("name")
        if name_attr:
            return f"[name='{name_attr}']"
        if name:
            return f"{role}:has-text('{name[:30]}')"
        return None

    def _build_form_locator(self, element) -> Optional[str]:
        name_attr = element.get_attribute("name")
        if name_attr:
            return f"[name='{name_attr}']"
        placeholder = element.get_attribute("placeholder")
        if placeholder:
            return f"[placeholder='{placeholder}']"
        aria = element.get_attribute("aria-label")
        if aria:
            return f"[aria-label='{aria}']"
        return None

    def _strip_html(self, html: str) -> str:
        if not html:
            return ""
        text = re.sub("(?is)<(script|style)[^>]*>.*?</\\1>", " ", html)
        text = re.sub(r"(?i)</(p|div|section|article|br|li|h[1-6])\s*>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?:\s*\n\s*)+", "\n", text)
        return text.strip()

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logger = logging.getLogger("autobrowser.reader")
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from autobrowser.core.schemas import (
    Affordance,
    BoundingBox,
    DomChangeDetail,
    DomSnapshotDiff,
    DistilledNode,
    PageEntity,
    PageFormField,
    PageStatePack,
)


_DOM_DISTILL_SCRIPT = """
(maxElements) => {
  const computePath = (el) => {
    const segments = [];
    let current = el;
    while (current && current.nodeType === Node.ELEMENT_NODE && segments.length < 8) {
      let segment = current.tagName.toLowerCase();
      if (current.id) {
        segments.unshift(`${segment}#${current.id}`);
        break;
      }
      const className = (current.getAttribute("class") || "").trim();
      if (className) {
        const tokens = className.split(/\\s+/).filter(Boolean).slice(0, 2);
        if (tokens.length) {
          segment += "." + tokens.join(".");
        }
      }
      if (current.parentElement) {
        const sameTagSiblings = Array.from(current.parentElement.children).filter(
          (node) => node.tagName === current.tagName,
        );
        if (sameTagSiblings.length > 1) {
          segment += `:nth-of-type(${sameTagSiblings.indexOf(current) + 1})`;
        }
      }
      segments.unshift(segment);
      current = current.parentElement;
    }
    return segments.join(" > ");
  };

  const computeAccessibleName = (el) => {
    const ariaLabelledby = el.getAttribute("aria-labelledby");
    if (ariaLabelledby) {
      const ids = ariaLabelledby.split(/\\s+/).filter(Boolean);
      const texts = ids
        .map((id) => document.getElementById(id))
        .filter((lbl) => !!lbl)
        .map((lbl) => (lbl.innerText || "").trim())
        .filter(Boolean);
      if (texts.length) {
        return texts.join(" ");
      }
    }
    const ariaLabel = el.getAttribute("aria-label");
    if (ariaLabel) return ariaLabel.trim();
    const labelAttr = el.getAttribute("label");
    if (labelAttr) return labelAttr.trim();
    const placeholder = el.getAttribute("placeholder");
    if (placeholder) return placeholder.trim();
    const title = el.getAttribute("title");
    if (title) return title.trim();
    if (el.tagName === "INPUT") {
      const id = el.getAttribute("id");
      if (id) {
        const label = document.querySelector(`label[for='${CSS.escape(id)}']`);
        if (label) {
          const labelText = (label.innerText || "").trim();
          if (labelText) return labelText;
        }
      }
    }
    return null;
  };

  const tags = [
    "a",
    "button",
    "input",
    "textarea",
    "select",
    "summary",
    "[role]",
    "[data-testid]",
    "[data-test]",
    "[data-qa]",
    "[data-role]",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p"
  ];
  const selector = tags.join(", ");
  const rawElements = Array.from(document.querySelectorAll(selector));
  const results = [];
  for (const el of rawElements) {
    if (!(el instanceof HTMLElement)) {
      continue;
    }
    if (results.length >= maxElements) {
      break;
    }
    const rect = el.getBoundingClientRect();
    const visible =
      rect.width > 1 &&
      rect.height > 1 &&
      rect.bottom > 0 &&
      rect.right > 0;
    const innerText = (el.innerText || "").replace(/\\s+/g, " ").trim();
    results.push({
      path: computePath(el),
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute("role"),
      text: innerText,
      accessibleName: computeAccessibleName(el),
      ariaLabelledby: el.getAttribute("aria-labelledby"),
      ariaDescribedby: el.getAttribute("aria-describedby"),
      label: el.getAttribute("aria-label"),
      nameAttr: el.getAttribute("name"),
      placeholder: el.getAttribute("placeholder"),
      title: el.getAttribute("title"),
      href: el.getAttribute("href"),
      type: el.getAttribute("type"),
      value: el.getAttribute("value"),
      id: el.id || null,
      classes: (el.getAttribute("class") || "").trim(),
      dataset: Object.assign({}, el.dataset || {}),
      visible,
      rect: {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      },
    });
  }
  return results;
}
"""


@dataclass(frozen=True)
class ReaderConfig:
    """Configuration options for the page reader pipeline."""

    snippets_limit: int = 4000
    max_dom_nodes: int = 120
    annotation_top_k: int = 20
    text_min_length: int = 40
    snippet_max_length: int = 400
    affordance_per_role_limit: int = 20
    affordance_output_limit: int = 40
    data_qa_limit: int = 40
    visible_only: bool = True


@dataclass(slots=True)
class ReaderSnapshot:
    """Intermediate artefacts captured during a read cycle."""

    affordances: List[Affordance]
    text_snippets: List[str]
    forms: List[List[PageFormField]]
    entities: List[PageEntity]
    distilled_nodes: List[DistilledNode]
    dom_hash: Optional[str]
    dom_diff: Optional[DomSnapshotDiff]
    annotated_screenshot: Optional[str]


class PageReader:
    """Extracts structured representation of the current page."""

    def __init__(
        self,
        snippets_limit: int = 4000,
        *,
        max_dom_nodes: int = 120,
        annotation_top_k: int = 20,
        config: Optional[ReaderConfig] = None,
    ) -> None:
        if config is None:
            config = ReaderConfig(
                snippets_limit=snippets_limit,
                max_dom_nodes=max_dom_nodes,
                annotation_top_k=annotation_top_k,
            )
        self._config = config
        self.snippets_limit = config.snippets_limit
        self.max_dom_nodes = max(10, config.max_dom_nodes)
        self.annotation_top_k = max(0, config.annotation_top_k)
        self.diff_limit = max(10, min(40, (self.annotation_top_k * 2) or 10))
        self._snapshots: Dict[str, Dict[str, DistilledNode]] = {}
        self._hashes: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()

    def read(
        self,
        page: Page,
        goals: List[str],
        screenshot_path: Optional[str],
        *,
        task_id: Optional[str] = None,
        step_index: Optional[int] = None,
    ) -> PageStatePack:
        logger.info("Reader: capturing page state", url=page.url)
        snapshot = self._capture_snapshot(
            page=page,
            goals=goals,
            screenshot_path=screenshot_path,
            task_id=task_id,
            step_index=step_index,
        )
        state = self._build_state_pack(
            page=page,
            goals=goals,
            screenshot_path=screenshot_path,
            snapshot=snapshot,
        )
        logger.info(
            "Reader: page state captured",
            url=page.url,
            affordances=len(snapshot.affordances),
            snippets=len(snapshot.text_snippets),
            forms=len(snapshot.forms),
            entities=len(snapshot.entities),
            distilled=len(snapshot.distilled_nodes),
            dom_changed=bool(snapshot.dom_diff and (snapshot.dom_diff.added or snapshot.dom_diff.removed or snapshot.dom_diff.changed)),
        )
        return state

    def _capture_snapshot(
        self,
        *,
        page: Page,
        goals: List[str],
        screenshot_path: Optional[str],
        task_id: Optional[str],
        step_index: Optional[int],
    ) -> ReaderSnapshot:
        affordances = self._gather_affordances(page)
        text_snippets = self._gather_text_snippets(page)
        forms = self._gather_forms(page)
        entities = self._derive_entities(text_snippets)
        page_key = self._page_key(page, task_id)
        distilled_nodes, dom_hash, dom_diff = self._distill_page(page, page_key)
        annotated_path = self._annotate_screenshot(screenshot_path, distilled_nodes)
        return ReaderSnapshot(
            affordances=affordances,
            text_snippets=text_snippets,
            forms=forms,
            entities=entities,
            distilled_nodes=distilled_nodes,
            dom_hash=dom_hash,
            dom_diff=dom_diff,
            annotated_screenshot=annotated_path,
        )

    def _build_state_pack(
        self,
        *,
        page: Page,
        goals: List[str],
        screenshot_path: Optional[str],
        snapshot: ReaderSnapshot,
    ) -> PageStatePack:
        affordance_limit = max(0, self._config.affordance_output_limit)
        return PageStatePack(
            url=page.url,
            title=self._safe_title(page),
            goals=goals,
            affordances=snapshot.affordances[:affordance_limit] if affordance_limit else snapshot.affordances,
            text_snippets=snapshot.text_snippets,
            forms=snapshot.forms,
            entities=snapshot.entities,
            screenshot_path=screenshot_path,
            annotated_screenshot_path=snapshot.annotated_screenshot,
            distilled_elements=snapshot.distilled_nodes,
            dom_hash=snapshot.dom_hash,
            dom_diff=snapshot.dom_diff,
        )

    def _page_key(self, page: Page, task_id: Optional[str]) -> str:
        prefix = task_id or "global"
        return f"{prefix}:{id(page)}"

    def _distill_page(
        self,
        page: Page,
        page_key: str,
    ) -> Tuple[List[DistilledNode], Optional[str], Optional[DomSnapshotDiff]]:
        raw_nodes = self._collect_dom_nodes(page)
        distilled_nodes, dom_hash = self._build_distilled_nodes(raw_nodes)
        dom_diff = self._update_dom_snapshot(page_key, distilled_nodes, dom_hash)
        return distilled_nodes, dom_hash, dom_diff

    def _collect_dom_nodes(self, page: Page) -> List[Dict[str, Any]]:
        try:
            return page.evaluate(_DOM_DISTILL_SCRIPT, self.max_dom_nodes) or []
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug("Reader: DOM distillation failed")
            return []

    def _build_distilled_nodes(
        self,
        raw_nodes: List[Dict[str, Any]],
    ) -> Tuple[List[DistilledNode], Optional[str]]:
        distilled: List[DistilledNode] = []
        seen_paths: Dict[str, int] = {}

        visible_nodes = [raw for raw in raw_nodes if raw.get("visible", False)]

        for idx, raw in enumerate(visible_nodes, start=1):
            tag = (raw.get("tag") or "element").lower()
            path = raw.get("path") or f"{tag}[{idx}]"
            if path in seen_paths:
                seen_paths[path] += 1
                path = f"{path}::{seen_paths[path]}"
            else:
                seen_paths[path] = 0

            accessible = (raw.get("accessibleName") or "").strip()
            aria_label = (raw.get("label") or "").strip()
            placeholder = (raw.get("placeholder") or "").strip()
            attr_name = (raw.get("nameAttr") or "").strip()
            text_value = (raw.get("text") or "").strip()
            title_attr = (raw.get("title") or "").strip()

            label_value = accessible or aria_label or placeholder or title_attr or None
            name_value = attr_name or None
            role = raw.get("role") or self._fallback_role(tag, raw.get("type"))
            locator = self._suggest_locator(raw, path, role)
            bbox = None
            rect = raw.get("rect") or {}
            if rect and raw.get("visible", False):
                bbox = BoundingBox(
                    x=float(rect.get("x", 0.0)),
                    y=float(rect.get("y", 0.0)),
                    width=float(rect.get("width", 0.0)),
                    height=float(rect.get("height", 0.0)),
                )

            node_hash = self._hash_node(
                path=path,
                tag=tag,
                role=role,
                label=label_value,
                name=name_value,
                text=text_value,
                locator=locator,
                bbox=bbox,
                visible=bool(raw.get("visible", False)),
            )

            distilled.append(
                DistilledNode(
                    node_id=idx,
                    path=path,
                    tag=tag,
                    role=role,
                    label=label_value[:120] if label_value else None,
                    name=name_value[:120] if name_value else None,
                    text=text_value[:240] if text_value else None,
                    locator=locator,
                    bbox=bbox,
                    visible=bool(raw.get("visible", False)),
                    hash=node_hash,
                )
            )

        hash_input = "|".join(f"{node.path}:{node.hash}" for node in distilled)
        dom_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest() if hash_input else None
        return distilled, dom_hash

    def _fallback_role(self, tag: str, input_type: Optional[str]) -> Optional[str]:
        tag_role_map = {
            "a": "link",
            "button": "button",
            "summary": "button",
            "select": "combobox",
            "textarea": "textbox",
            "label": "label",
        }
        if tag == "input":
            input_type = (input_type or "").lower()
            if input_type in {"email", "text", "search", "tel", "url", "password"}:
                return "textbox"
            if input_type in {"number"}:
                return "spinbutton"
            if input_type in {"checkbox"}:
                return "checkbox"
            if input_type in {"radio"}:
                return "radio"
            if input_type in {"submit", "button"}:
                return "button"
        return tag_role_map.get(tag)

    def _suggest_locator(
        self,
        raw: Dict[str, Any],
        path: str,
        role: Optional[str],
    ) -> Optional[str]:
        identifier = raw.get("id")
        if identifier:
            return f"#{identifier}"

        dataset = raw.get("dataset") or {}
        data_attr_map = {
            "testid": "data-testid",
            "test": "data-test",
            "qa": "data-qa",
            "role": "data-role",
            "id": "data-id",
        }
        for key, attr in data_attr_map.items():
            value = dataset.get(key)
            if value:
                return f"[{attr}='{value}']"

        name_attr = raw.get("nameAttr")
        if name_attr:
            return f"[name='{name_attr}']"

        href = raw.get("href")
        if href:
            return f"a[href='{href}']"

        label = raw.get("accessibleName") or raw.get("text")
        if role and label:
            snippet = str(label).strip()[:30]
            if snippet:
                escaped = snippet.replace("'", "\\'")
                return f"{role}:has-text('{escaped}')"

        # Fallback to distilled path if nothing else usable.
        simplified = path.replace(" > ", " ")
        return simplified if simplified else None

    def _hash_node(
        self,
        *,
        path: str,
        tag: str,
        role: Optional[str],
        label: Optional[str],
        name: Optional[str],
        text: Optional[str],
        locator: Optional[str],
        bbox: Optional[BoundingBox],
        visible: bool,
    ) -> str:
        payload = [
            path,
            tag or "",
            role or "",
            label or "",
            name or "",
            text or "",
            locator or "",
            "1" if visible else "0",
        ]
        if bbox:
            payload.append(
                f"{bbox.x:.1f}:{bbox.y:.1f}:{bbox.width:.1f}:{bbox.height:.1f}"
            )
        materialized = "|".join(payload)
        return hashlib.sha1(materialized.encode("utf-8", "ignore")).hexdigest()

    def _update_dom_snapshot(
        self,
        page_key: str,
        nodes: List[DistilledNode],
        dom_hash: Optional[str],
    ) -> Optional[DomSnapshotDiff]:
        node_map = {node.path: node for node in nodes}

        with self._lock:
            previous_nodes = self._snapshots.get(page_key, {})
            previous_hash = self._hashes.get(page_key)

            added: List[DomChangeDetail] = []
            removed: List[DomChangeDetail] = []
            changed: List[DomChangeDetail] = []

            current_paths = set(node_map.keys())
            previous_paths = set(previous_nodes.keys())

            for path in sorted(current_paths - previous_paths):
                added.append(self._change_from_node(node_map[path]))

            for path in sorted(previous_paths - current_paths):
                removed.append(self._change_from_node(previous_nodes[path]))

            for path in sorted(current_paths & previous_paths):
                previous = previous_nodes[path]
                current = node_map[path]
                if previous.hash != current.hash:
                    fields_changed = self._changed_fields(previous, current)
                    changed.append(
                        DomChangeDetail(
                            path=current.path,
                            locator=current.locator,
                            summary=self._summarize_node(current),
                            fields_changed=fields_changed,
                        )
                    )

            if dom_hash:
                self._hashes[page_key] = dom_hash
            else:
                self._hashes.pop(page_key, None)
            self._snapshots[page_key] = {
                path: node.model_copy(deep=True) for path, node in node_map.items()
            }

        added = added[: self.diff_limit]
        removed = removed[: self.diff_limit]
        changed = changed[: self.diff_limit]

        if not added and not removed and not changed:
            return None

        return DomSnapshotDiff(
            current_hash=dom_hash or "",
            previous_hash=previous_hash,
            added=added,
            removed=removed,
            changed=changed,
        )

    def _change_from_node(self, node: DistilledNode) -> DomChangeDetail:
        return DomChangeDetail(
            path=node.path,
            locator=node.locator,
            summary=self._summarize_node(node),
        )

    def _summarize_node(self, node: DistilledNode) -> str:
        pieces: List[str] = []
        if node.role:
            pieces.append(node.role)
        else:
            pieces.append(node.tag)
        label = node.label or node.text or node.locator or node.name
        if label:
            pieces.append(label[:80])
        return " • ".join([part for part in pieces if part])

    def _changed_fields(self, previous: DistilledNode, current: DistilledNode) -> List[str]:
        fields: List[str] = []
        if previous.text != current.text:
            fields.append("text")
        if previous.label != current.label:
            fields.append("label")
        if previous.name != current.name:
            fields.append("name")
        if previous.locator != current.locator:
            fields.append("locator")
        if previous.visible != current.visible:
            fields.append("visibility")
        if self._bbox_changed(previous.bbox, current.bbox):
            fields.append("bbox")
        return fields

    def _bbox_changed(
        self,
        previous: Optional[BoundingBox],
        current: Optional[BoundingBox],
    ) -> bool:
        if previous is None and current is None:
            return False
        if previous is None or current is None:
            return True
        tolerance = 1.0
        for attr in ("x", "y", "width", "height"):
            if abs(getattr(previous, attr) - getattr(current, attr)) > tolerance:
                return True
        return False

    def _annotate_screenshot(
        self,
        screenshot_path: Optional[str],
        nodes: List[DistilledNode],
    ) -> Optional[str]:
        if not screenshot_path or self.annotation_top_k <= 0:
            return None

        source = Path(screenshot_path)
        if not source.exists():
            return None

        candidates = [node for node in nodes if node.bbox and node.visible]
        if not candidates:
            return None

        candidates = candidates[: self.annotation_top_k]

        try:
            with Image.open(source) as base_image:
                image = base_image.convert("RGBA")
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug(
                "Reader: failed to open screenshot for annotation",
                path=screenshot_path,
            )
            return None

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        width, height = image.size
        for node in candidates:
            bbox = node.bbox
            if not bbox:
                continue
            x1 = max(0, int(bbox.x))
            y1 = max(0, int(bbox.y))
            x2 = max(0, int(bbox.x + bbox.width))
            y2 = max(0, int(bbox.y + bbox.height))
            if x1 >= width or y1 >= height:
                continue
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            draw.rectangle([x1, y1, x2, y2], outline="#FF6B6B", width=2)

            label = str(node.node_id)
            bbox_metrics = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_metrics[2] - bbox_metrics[0]
            text_h = bbox_metrics[3] - bbox_metrics[1]
            padding = 2
            box_w = text_w + padding * 2
            box_h = text_h + padding * 2
            box_x2 = min(width - 1, x1 + box_w)
            box_y2 = min(height - 1, y1 + box_h)
            draw.rectangle([x1, y1, box_x2, box_y2], fill="#FF6B6B")
            draw.text(
                (x1 + padding, y1 + padding),
                label,
                fill="white",
                font=font,
            )

        annotated_path = source.with_name(f"{source.stem}_annotated{source.suffix}")
        try:
            image.save(annotated_path)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug(
                "Reader: failed to save annotated screenshot",
                path=str(annotated_path),
            )
            return None
        finally:
            image.close()

        return str(annotated_path)

    def _safe_title(self, page: Page) -> str:
        try:
            return page.title()
        except Exception:  # noqa: BLE001
            return ""

    def _gather_affordances(self, page: Page) -> List[Affordance]:
        affordances: List[Affordance] = []
        seen: set[str] = set()

        def record_affordance(element, role: Optional[str], name: str, locator_hint: Optional[str], extra: Dict[str, Any]) -> None:
            key = locator_hint or f"{role}:{name}"
            if not key or key in seen:
                return
            seen.add(key)
            affordances.append(
                Affordance(
                    role=role,
                    name=name[:120] if name else None,
                    locator=locator_hint,
                    extra=extra,
                )
            )

        role_limit = max(1, self._config.affordance_per_role_limit)
        require_visible = self._config.visible_only

        for role in ["button", "link", "textbox", "combobox"]:
            try:
                locator = page.get_by_role(role=role)
            except Exception:  # noqa: BLE001
                logger.warning("Reader: failed to query role", role=role)
                continue
            try:
                count = min(locator.count(), role_limit)
            except Exception:  # noqa: BLE001
                logger.warning("Reader: failed to count elements for role", role=role)
                count = 0
            for idx in range(count):
                element = locator.nth(idx)
                if require_visible:
                    try:
                        if not element.is_visible(timeout=500):
                            continue
                    except Exception:  # noqa: BLE001
                        logger.debug("Reader: visibility check failed", role=role, index=idx)
                        continue
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
                locator_hint = self._build_locator_hint(element, role, name)
                record_affordance(
                    element,
                    role,
                    name,
                    locator_hint,
                    {
                        "href": element.get_attribute("href") if role == "link" else None,
                        "data_qa": element.get_attribute("data-qa"),
                    },
                )

        # Additional affordances based on data-qa attributes
        try:
            data_qa_locator = page.locator("[data-qa]")
            count = min(data_qa_locator.count(), self._config.data_qa_limit)
            for idx in range(count):
                element = data_qa_locator.nth(idx)
                if require_visible:
                    try:
                        if not element.is_visible(timeout=500):
                            continue
                    except Exception:  # noqa: BLE001
                        logger.debug("Reader: visibility check failed for data-qa element", index=idx)
                        continue
                data_qa = element.get_attribute("data-qa") or ""
                name = ""
                try:
                    name = element.inner_text().strip()
                except Exception:  # noqa: BLE001
                    name = ""
                locator_hint = self._build_locator_hint(element, None, name)
                if not locator_hint and data_qa:
                    locator_hint = f"[data-qa='{data_qa}']"
                try:
                    tag_name = element.evaluate("el => el.tagName")
                except Exception:  # noqa: BLE001
                    tag_name = None
                record_affordance(
                    element,
                    None,
                    name or data_qa,
                    locator_hint,
                    {
                        "data_qa": data_qa,
                        "tag": tag_name,
                    },
                )
        except Exception:  # noqa: BLE001
            logger.debug("Reader: failed to gather data-qa affordances")

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
        min_len = max(0, self._config.text_min_length)
        max_len = max(min_len, self._config.snippet_max_length)
        budget = self.snippets_limit
        consumed = 0
        for paragraph in text_content.split("\n"):
            paragraph = paragraph.strip()
            if len(paragraph) < min_len:
                continue
            snippet = paragraph[:max_len]
            chunks.append(snippet)
            consumed += len(snippet)
            if consumed > budget:
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
                field_type = element.get_attribute("type") or ""
                data_qa = element.get_attribute("data-qa")
                hidden = (field_type or "").lower() == "hidden"
                fields.append(
                    PageFormField(
                        label=label,
                        name=element.get_attribute("name"),
                        value=element.get_attribute("value"),
                        type=field_type,
                        locator=self._build_form_locator(element),
                        extra={
                            "data_qa": data_qa,
                            "hidden": hidden,
                        },
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

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_PATTERNS: List[Dict[str, str]] = [
    {
        "category": "auth_indicator",
        "text": "Выйти",
    },
    {
        "category": "auth_indicator",
        "text": "Logout",
    },
    {
        "category": "auth_prompt",
        "text": "Войти",
    },
    {
        "category": "auth_prompt",
        "text": "Log in",
    },
    {
        "category": "search_input",
        "role": "textbox",
    },
    {
        "category": "apply_button",
        "text": "Откликнуться",
    },
    {
        "category": "apply_button",
        "text": "Apply",
    },
]


@dataclass
class HeuristicPattern:
    category: str
    domain: Optional[str] = None
    locator: Optional[str] = None
    role: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def matches(self, *, role: Optional[str], locator: Optional[str], text: Optional[str]) -> bool:
        if self.role and role and self.role != role:
            return False
        if self.locator and locator:
            if self.locator != locator:
                return False
        if self.text and text:
            if self.text.lower() not in text.lower():
                return False
        return True

    def to_dict(self) -> Dict[str, Optional[str]]:
        payload = {
            "category": self.category,
            "domain": self.domain,
            "locator": self.locator,
            "role": self.role,
            "text": self.text,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Optional[str]]) -> "HeuristicPattern":
        metadata = data.get("metadata") or {}
        return cls(
            category=data["category"],
            domain=data.get("domain"),
            locator=data.get("locator"),
            role=data.get("role"),
            text=data.get("text"),
            metadata=metadata,
        )


class HeuristicStore:
    """Persistent store of domain-agnostic heuristics discovered during browsing."""

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._patterns: List[HeuristicPattern] = []
        self._load()

    def find(
        self,
        category: str,
        *,
        domain: Optional[str] = None,
    ) -> List[HeuristicPattern]:
        results: List[HeuristicPattern] = []
        for pattern in self._patterns:
            if pattern.category != category:
                continue
            if pattern.domain and domain and pattern.domain != domain:
                continue
            results.append(pattern)
        return results

    def record(
        self,
        category: str,
        *,
        domain: Optional[str] = None,
        locator: Optional[str] = None,
        role: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        candidate = HeuristicPattern(
            category=category,
            domain=domain,
            locator=locator,
            role=role,
            text=text,
            metadata=metadata or {},
        )
        if self._exists(candidate):
            return
        self._patterns.append(candidate)
        self._save()

    def _exists(self, candidate: HeuristicPattern) -> bool:
        for pattern in self._patterns:
            if (
                pattern.category == candidate.category
                and pattern.domain == candidate.domain
                and pattern.locator == candidate.locator
                and pattern.role == candidate.role
                and (pattern.text or "").lower() == (candidate.text or "").lower()
            ):
                return True
        return False

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                data = []
        else:
            data = DEFAULT_PATTERNS
        self._patterns = [HeuristicPattern.from_dict(item) for item in data]
        if not self._path.exists():
            self._save()

    def _save(self) -> None:
        payload = [pattern.to_dict() for pattern in self._patterns]
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def all_categories(self) -> Iterable[str]:
        return sorted({pattern.category for pattern in self._patterns})

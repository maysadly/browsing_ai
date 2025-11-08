from __future__ import annotations

import re
from typing import Iterable


def selector_by_text(text: str) -> str:
    escaped = re.escape(text)
    return f'text="{escaped}"'


def selector_by_role(role: str, name: str | None = None) -> str:
    if name:
        return f'role={role}[name="{name}"]'
    return f"role={role}"


def normalize_host(url: str) -> str:
    pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)/?")
    match = pattern.match(url)
    if not match:
        return url.lower()
    return match.group(1).lower()


def expand_allowlist(hosts: Iterable[str]) -> set[str]:
    normalised = {host.lower() for host in hosts}
    expanded = set(normalised)
    expanded.update(host.lstrip("www.") for host in normalised)
    return expanded

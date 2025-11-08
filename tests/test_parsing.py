from __future__ import annotations

import pytest

from agent.errors import ParsingError
from agent.types import Action, json_repair


def test_json_repair_trailing_commas() -> None:
    raw = """
    {
        "type": "click",
        "selector": "#login",
    }
    """
    action = json_repair(raw, Action)
    assert action.type == "click"
    assert action.selector == "#login"


def test_json_repair_single_quotes() -> None:
    raw = "{'type':'open_url','url':'https://booking.com'}"
    action = json_repair(raw, Action)
    assert action.type == "open_url"
    assert action.url == "https://booking.com"


def test_json_repair_invalid() -> None:
    with pytest.raises(ParsingError):
        json_repair("not json at all", Action)

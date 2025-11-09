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


def test_json_repair_plan_status_field() -> None:
    raw = '{"type":"scroll","reason":"[Task 1/2: Explore] Scrolling down. plan_status=ongoing","plan_status":"ongoing"}'
    action = json_repair(raw, Action)
    assert action.type == "scroll"
    assert action.plan_status == "in_progress"


def test_json_repair_switch_tab_by_index() -> None:
    raw = (
        '{"type":"switch_tab","tab_index":1,'
        '"reason":"[Task 2/3: Resume review] Switch to the resume tab. plan_status=ongoing"}'
    )
    action = json_repair(raw, Action)
    assert action.type == "switch_tab"
    assert action.tab_index == 1


def test_switch_tab_requires_hint() -> None:
    with pytest.raises(ParsingError):
        json_repair(
            '{"type":"switch_tab","reason":"[Task 1/1: Focus] plan_status=ongoing"}',
            Action,
        )


def test_json_repair_invalid() -> None:
    with pytest.raises(ParsingError):
        json_repair("not json at all", Action)

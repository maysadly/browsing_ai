from __future__ import annotations

from agent.core.evaluator import StepEvaluator
from agent.types import Action, InteractiveElement, Observation


def make_observation(visible_text: str, elements: list[InteractiveElement]) -> Observation:
    return Observation(
        url="https://example.com",
        title="Test",
        visible_text=visible_text,
        interactive_elements=elements,
        errors=[],
        page_size=None,
        network_idle=True,
        screenshot_path=None,
    )


def test_detect_blocker_login_gate() -> None:
    elements = [
        InteractiveElement(
            tag="input",
            role="input",
            text="",
            selector="[name=\"login\"]",
            attributes={"name": "login", "type": "text"},
        ),
        InteractiveElement(
            tag="button",
            role="button",
            text="Продолжить",
            selector=".primary",
            attributes={"type": "submit"},
        ),
    ]
    observation = make_observation("Введите телефон\nПродолжить", elements)

    assessment = StepEvaluator().assess(observation)

    assert assessment.login_required is True
    assert assessment.forced_action is not None
    assert assessment.forced_action.type == "await_user_login"
    assert isinstance(assessment.forced_action, Action)


def test_detect_blocker_captcha() -> None:
    observation = make_observation(
        "Please complete the captcha challenge to continue",
        [],
    )

    assessment = StepEvaluator().assess(observation)

    assert assessment.captcha_detected is True
    assert assessment.forced_action is not None
    assert assessment.forced_action.type == "abort"
    assert assessment.forced_action.text == "captcha or bot protection detected"


def test_detect_blocker_no_gate() -> None:
    elements = [
        InteractiveElement(
            tag="input",
            role="input",
            text="",
            selector="#search",
            attributes={"name": "text", "type": "search"},
        )
    ]
    observation = make_observation("Search for jobs", elements)

    assessment = StepEvaluator().assess(observation)

    assert assessment.login_required is False
    assert assessment.captcha_detected is False
    assert assessment.forced_action is None

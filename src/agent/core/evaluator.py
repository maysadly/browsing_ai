from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..types import Action, Observation, RunStatus


@dataclass(slots=True)
class EvaluationResult:
    continue_run: bool
    status: RunStatus
    final_answer: str | None = None
    error: str | None = None
    replan: bool = False


class StepEvaluator:
    """Simple heuristics for detecting completion, loops, or blockers."""

    CAPTCHA_PATTERNS = (
        re.compile(r"captcha", re.IGNORECASE),
        re.compile(r"verify you are", re.IGNORECASE),
    )
    CAPTCHA_KEYWORDS: Sequence[str] = (
        "captcha",
        "robot",
        "verify",
        "антибот",
        "проверка",
        "я не робот",
    )

    LOGIN_TEXT_KEYWORDS: Sequence[str] = (
        "войти",
        "вход",
        "регистрация",
        "зарегистрироваться",
        "sign in",
        "sign up",
        "log in",
        "log on",
        "register",
        "username",
        "password",
        "email",
        "phone",
        "телефон",
        "пароль",
        "аккаунт",
        "account",
        "continue",
        "продолжить",
    )
    LOGIN_INPUT_NAMES: Sequence[str] = (
        "login",
        "email",
        "mail",
        "user",
        "username",
        "phone",
        "tel",
        "password",
        "passwd",
        "pass",
        "otp",
        "code",
    )

    @dataclass(slots=True)
    class BlockerAssessment:
        login_required: bool
        captcha_detected: bool
        forced_action: Action | None

    def assess(self, observation: Observation) -> BlockerAssessment:
        text_content = observation.visible_text
        text_lower = text_content.lower()

        captcha_detected = any(pattern.search(text_content) for pattern in self.CAPTCHA_PATTERNS) or any(
            keyword in text_lower for keyword in self.CAPTCHA_KEYWORDS
        )
        login_required = self._looks_like_login(observation, text_lower)
        forced_action: Action | None = None

        if login_required:
            message = self._make_login_message(text_lower)
            forced_action = Action(
                type="await_user_login",
                message=message,
                reason="Evaluator detected login or registration gate",
            )

        return StepEvaluator.BlockerAssessment(
            login_required=login_required,
            captcha_detected=captcha_detected,
            forced_action=forced_action,
        )

    def evaluate(
        self,
        action: Action,
        observation_before: Observation,
        observation_after: Observation,
        previous_excerpt: str | None,
    ) -> EvaluationResult:
        if action.type == "finish":
            return EvaluationResult(continue_run=False, status="ok", final_answer=action.text)
        if action.type == "abort":
            return EvaluationResult(continue_run=False, status="aborted", error=action.text)

        text = observation_after.visible_text
        excerpt = text[:200]
        if previous_excerpt is not None and previous_excerpt == excerpt:
            return EvaluationResult(continue_run=True, status="ok", replan=True)

        return EvaluationResult(continue_run=True, status="ok")

    def _looks_like_login(self, observation: Observation, text_lower: str) -> bool:
        text_hint = any(keyword in text_lower for keyword in self.LOGIN_TEXT_KEYWORDS)
        input_hint = False
        button_hint = False

        for element in observation.interactive_elements:
            tag = (element.tag or "").lower()
            element_text = (element.text or "").strip().lower()
            attributes = {k: (v or "").lower() for k, v in element.attributes.items()}

            if tag == "input" and self._is_login_input(attributes):
                input_hint = True
            if tag in {"button", "a"} and any(keyword in element_text for keyword in self.LOGIN_TEXT_KEYWORDS):
                button_hint = True

        return (text_hint and input_hint) or (input_hint and button_hint)

    def _is_login_input(self, attributes: dict[str, str]) -> bool:
        input_type = attributes.get("type", "")
        name_attr = attributes.get("name", "")
        aria_label = attributes.get("aria_label", "")

        if input_type in {"password", "email", "tel", "phone", "number"}:
            return True
        fields = (name_attr, aria_label, attributes.get("id", ""))
        return any(
            any(keyword in value for keyword in self.LOGIN_INPUT_NAMES)
            for value in fields
            if value
        )

    @staticmethod
    def _make_login_message(text_lower: str) -> str:
        if any(russian in text_lower for russian in ("войти", "телефон", "пароль", "регистрация")):
            return "Выполните вход вручную, затем нажмите Enter."
        return "Please complete the sign-in manually, then press Enter to continue."

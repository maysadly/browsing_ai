from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

from loguru import logger

from autobrowser.core.schemas import Action, Affordance, PageFormField, PageStatePack, Task
from autobrowser.llm.provider import OpenAIProvider


class PlannerAgent:
    """High level planner and ReAct loop orchestrator."""

    _REQUIRED_FIELDS: Dict[str, List[str]] = {
        "navigate": ["url"],
        "query": ["selector_or_locator"],
        "extract": ["locator"],
        "click": ["locator_or_text"],
    }

    def __init__(self, llm: OpenAIProvider) -> None:
        self._llm = llm
        self._system_prompt = llm.prompt_registry.load("planner.system.txt")
        self._missing_action_prompt = (
            "You are fixing a planner output that omitted a tool call. "
            "Return exactly one JSON object with keys `type` and `params` suitable for dispatching to browser tools. "
            "If no action is possible, respond with `{\"type\": \"wait\", \"params\": {\"reason\": \"<short note>\"}}`."
        )
        self._repair_prompt = (
            "You repair incomplete tool calls for a browser automation agent. "
            "Always return a JSON object with keys `type` and `params`. "
            "Include every mandatory parameter (url, locator, selector, etc.). "
            "Prefer stable selectors (data attributes, role+name) and fully qualified URLs. "
            "Do not add explanation outside the JSON object."
        )
        self._last_plan: List[str] = []
        self._max_repair_attempts = 2

    def build_initial_plan(self, task: Task) -> List[str]:
        user_message = (
            "Goal: {goal}\n"
            "Provide a numbered list (3-6 bullet points) of major steps to solve the goal."
        ).format(goal=task.goal)

        logger.info("Planner: building initial plan", task_id=task.id)
        response = self._llm.chat(
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.1,
        )
        content = response["choices"][0]["message"].get("content", "")
        lines = [line.strip(" -") for line in content.splitlines() if line.strip()]
        plan = [line for line in lines if line]
        self._last_plan = plan
        logger.info("Planner: initial plan lines parsed", count=len(plan))
        return plan

    def decide_next_action(
        self,
        task: Task,
        page_state: PageStatePack,
        short_history: List[str],
        rag_snippets: List[str],
    ) -> Tuple[Action, str]:
        context_chunks = [
            f"Current URL: {page_state.url}",
            f"Page title: {page_state.title}",
            "Recent steps:",
            *[f"- {item}" for item in short_history[-5:]],
            "Relevant snippets:",
            *[f"- {item}" for item in rag_snippets[:5]],
        ]

        affordance_lines = self._format_affordances(page_state.affordances)
        if affordance_lines:
            context_chunks.append("Candidate elements:")
            context_chunks.extend(affordance_lines)

        form_lines = self._format_forms(page_state.forms)
        if form_lines:
            context_chunks.append("Forms detected:")
            context_chunks.extend(form_lines)

        snippet_lines = self._format_snippets(page_state.text_snippets)
        if snippet_lines:
            context_chunks.append("On-page snippets:")
            context_chunks.extend(snippet_lines)

        context_chunks.extend(
            [
            "Goal:",
            task.goal,
            ]
        )
        user_content = "\n".join(context_chunks)

        logger.info(
            "Planner: deciding next action",
            task_id=task.id,
            history=len(short_history),
            rag=len(rag_snippets),
        )
        response = self._llm.chat(
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_content}],
            tools=[self._browser_action_tool_schema()],
            temperature=0.1,
        )

        choice = response["choices"][0]["message"]
        note = choice.get("content") or ""
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            logger.warning(
                "Planner returned no tool call; invoking repair helper",
                task_id=task.id,
                finish_reason=response["choices"][0].get("finish_reason"),
            )
            action = self._repair_missing_action(task, page_state, short_history, rag_snippets, choice)
            return action, note

        arguments = tool_calls[0]["function"].get("arguments", "{}")
        payload = json.loads(arguments)
        action = Action(type=payload["type"], params=payload.get("params", {}))
        logger.info(
            "Planner: action selected",
            task_id=task.id,
            action=action.type,
            note=note,
        )

        action = self._postprocess_action(task, action, page_state, short_history, rag_snippets)
        return action, note

    def _browser_action_tool_schema(self) -> Dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": "dispatch_action",
                "description": "Dispatch a browser action to progress towards the task goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "navigate",
                                "query",
                                "click",
                                "type",
                                "select",
                                "wait",
                                "scroll",
                                "extract",
                                "confirm_gate",
                            ],
                        },
                        "params": {
                            "type": "object",
                            "default": {},
                            "description": "Arguments for the action. Keep optional values compact.",
                        },
                    },
                    "required": ["type"],
                },
            },
        }

    def _postprocess_action(
        self,
        task: Task,
        action: Action,
        page_state: PageStatePack,
        history: List[str],
        rag: List[str],
    ) -> Action:
        current = action
        attempts = 0
        while attempts <= self._max_repair_attempts:
            missing = self._detect_missing_fields(current)
            if not missing:
                return current
            logger.warning(
                "Planner: repairing incomplete action",
                action=current.type,
                missing=missing,
                attempt=attempts + 1,
            )
            current = self._repair_action(task, page_state, history, rag, current, missing)
            attempts += 1

        # Final fallback to heuristics
        params = self._apply_heuristics(task, page_state, current)
        final = Action(type=current.type, params=params)
        missing = self._detect_missing_fields(final)
        if missing:
            logger.error(
                "Planner: unable to repair action after heuristics",
                action=final.type,
                missing=missing,
            )
            raise ValueError(f"Unable to repair action {final.type}: missing {missing}")
        logger.info("Planner: action repaired via heuristics", action=final.type, params=final.params)
        return final

    def _detect_missing_fields(self, action: Action) -> List[str]:
        requirements = self._REQUIRED_FIELDS.get(action.type, [])
        params = action.params or {}
        missing: List[str] = []
        for requirement in requirements:
            if requirement == "url" and not params.get("url"):
                missing.append("url")
            elif requirement == "selector_or_locator" and not (params.get("selector") or params.get("locator")):
                missing.append("selector/locator")
            elif requirement == "locator" and not params.get("locator"):
                missing.append("locator")
            elif requirement == "locator_or_text" and not (params.get("locator") or params.get("text")):
                missing.append("locator/text")
        return missing

    def _repair_action(
        self,
        task: Task,
        page_state: PageStatePack,
        history: List[str],
        rag: List[str],
        action: Action,
        missing: List[str],
    ) -> Action:
        context = {
            "goal": task.goal,
            "current_url": page_state.url,
            "page_title": page_state.title,
            "affordances": [
                {"role": a.role, "name": a.name, "locator": a.locator}
                for a in page_state.affordances[:20]
            ],
            "recent_steps": history[-5:],
            "rag_snippets": rag[:5],
            "original_action": action.model_dump(),
            "missing_fields": missing,
        }

        response = self._llm.chat(
            system=self._repair_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Complete the tool call so that all mandatory parameters are present.\n"
                        f"Context: {json.dumps(context, ensure_ascii=False)}"
                    ),
                }
            ],
            temperature=0,
        )

        content = response["choices"][0]["message"].get("content", "")
        logger.debug("Planner repair response", content=content)
        data = self._extract_json_object(content)
        repaired = Action(type=data["type"], params=data.get("params", {}))
        logger.info("Planner: action repaired", action=repaired.type, params=repaired.params)
        return repaired

    def _repair_missing_action(
        self,
        task: Task,
        page_state: PageStatePack,
        history: List[str],
        rag: List[str],
        planner_message: Dict[str, object],
    ) -> Action:
        context = {
            "goal": task.goal,
            "current_url": page_state.url,
            "page_title": page_state.title,
            "affordances": [
                {"role": a.role, "name": a.name, "locator": a.locator}
                for a in page_state.affordances[:20]
            ],
            "forms": [
                [
                    {
                        "label": field.label,
                        "name": field.name,
                        "locator": field.locator,
                        "type": field.type,
                    }
                    for field in form[:6]
                ]
                for form in page_state.forms[:3]
            ],
            "recent_steps": history[-5:],
            "rag_snippets": rag[:5],
            "planner_output": planner_message,
        }

        response = self._llm.chat(
            system=self._missing_action_prompt,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                }
            ],
            temperature=0,
        )
        content = response["choices"][0]["message"].get("content", "")
        logger.debug("Planner missing-action repair response", content=content)
        data = self._extract_json_object(content)
        action = Action(type=data["type"], params=data.get("params", {}))
        logger.info("Planner: missing action repaired", action=action.type, params=action.params)
        return action

    def _apply_heuristics(self, task: Task, page_state: PageStatePack, action: Action) -> Dict[str, object]:
        params = dict(action.params)
        if action.type == "navigate" and not params.get("url"):
            params["url"] = self._infer_navigation_target(task, page_state)
        elif action.type == "query" and not (params.get("selector") or params.get("locator")):
            locator = self._select_affordance_locator(
                page_state,
                keywords=self._extract_search_keywords(task.goal) or "search query results list",
                roles={"textbox", "combobox"},
            )
            params["locator"] = locator or "body"
        elif action.type == "extract" and not params.get("locator"):
            locator = self._select_affordance_locator(
                page_state,
                keywords=["ваканс", "job", "result", "listing", "card"],
                roles={"link", "article", "listitem"},
            )
            params["locator"] = locator or self._select_form_field_locator(
                page_state, keywords="result card vacancy", field_types=None
            ) or "body"
        elif action.type == "click" and not (params.get("locator") or params.get("text")):
            locator = self._select_affordance_locator(
                page_state,
                keywords=["поиск", "search", "найти", "submit", "login", "sign", "примен", "отклик"],
                roles={"button", "link"},
            )
            if locator:
                params["locator"] = locator
        elif action.type == "type" and not params.get("locator"):
            locator = self._select_affordance_locator(
                page_state,
                keywords=self._extract_search_keywords(task.goal)
                or "search запрос query keywords",
                roles={"textbox", "combobox"},
            )
            if not locator:
                locator = self._select_form_field_locator(
                    page_state, keywords="search query keywords", field_types={"text", "search"}
                )
            if locator:
                params["locator"] = locator
            params.setdefault("clear", True)
        elif action.type == "select" and not params.get("locator"):
            locator = self._select_form_field_locator(
                page_state, keywords="filter sort region location", field_types={"select", "dropdown"}
            )
            if locator:
                params["locator"] = locator
        return params

    def _infer_navigation_target(self, task: Task, page_state: PageStatePack) -> str:
        text_pool = " ".join([task.goal, *self._last_plan])
        url = self._extract_url_from_text(text_pool)
        if url:
            return url
        if page_state.url and page_state.url not in ("about:blank", ""):
            return page_state.url
        keywords = self._extract_search_keywords(task.goal) or task.goal[:120]
        logger.warning("Planner: defaulting to web search", query=keywords)
        return f"https://www.google.com/search?q={quote_plus(keywords)}"

    def _format_affordances(self, affordances: List[Affordance], limit: int = 12) -> List[str]:
        lines: List[str] = []
        for idx, affordance in enumerate(affordances[:limit], start=1):
            parts: List[str] = [f"{idx}. role={affordance.role or 'unknown'}"]
            if affordance.name:
                parts.append(f"name={affordance.name[:60]}")
            if affordance.locator:
                parts.append(f"locator={affordance.locator}")
            href = affordance.extra.get("href") if affordance.extra else None
            if href:
                parts.append(f"href={href[:80]}")
            lines.append("; ".join(parts))
        return lines

    def _format_forms(self, forms: List[List[PageFormField]], limit: int = 3) -> List[str]:
        lines: List[str] = []
        for idx, form in enumerate(forms[:limit], start=1):
            field_descriptions: List[str] = []
            for field in form[:5]:
                label = field.label or field.name or field.type or "field"
                locator = field.locator or field.name or ""
                field_descriptions.append(f"{label[:40]} -> {locator[:60]}")
            if field_descriptions:
                lines.append(f"{idx}. " + "; ".join(field_descriptions))
        return lines

    def _format_snippets(self, snippets: List[str], limit: int = 3, width: int = 180) -> List[str]:
        lines: List[str] = []
        for snippet in snippets[:limit]:
            cleaned = snippet.replace("\n", " ").strip()
            if not cleaned:
                continue
            if len(cleaned) > width:
                cleaned = cleaned[: width - 3] + "..."
            lines.append(f"- {cleaned}")
        return lines

    def _select_affordance_locator(
        self,
        page_state: PageStatePack,
        keywords: Iterable[str] | str,
        roles: Optional[set[str]] = None,
    ) -> Optional[str]:
        if isinstance(keywords, str):
            keyword_tokens = [kw.lower() for kw in keywords.split() if kw]
        else:
            keyword_tokens = []
            for keyword in keywords:
                keyword_tokens.extend([kw.lower() for kw in keyword.split() if kw])
        for affordance in page_state.affordances:
            if roles and affordance.role not in roles:
                continue
            name = (affordance.name or "").lower()
            if keyword_tokens and not any(token in name for token in keyword_tokens):
                continue
            if affordance.locator:
                return affordance.locator
            if affordance.role and affordance.name:
                return f"{affordance.role}:has-text('{affordance.name[:30]}')"
        return None

    def _select_form_field_locator(
        self,
        page_state: PageStatePack,
        keywords: str,
        field_types: Optional[set[str]] = None,
    ) -> Optional[str]:
        keyword_tokens = [kw.lower() for kw in keywords.split() if kw]
        for form in page_state.forms:
            for field in form:
                if field_types and field.type and field.type.lower() not in field_types:
                    continue
                label = " ".join(
                    part for part in [field.label, field.name, field.type] if part
                ).lower()
                if keyword_tokens and not any(token in label for token in keyword_tokens):
                    continue
                if field.locator:
                    return field.locator
        return None

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        explicit = re.search(r"https?://[^\s]+", text)
        if explicit:
            return explicit.group(0).rstrip(".,)")
        return None

    def _extract_search_keywords(self, text: str) -> str:
        words = re.findall(r"[a-zA-ZА-Яа-я0-9+-]+", text.lower())
        stop_words = {
            "найди",
            "найдите",
            "найти",
            "подходящие",
            "подходящих",
            "вакансии",
            "вакансию",
            "вакансий",
            "и",
            "с",
            "на",
            "по",
            "в",
            "моем",
            "моём",
            "резюме",
            "сопроводительным",
            "предварительно",
            "изучив",
            "откликнись",
            "откликнуться",
            "профиле",
            "profile",
            "hh",
            "ru",
            "hh.ru",
            "https",
            "http",
        }
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(filtered[:6])

    def _extract_json_object(self, text: str) -> Dict[str, object]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Repair response did not contain JSON object.")
        return json.loads(match.group(0))

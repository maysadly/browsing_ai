from __future__ import annotations

from typing import Dict, List

from loguru import logger

from autobrowser.core.schemas import Action, PageFormField, PageStatePack


class ApplierAgent:
    """Plans series of actions to populate forms."""

    def plan_form_fill(
        self,
        page_state: PageStatePack,
        payload: Dict[str, str],
    ) -> List[Action]:
        actions: List[Action] = []
        for form in page_state.forms:
            for field in form:
                if not field.name:
                    continue
                if field.name in payload:
                    actions.append(
                        Action(
                            type="type",
                            params={
                                "locator": field.locator or f"[name='{field.name}']",
                                "text": payload[field.name],
                            },
                        )
                    )
        if payload.get("_submit"):
            actions.append(
                Action(
                    type="click",
                    params={"locator": payload["_submit"]},
                )
            )
        logger.info("Applier: planned form actions", planned=len(actions))
        return actions

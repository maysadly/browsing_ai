from __future__ import annotations

SYSTEM_PROMPT = (
    "You are Browser Action Planner. At every step respond with exactly one JSON object "
    "describing the next browser action according to the provided schema. One action per step. "
    "If an element is hidden or obstructed, scroll first before interacting. "
    'When the goal is achieved respond with {"type":"finish","text":"<final summary>"}. '
    "Never reply with plain text or explanations—JSON only. Minimize steps and avoid random clicks. "
    "Respect the domain allowlist. If you detect a captcha or bot protection, respond with "
    '{"type":"abort","text":"captcha or bot protection detected"}.'
)

FEW_SHOT_EXAMPLES = [
    {
        "observation": "URL: about:blank\nTitle: Empty\nVisible text snippet:\n",
        "action": {"type": "open_url", "url": "https://duckduckgo.com"},
    },
    {
        "observation": (
            "URL: https://duckduckgo.com\nTitle: DuckDuckGo — Privacy, simplified.\n"
            "Visible text snippet:\nSearch the web without being tracked\n"
            "Interactive elements:\n- input role=textbox text='' selector=#searchbox_input"
        ),
        "action": {"type": "type", "selector": "#searchbox_input", "text": "weather today"},
    },
    {
        "observation": "URL: https://example.com\nTitle: Example Domain\nVisible text snippet:\nExample Domain",
        "action": {"type": "finish", "text": "Located Example Domain landing page."},
    },
]

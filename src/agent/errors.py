from __future__ import annotations


class AgentError(Exception):
    """Base class for agent specific exceptions."""


class ParsingError(AgentError):
    """Raised when a JSON payload cannot be parsed into a valid schema."""


class LLMError(AgentError):
    """Raised when an LLM provider returns an error."""


class BrowserError(AgentError):
    """Raised for Playwright automation failures."""


class SandboxViolation(AgentError):
    """Raised when sandbox constraints are violated."""


class ObservationError(AgentError):
    """Raised when observation collection fails."""

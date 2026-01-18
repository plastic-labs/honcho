"""Falsifier agent for testing predictions through contradiction search."""

from .agent import FalsifierAgent
from .config import FalsifierConfig
from .prompts import FALSIFIER_SYSTEM_PROMPT, FALSIFIER_TASK_PROMPT

__all__ = [
    "FalsifierAgent",
    "FalsifierConfig",
    "FALSIFIER_SYSTEM_PROMPT",
    "FALSIFIER_TASK_PROMPT",
]

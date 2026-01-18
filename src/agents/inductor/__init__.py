"""Inductor agent for extracting patterns from unfalsified predictions."""

from .agent import InductorAgent
from .config import InductorConfig
from .prompts import INDUCTOR_SYSTEM_PROMPT, INDUCTOR_TASK_PROMPT

__all__ = [
    "InductorAgent",
    "InductorConfig",
    "INDUCTOR_SYSTEM_PROMPT",
    "INDUCTOR_TASK_PROMPT",
]

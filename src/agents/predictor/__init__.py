"""Predictor agent for generating blind predictions from hypotheses."""

from .agent import PredictorAgent
from .config import PredictorConfig
from .prompts import PREDICTOR_SYSTEM_PROMPT, PREDICTOR_TASK_PROMPT

__all__ = [
    "PredictorAgent",
    "PredictorConfig",
    "PREDICTOR_SYSTEM_PROMPT",
    "PREDICTOR_TASK_PROMPT",
]

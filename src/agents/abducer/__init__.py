"""Abducer agent for hypothesis generation.

The Abducer generates explanatory hypotheses from observations (premises).
It follows the scientific method by:
1. Retrieving recent explicit observations
2. Grouping related observations by topic/entity
3. Generating candidate hypotheses that explain the observations
4. Scoring hypotheses by explanatory power
5. Storing hypotheses with complete provenance
"""

from .agent import AbducerAgent
from .config import AbducerConfig
from .prompts import ABDUCER_SYSTEM_PROMPT, ABDUCER_TASK_PROMPT

__all__ = [
    "AbducerAgent",
    "AbducerConfig",
    "ABDUCER_SYSTEM_PROMPT",
    "ABDUCER_TASK_PROMPT",
]

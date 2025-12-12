"""
Coordinator for dream specialists.

The coordinator uses a cheap model to decide which specialists to invoke
based on the pre-scanned context. This avoids running unnecessary specialists
and reduces overall token usage.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.config import settings
from src.utils.clients import honcho_llm_call

if TYPE_CHECKING:
    from src.dreamer.prescan import DreamContext

logger = logging.getLogger(__name__)


COORDINATOR_PROMPT = """You are a dream coordinator. Based on the pre-scanned context, decide which specialist agents to invoke.

## Context Summary
- Explicit observations: {explicit_count}
- Existing deductive observations: {deductive_count}
- Existing inductive observations: {inductive_count}
- Pattern clusters: {cluster_count}

## Available Specialists
1. **deduction** - Creates logical inferences AND detects temporal knowledge updates from explicit facts
2. **induction** - Creates pattern generalizations from observation clusters (top 10)

## Decision Guidelines
- ALWAYS run **deduction** if explicit_count > 0 (extract implicit knowledge + detect updates)
- Run **induction** if cluster_count >= 2 (enough patterns to generalize)
- If nothing needs to be done, return empty list

Return ONLY a JSON array of specialist names in priority order.
Example: ["deduction", "induction"]
No explanation, just the JSON array."""


async def coordinate_dream(context: DreamContext) -> list[str]:
    """
    Use a cheap Haiku call to decide which specialists to invoke.

    Args:
        context: Pre-computed dream context

    Returns:
        List of specialist names to run, in priority order
    """
    # First, apply heuristics to see if we even need to call LLM
    specialists_needed = _apply_heuristics(context)

    # If heuristics are confident, skip LLM call
    if specialists_needed is not None:
        logger.info(f"Coordinator (heuristics): {specialists_needed}")
        return specialists_needed

    # Otherwise, use LLM to decide
    prompt = COORDINATOR_PROMPT.format(
        explicit_count=context.explicit_count,
        deductive_count=context.deductive_count,
        inductive_count=context.inductive_count,
        cluster_count=context.cluster_count,
    )

    response = await honcho_llm_call(
        llm_settings=settings.DREAM,
        prompt=prompt,
        max_tokens=200,
        track_name="Dreamer/Coordinator",
    )

    specialists = _parse_specialist_list(response.content)
    logger.info(f"Coordinator (LLM): {specialists}")
    return specialists


def _apply_heuristics(context: DreamContext) -> list[str] | None:
    """
    Apply simple heuristics to determine specialists without LLM.

    Returns None if uncertain and LLM should decide.
    Returns list of specialists if confident.
    """
    specialists: list[str] = []

    # Run deduction if we have explicit observations (handles both inference + temporal updates)
    if context.explicit_count > 0:
        specialists.append("deduction")

    # Run induction if we have pattern clusters
    if context.cluster_count >= 2:
        specialists.append("induction")

    # If we found clear signals, return them
    if specialists:
        return specialists

    # Nothing to do
    return []


def _parse_specialist_list(response: str) -> list[str]:
    """
    Parse the LLM response to extract specialist names.

    Args:
        response: Raw LLM response

    Returns:
        List of valid specialist names
    """
    valid_specialists = {"deduction", "induction"}

    # Try to parse as JSON first
    try:
        # Find JSON array in response
        response = response.strip()
        if response.startswith("["):
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [s for s in parsed if s in valid_specialists]
    except json.JSONDecodeError:
        pass

    # Fall back to simple string matching
    specialists: list[str] = []
    for name in valid_specialists:
        if name in response.lower():
            specialists.append(name)

    # If nothing matched, return a safe default
    if not specialists:
        logger.warning(
            f"Could not parse specialists from response: {response[:100]}. "
            "Defaulting to deduction only."
        )
        return ["deduction"]

    return specialists

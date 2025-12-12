"""
Dream orchestrator for the specialist-based architecture.

This module coordinates the full dream cycle:
1. Pre-scan (no LLM) - gather all context
2. Coordinate - decide which specialists to run
3. Execute specialists - run in parallel where possible
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dreamer.prescan import prescan_for_dream
from src.dreamer.specialists import SPECIALISTS
from src.utils.agent_tools import create_tool_executor
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
)

if TYPE_CHECKING:
    from src.dreamer.prescan import DreamContext

logger = logging.getLogger(__name__)


async def run_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
) -> None:
    """
    Run a full dream cycle with pre-scan + orchestrator + specialists.

    This is the new efficient dream implementation that:
    1. Pre-computes all context (no LLM calls)
    2. Coordinates which specialists to run (cheap Haiku call or heuristics)
    3. Runs specialists in parallel where possible (focused LLM calls)

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier
    """
    run_id = str(uuid.uuid4())[:8]
    task_name = f"dream_orchestrator_{run_id}"
    start_time = time.perf_counter()

    # Phase 1: Pre-scan (no LLM)
    logger.info(f"[{run_id}] Phase 1: Pre-scan")
    prescan_start = time.perf_counter()

    context = await prescan_for_dream(
        db,
        workspace_name,
        observer,
        observed,
    )

    prescan_ms = (time.perf_counter() - prescan_start) * 1000
    accumulate_metric(task_name, "prescan_duration", prescan_ms, "ms")
    accumulate_metric(task_name, "context", context.summary(), "blob")

    # Phase 2: Coordinate
    logger.info(f"[{run_id}] Phase 2: Coordinate")
    coord_start = time.perf_counter()

    orchestrator_response: OrchestratorResponse = await coordinate_dream(context)

    coord_ms = (time.perf_counter() - coord_start) * 1000
    accumulate_metric(task_name, "orchestrator_duration", coord_ms, "ms")
    accumulate_metric(
        task_name,
        "specialists_selected",
        str(orchestrator_response.specialists),
        "blob",
    )

    # Phase 3: Run specialists
    logger.info(
        f"[{run_id}] Phase 3: Running specialists: {orchestrator_response.specialists}"
    )

    # Create tool executor (shared by all specialists)
    tool_executor = create_tool_executor(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        session_name=session_name,
        include_observation_ids=True,
        history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
    )

    # Determine execution order
    # knowledge_update and deduction can run in parallel
    # induction runs after deduction (can use new deductive obs)
    parallel_phase_1: list[str] = []
    parallel_phase_2: list[str] = []

    for name in orchestrator_response.specialists:
        if name in ("knowledge_update", "deduction"):
            parallel_phase_1.append(name)
        elif name == "induction":
            parallel_phase_2.append(name)

    # Execute Phase 1 specialists in parallel
    if parallel_phase_1:
        logger.info(f"[{run_id}] Running parallel phase 1: {parallel_phase_1}")
        phase_1_tasks = [
            SPECIALISTS[name].run(context, observed, tool_executor)
            for name in parallel_phase_1
        ]
        phase_1_results = await asyncio.gather(*phase_1_tasks, return_exceptions=True)

        for name, res in zip(parallel_phase_1, phase_1_results, strict=True):
            if isinstance(res, BaseException):
                logger.error(f"[{run_id}] Specialist {name} failed: {res}")

    # Execute Phase 2 specialists in parallel (after phase 1)
    if parallel_phase_2:
        logger.info(f"[{run_id}] Running parallel phase 2: {parallel_phase_2}")
        phase_2_tasks = [
            SPECIALISTS[name].run(context, observed, tool_executor)
            for name in parallel_phase_2
        ]
        phase_2_results = await asyncio.gather(*phase_2_tasks, return_exceptions=True)

        for name, res in zip(parallel_phase_2, phase_2_results, strict=True):
            if isinstance(res, BaseException):
                logger.error(f"[{run_id}] Specialist {name} failed: {res}")

    # Log final metrics
    duration_ms = (time.perf_counter() - start_time) * 1000
    accumulate_metric(task_name, "total_duration", duration_ms, "ms")
    accumulate_metric(
        task_name, "specialists_run", len(orchestrator_response.specialists), "count"
    )

    log_performance_metrics("dream_orchestrator", run_id)


ORCHESTRATOR_PROMPT = """You are a dream orchestrator. Based on the pre-scanned context, decide which specialist agents to invoke.

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


class OrchestratorResponse(BaseModel):
    specialists: list[str]


async def coordinate_dream(context: DreamContext) -> OrchestratorResponse:
    """
    Use a cheap Haiku call to decide which specialists to invoke.

    Args:
        context: Pre-computed dream context

    Returns:
        List of specialist names to run, in priority order
    """
    # First, apply heuristics to see if we even need to call LLM
    heuristic_response = _apply_heuristics(context)

    # If heuristics are confident, skip LLM call
    if heuristic_response is not None:
        logger.info(f"orchestrator (heuristics): {heuristic_response.specialists}")
        return heuristic_response

    # Otherwise, use LLM to decide
    prompt = ORCHESTRATOR_PROMPT.format(
        explicit_count=context.explicit_count,
        deductive_count=context.deductive_count,
        inductive_count=context.inductive_count,
        cluster_count=context.cluster_count,
    )

    response: HonchoLLMCallResponse[OrchestratorResponse] = await honcho_llm_call(
        llm_settings=settings.DREAM,
        prompt=prompt,
        max_tokens=200,
        track_name="Dream Orchestrator",
        response_model=OrchestratorResponse,
    )
    orchestrator_response = response.content

    logger.info(f"orchestrator (LLM): {orchestrator_response.specialists}")
    return orchestrator_response


def _apply_heuristics(context: DreamContext) -> OrchestratorResponse | None:
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
    if len(specialists) >= 1:
        return OrchestratorResponse(specialists=specialists)

    return None

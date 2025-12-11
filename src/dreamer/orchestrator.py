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
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dreamer.coordinator import coordinate_dream
from src.dreamer.prescan import prescan_for_dream
from src.dreamer.specialists import SPECIALISTS
from src.utils.agent_tools import create_tool_executor
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of a dream cycle."""

    specialists_run: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    total_tokens: int = 0
    duration_ms: float = 0.0
    context_summary: str = ""


async def run_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
) -> DreamResult:
    """
    Run a full dream cycle with pre-scan + coordinator + specialists.

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

    Returns:
        DreamResult with summary of what was done
    """
    run_id = str(uuid.uuid4())[:8]
    task_name = f"dream_orchestrator_{run_id}"
    start_time = time.perf_counter()

    result = DreamResult()

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
    result.context_summary = context.summary()
    accumulate_metric(task_name, "prescan_duration", prescan_ms, "ms")
    accumulate_metric(task_name, "context", context.summary(), "blob")

    # Phase 2: Coordinate
    logger.info(f"[{run_id}] Phase 2: Coordinate")
    coord_start = time.perf_counter()

    specialists_to_run = await coordinate_dream(context)
    result.specialists_run = specialists_to_run

    coord_ms = (time.perf_counter() - coord_start) * 1000
    accumulate_metric(task_name, "coordinator_duration", coord_ms, "ms")
    accumulate_metric(task_name, "specialists_selected", str(specialists_to_run), "blob")

    if not specialists_to_run:
        logger.info(f"[{run_id}] No specialists needed, dream complete")
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    # Phase 3: Run specialists
    logger.info(f"[{run_id}] Phase 3: Running specialists: {specialists_to_run}")

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

    for name in specialists_to_run:
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

        for name, res in zip(parallel_phase_1, phase_1_results):
            if isinstance(res, BaseException):
                logger.error(f"[{run_id}] Specialist {name} failed: {res}")
                result.results[name] = f"ERROR: {res}"
            else:
                result.results[name] = str(res)

    # Execute Phase 2 specialists in parallel (after phase 1)
    if parallel_phase_2:
        logger.info(f"[{run_id}] Running parallel phase 2: {parallel_phase_2}")
        phase_2_tasks = [
            SPECIALISTS[name].run(context, observed, tool_executor)
            for name in parallel_phase_2
        ]
        phase_2_results = await asyncio.gather(*phase_2_tasks, return_exceptions=True)

        for name, res in zip(parallel_phase_2, phase_2_results):
            if isinstance(res, BaseException):
                logger.error(f"[{run_id}] Specialist {name} failed: {res}")
                result.results[name] = f"ERROR: {res}"
            else:
                result.results[name] = str(res)

    # Log final metrics
    result.duration_ms = (time.perf_counter() - start_time) * 1000
    accumulate_metric(task_name, "total_duration", result.duration_ms, "ms")
    accumulate_metric(task_name, "specialists_run", len(specialists_to_run), "count")

    log_performance_metrics("dream_orchestrator", run_id)

    logger.info(
        f"[{run_id}] Dream complete: ran {len(specialists_to_run)} specialists in {result.duration_ms:.0f}ms"
    )

    return result


async def process_orchestrated_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
) -> str:
    """
    Process a dream using the orchestrated specialist architecture.

    This is the entry point for the new efficient dream system.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier

    Returns:
        Summary string of what was done
    """
    result = await run_dream(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        session_name=session_name,
    )

    # Build summary
    if not result.specialists_run:
        return "No consolidation needed - memory is up to date."

    summaries = [
        f"- {name}: {res[:200]}..." if len(res) > 200 else f"- {name}: {res}"
        for name, res in result.results.items()
    ]

    return (
        f"Dream completed in {result.duration_ms:.0f}ms.\n"
        f"Context: {result.context_summary}\n"
        f"Specialists run: {', '.join(result.specialists_run)}\n"
        f"Results:\n" + "\n".join(summaries)
    )

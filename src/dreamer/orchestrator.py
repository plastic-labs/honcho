"""
Dream orchestrator for the specialist-based architecture.

This module coordinates the full dream cycle:
0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
1. Run deduction specialist (self-directed exploration, creates deductive observations)
2. Run induction specialist (self-directed exploration, creates inductive observations)

Specialists are self-directed agents that explore the observation space and create
higher-level observations. When surprisal sampling finds interesting observations,
they're passed as hints, but specialists are free to follow the evidence wherever it leads.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import sentry_sdk
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.specialists import SPECIALISTS, SpecialistResult
from src.dreamer.surprisal import SurprisalScore  # type: ignore
from src.exceptions import SpecialistExecutionError, SurprisalError
from src.schemas import DreamType
from src.telemetry.events import DreamRunEvent, emit
from src.telemetry.logging import (
    accumulate_metric,
    log_performance_metrics,
)
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of a dream cycle for telemetry reporting."""

    # Run identification
    run_id: str
    specialists_run: list[str]

    # Specialist outcomes
    deduction_success: bool
    induction_success: bool

    # Surprisal sampling
    surprisal_enabled: bool
    surprisal_conclusion_count: int

    # Aggregate metrics
    total_iterations: int
    total_duration_ms: float
    input_tokens: int
    output_tokens: int


async def run_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
) -> DreamResult | None:
    """
    Run a full dream cycle with optional surprisal-based sampling.

    The dream cycle runs specialists sequentially:
    0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
    1. Deduction specialist: Creates deductive observations from explicit facts
    2. Induction specialist: Creates inductive observations from patterns

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier if specified
    """
    if not settings.DREAM.ENABLED:
        return None

    run_id = str(uuid.uuid4())[:8]
    task_name = f"dream_orchestrator_{run_id}"
    start_time = time.perf_counter()

    logger.info(
        f"[{run_id}] Starting dream cycle for {workspace_name}/{observer}/{observed}"
    )

    if session_name is not None:
        session = await crud.get_session(
            db, workspace_name=workspace_name, session_name=session_name
        )
    else:
        session = None

    workspace = await crud.get_workspace(db, workspace_name=workspace_name)

    configuration = get_configuration(None, session, workspace)
    if not configuration.dream.enabled:
        logger.info(
            f"[{run_id}] Dreams disabled for {workspace_name}/{session_name}, skipping dream"
        )
        return None

    # Track specialist outcomes
    deduction_success = False
    induction_success = False
    surprisal_observation_count = 0
    deduction_result: SpecialistResult | None = None
    induction_result: SpecialistResult | None = None

    # Phase 0: Surprisal-based sampling (if enabled)
    # Specialists are self-directed by default - hints are optional suggestions
    exploration_hints: list[str] | None = None

    if settings.DREAM.SURPRISAL.ENABLED:
        logger.info(f"[{run_id}] Phase 0: Computing surprisal scores")
        try:
            from src.dreamer.surprisal import sample_observations_with_surprisal

            high_surprisal_obs = await sample_observations_with_surprisal(
                db=db,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )

            logger.info(
                f"[{run_id}] Surprisal: Found {len(high_surprisal_obs)} high-surprisal observations"
            )
            surprisal_observation_count = len(high_surprisal_obs)
            accumulate_metric(
                task_name, "surprisal_observations", len(high_surprisal_obs), "count"
            )

            if len(high_surprisal_obs) > 0:
                # Use high-surprisal observations as hints for exploration
                exploration_hints = _create_queries_from_surprisal(high_surprisal_obs)
                logger.info(
                    f"[{run_id}] ✨ SURPRISAL HINTS: Suggesting {len(exploration_hints)} "
                    + "high-surprisal topics for specialists to investigate"
                )
                logger.info(
                    f"[{run_id}] Targeting observations with surprisal range: "
                    + f"{high_surprisal_obs[-1].surprisal:.3f} to {high_surprisal_obs[0].surprisal:.3f}"
                )
            else:
                logger.info(
                    f"[{run_id}] No high-surprisal observations - specialists will explore freely"
                )

        except SurprisalError as e:
            logger.error(f"[{run_id}] Surprisal sampling failed: {e}", exc_info=True)
            accumulate_metric(task_name, "surprisal_error", str(e), "blob")
            # Specialists will explore freely without hints

    # Phase 1: Run deduction specialist
    logger.info(f"[{run_id}] Phase 1: Running deduction specialist")
    deduction_specialist = SPECIALISTS["deduction"]
    try:
        deduction_result = await deduction_specialist.run(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            hints=exploration_hints,
            configuration=configuration,
            parent_run_id=run_id,
        )
        logger.info(
            f"[{run_id}] Deduction completed: {deduction_result.content[:200]}..."
        )
        accumulate_metric(
            task_name, "deduction_result", deduction_result.content, "blob"
        )
        deduction_success = deduction_result.success
    except SpecialistExecutionError as e:
        logger.error(f"[{run_id}] Deduction specialist failed: {e}", exc_info=True)
        accumulate_metric(task_name, "deduction_error", str(e), "blob")

    # Phase 2: Run induction specialist (after deduction so it can see new deductive obs)
    logger.info(f"[{run_id}] Phase 2: Running induction specialist")
    induction_specialist = SPECIALISTS["induction"]
    try:
        induction_result = await induction_specialist.run(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            hints=exploration_hints,
            configuration=configuration,
            parent_run_id=run_id,
        )
        logger.info(
            f"[{run_id}] Induction completed: {induction_result.content[:200]}..."
        )
        accumulate_metric(
            task_name, "induction_result", induction_result.content, "blob"
        )
        induction_success = induction_result.success
    except SpecialistExecutionError as e:
        logger.error(f"[{run_id}] Induction specialist failed: {e}", exc_info=True)
        accumulate_metric(task_name, "induction_error", str(e), "blob")

    # Log final metrics
    duration_ms = (time.perf_counter() - start_time) * 1000
    accumulate_metric(task_name, "total_duration", duration_ms, "ms")

    logger.info(f"[{run_id}] Dream cycle completed in {duration_ms:.0f}ms")
    log_performance_metrics("dream_orchestrator", run_id)

    # Aggregate metrics from specialist results
    total_iterations = (deduction_result.iterations if deduction_result else 0) + (
        induction_result.iterations if induction_result else 0
    )
    total_input_tokens = (deduction_result.input_tokens if deduction_result else 0) + (
        induction_result.input_tokens if induction_result else 0
    )
    total_output_tokens = (
        deduction_result.output_tokens if deduction_result else 0
    ) + (induction_result.output_tokens if induction_result else 0)

    # Emit DreamRunEvent with aggregated metrics
    emit(
        DreamRunEvent(
            run_id=run_id,
            workspace_name=workspace_name,
            session_name=session_name,
            observer=observer,
            observed=observed,
            specialists_run=["deduction", "induction"],
            deduction_success=deduction_success,
            induction_success=induction_success,
            surprisal_enabled=settings.DREAM.SURPRISAL.ENABLED,
            surprisal_conclusion_count=surprisal_observation_count,
            total_iterations=total_iterations,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_duration_ms=duration_ms,
        )
    )

    return DreamResult(
        run_id=run_id,
        specialists_run=["deduction", "induction"],
        deduction_success=deduction_success,
        induction_success=induction_success,
        surprisal_enabled=settings.DREAM.SURPRISAL.ENABLED,
        surprisal_conclusion_count=surprisal_observation_count,
        total_iterations=total_iterations,
        total_duration_ms=duration_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def _create_queries_from_surprisal(
    high_surprisal_obs: list[SurprisalScore],
) -> list[str]:
    """
    Create search queries from high-surprisal observations.

    Strategy: Use observation content as semantic search queries.
    Truncate if too long (>200 chars).

    Args:
        high_surprisal_obs: List of SurprisalScore objects

    Returns:
        List of query strings (max 10)
    """
    queries: list[Any] = []
    for score in high_surprisal_obs:
        content = score.observation.content
        if len(content) > 200:
            content = content[:200] + "..."
        queries.append(content)
    return queries[:10]  # Limit to 10 queries


@sentry_sdk.trace
async def process_dream(
    payload: DreamPayload,
    workspace_name: str,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"""
(っ- ‸ - ς)ᶻ z 𐰁 ᶻ z 𐰁 ᶻ z 𐰁\n
DREAM: {payload.dream_type} documents for {workspace_name}/{payload.observer}/{payload.observed}\n
𐰁 z ᶻ 𐰁 z ᶻ 𐰁 z ᶻ(っ- ‸ - ς)"""
    )

    try:
        match payload.dream_type:
            case DreamType.OMNI:
                async with tracked_db("dream_orchestrator") as db:
                    result = await run_dream(
                        db=db,
                        workspace_name=workspace_name,
                        observer=payload.observer,
                        observed=payload.observed,
                        session_name=payload.session_name,
                    )

                # Log completion (telemetry event already emitted in run_dream)
                if result is not None:
                    logger.info(
                        f"Dream completed: run_id={result.run_id}, "
                        + f"iterations={result.total_iterations}, "
                        + f"duration={result.total_duration_ms:.0f}ms"
                    )

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails

"""
Dream orchestrator for the specialist-based architecture.

This module coordinates the full dream cycle:
0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
1. Generate probing questions about the peer (or use surprisal-based queries)
2. Run deduction specialist (creates deductive observations, deletes duplicates)
3. Run induction specialist (creates inductive observations from explicit + deductive)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dreamer.specialists import SPECIALISTS
from src.dreamer.surprisal import SurprisalScore  # type: ignore
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
)

logger = logging.getLogger(__name__)

# Predefined probing questions to guide the specialists
# These serve as semantic entry points for searching observations
PROBING_QUESTIONS: list[str] = [
    "What information has changed or been updated? Look for dates, deadlines, schedules that moved.",
    "What decisions or plans have changed? Look for rescheduled, moved, changed, updated.",
    # Temporal and sequential events
    "What events happened in sequence? Look for things that happened first, then, after, before.",
    "What deadlines, dates, or scheduled events have been mentioned?",
    # Identity and background
    "What do we know about this person's identity, name, or background?",
    # Recent activity
    "What has this entity been doing or discussing recently?",
    # Preferences and interests
    "What are their preferences, likes, or dislikes?",
    # Relationships
    "Who are the important people in their life (family, friends, colleagues)?",
    # Goals and plans
    "What goals, plans, or aspirations have they shared? Have any changed?",
]


async def run_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
) -> None:
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
        session_name: Session identifier
    """

    run_id = str(uuid.uuid4())[:8]
    task_name = f"dream_orchestrator_{run_id}"
    start_time = time.perf_counter()

    logger.info(
        f"[{run_id}] Starting dream cycle for {workspace_name}/{observer}/{observed}"
    )

    # Phase 0: Surprisal-based sampling (if enabled)
    probing_questions = PROBING_QUESTIONS  # Default

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
            accumulate_metric(
                task_name, "surprisal_observations", len(high_surprisal_obs), "count"
            )

            # Hybrid mode: Replace if sufficient high-surprisal observations
            if (
                len(high_surprisal_obs)
                >= settings.DREAM.SURPRISAL.MIN_HIGH_SURPRISAL_FOR_REPLACE
            ):
                probing_questions = _create_queries_from_surprisal(high_surprisal_obs)
                logger.info(
                    f"[{run_id}] ✨ SURPRISAL REPLACE MODE: Using {len(probing_questions)} "
                    + "surprisal-based queries instead of standard questions"
                )
                logger.info(
                    f"[{run_id}] Targeting observations with surprisal range: "
                    + f"{high_surprisal_obs[-1].surprisal:.3f} to {high_surprisal_obs[0].surprisal:.3f}"
                )
            elif len(high_surprisal_obs) > 0:
                # Supplement mode: Add to standard questions
                surprisal_queries = _create_queries_from_surprisal(high_surprisal_obs)
                probing_questions = surprisal_queries + PROBING_QUESTIONS
                logger.info(
                    f"[{run_id}] ✨ SURPRISAL SUPPLEMENT MODE: Adding {len(surprisal_queries)} "
                    + f"surprisal queries to {len(PROBING_QUESTIONS)} standard questions"
                )
                logger.info(
                    f"[{run_id}] Targeting observations with surprisal range: "
                    + f"{high_surprisal_obs[-1].surprisal:.3f} to {high_surprisal_obs[0].surprisal:.3f}"
                )
            else:
                logger.info(
                    f"[{run_id}] No high-surprisal observations found using standard probing questions"
                )

        except Exception as e:
            logger.error(f"[{run_id}] Surprisal sampling failed: {e}", exc_info=True)
            accumulate_metric(task_name, "surprisal_error", str(e), "blob")
            # Fall back to standard probing questions

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
            probing_questions=probing_questions,
        )
        logger.info(f"[{run_id}] Deduction completed: {deduction_result[:200]}...")
        accumulate_metric(task_name, "deduction_result", deduction_result, "blob")
    except Exception as e:
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
            probing_questions=probing_questions,
        )
        logger.info(f"[{run_id}] Induction completed: {induction_result[:200]}...")
        accumulate_metric(task_name, "induction_result", induction_result, "blob")
    except Exception as e:
        logger.error(f"[{run_id}] Induction specialist failed: {e}", exc_info=True)
        accumulate_metric(task_name, "induction_error", str(e), "blob")

    # Log final metrics
    duration_ms = (time.perf_counter() - start_time) * 1000
    accumulate_metric(task_name, "total_duration", duration_ms, "ms")

    logger.info(f"[{run_id}] Dream cycle completed in {duration_ms:.0f}ms")
    log_performance_metrics("dream_orchestrator", run_id)


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

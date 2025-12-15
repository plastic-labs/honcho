"""
Dream orchestrator for the specialist-based architecture.

This module coordinates the full dream cycle:
1. Generate probing questions about the peer
2. Run deduction specialist (creates deductive observations, deletes duplicates)
3. Run induction specialist (creates inductive observations from explicit + deductive)
"""

from __future__ import annotations

import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.dreamer.specialists import SPECIALISTS
from src.utils.logging import (
    accumulate_metric,
    log_performance_metrics,
)

logger = logging.getLogger(__name__)

# Predefined probing questions to guide the specialists
# These serve as semantic entry points for searching observations
PROBING_QUESTIONS: list[str] = [
    # Knowledge updates and changes (HIGH PRIORITY)
    "What information has changed or been updated? Look for dates, deadlines, schedules that moved.",
    "What decisions or plans have changed? Look for rescheduled, moved, changed, updated.",
    # Temporal and sequential events
    "What events happened in sequence? Look for things that happened first, then, after, before.",
    "What deadlines, dates, or scheduled events have been mentioned?",
    # Identity and background
    "What do we know about this person's identity, name, or background?",
    "What is their profession, job, or work situation? Has it changed?",
    # Recent activity
    "What has this person been doing or discussing recently?",
    "What events or activities have they mentioned?",
    # Preferences and interests
    "What are their preferences, likes, or dislikes?",
    "What hobbies, interests, or activities do they enjoy?",
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
    Run a full dream cycle.

    The dream cycle runs specialists sequentially:
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
            probing_questions=PROBING_QUESTIONS,
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
            probing_questions=PROBING_QUESTIONS,
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

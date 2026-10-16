"""Observed progress, separate from a work unit's renewable ownership lease."""

import logging
from collections.abc import Awaitable, Callable
from typing import Literal

from sqlalchemy import func, update

from src import models
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)

WorkUnitStage = Literal[
    "claimed",
    "representation",
    "summary",
    "surprisal",
    "deduction",
    "induction",
    "card_refresh",
]
ProgressCallback = Callable[[WorkUnitStage], Awaitable[None]]


def work_unit_progress(
    work_unit_key: str, claim_id: str, queue_item_id: int | None = None
) -> ProgressCallback:
    """Bind progress to one claim, so a replaced worker cannot update its successor."""

    async def report(stage: WorkUnitStage) -> None:
        try:
            async with tracked_db("work_unit.progress") as db:
                await db.execute(
                    update(models.ActiveQueueSession)
                    .where(
                        models.ActiveQueueSession.id == claim_id,
                        models.ActiveQueueSession.work_unit_key == work_unit_key,
                    )
                    .values(
                        stage=stage,
                        current_queue_item_id=queue_item_id,
                        last_activity_at=func.clock_timestamp(),
                        # Override the model's onupdate: observing a stage must
                        # not change the existing lease/timeout behavior.
                        last_updated=models.ActiveQueueSession.last_updated,
                    )
                )
                await db.commit()
        except Exception:
            # Progress is observational. A reporting failure must neither turn
            # successful work into a retry nor swallow task cancellation.
            logger.warning("Could not report work-unit progress", exc_info=True)

    return report

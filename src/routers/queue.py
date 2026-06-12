"""Queue purge endpoint for stranded work units.

Adds an API endpoint to cancel/purge work units that are stranded
in the queue after their associated sessions have been soft-deleted.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import db, models
from src.dependencies import require_auth

logger = logging.getLogger(__name__)

router = APIRouter()


@router.delete(
    "/{workspace_id}/queue",
    status_code=200,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id"))
    ],
)
async def purge_stranded_work_units(
    workspace_id: str = Path(...),
    session_id: str | None = Query(None, description="Optional: purge only for specific session"),
    status: Literal["all", "unprocessed"] = Query("unprocessed", description="Which items to purge"),
    db: AsyncSession = db,
):
    """
    Purge stranded work units from the queue.

    When sessions are soft-deleted, their associated work units remain
    in the queue permanently. This endpoint cleans them up.

    Args:
        workspace_id: The workspace to purge queue items for
        session_id: Optional session ID to limit purge scope
        status: "unprocessed" (default) or "all" to include processed items
    """
    try:
        # Build filter conditions
        conditions = [
            models.QueueItem.workspace_name == workspace_id,
        ]

        if session_id:
            conditions.append(models.QueueItem.session_id == session_id)

        if status == "unprocessed":
            conditions.append(models.QueueItem.processed == False)

        # Count items to be purged
        count_query = select(models.QueueItem).where(*conditions)
        result = await db.execute(count_query)
        items_to_purge = result.scalars().all()
        count = len(items_to_purge)

        if count == 0:
            return {
                "message": "No stranded work units found",
                "purged_count": 0,
            }

        # Delete the items
        delete_query = delete(models.QueueItem).where(*conditions)
        await db.execute(delete_query)
        await db.commit()

        logger.info(
            "Purged %d stranded work units for workspace %s (session: %s)",
            count,
            workspace_id,
            session_id or "all",
        )

        return {
            "message": f"Purged {count} stranded work units",
            "purged_count": count,
            "workspace_id": workspace_id,
            "session_id": session_id,
        }

    except Exception as e:
        logger.error(f"Failed to purge queue: {e}")
        await db.rollback()
        raise

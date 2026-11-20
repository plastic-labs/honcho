"""The `organize` task: refresh a workspace's suggestions about its imports.

Queued by `POST /ingest/suggestions/refresh` and run by the deriver, so a large
import never ties up a request. It reads the import in a read-only window,
runs the rules with no session held, then writes the suggestions in one short
transaction (`crud.ingest_suggestions`). A workspace that is gone or isn't an
auto workspace has nothing to organize. Experimental.
"""

import datetime
import logging
import time

from sqlalchemy import select

from src import models
from src.crud.ingest_suggestions import load_inputs, open_high_pairs, save_suggestions
from src.dependencies import tracked_db
from src.organizer.identity import suggest_with

logger = logging.getLogger(__name__)


async def refresh_suggestions(workspace_name: str) -> None:
    """Run the identity rules over the workspace's imports and store what they suggest."""
    started = datetime.datetime.now(datetime.UTC)
    async with tracked_db("organize_read", read_only=True) as db:
        workspace_id = await db.scalar(
            select(models.Workspace.id).where(
                models.Workspace.tenant_id == models.scoped_tenant_id(),
                models.Workspace.name == workspace_name,
                models.Workspace.auto_enabled_at.is_not(None),
            )
        )
        if workspace_id is None:
            logger.info("Not organizing %s: no such auto workspace", workspace_name)
            return
        inputs = await load_inputs(db, workspace_name)
        retained = await open_high_pairs(db, workspace_name)

    clock = time.monotonic()
    # Checked with the `high` suggestions still open from earlier refreshes:
    # accepting them all must not join two people either.
    found, demoted = suggest_with(inputs.accounts, inputs.posts, retained)
    elapsed = time.monotonic() - clock

    async with tracked_db("organize_write") as db:
        written = await save_suggestions(
            db,
            workspace_name,
            workspace_id,
            found,
            inputs.snapshots,
            started,
            demoted,
        )
        await db.commit()
    logger.info(
        "Organized %s: %d accounts, %d records, %d suggestions (%d written) in %.1fs",
        workspace_name,
        len(inputs.accounts),
        len(inputs.posts),
        len(found),
        written,
        elapsed,
    )

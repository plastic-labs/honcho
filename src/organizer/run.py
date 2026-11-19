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
from src.crud.ingest_suggestions import load_inputs, save_suggestions
from src.dependencies import tracked_db
from src.organizer.identity import suggest

logger = logging.getLogger(__name__)


async def refresh_suggestions(workspace_name: str) -> None:
    """Run the identity rules over the workspace's imports and store what they suggest."""
    started = datetime.datetime.now(datetime.UTC)
    async with tracked_db("organize_read", read_only=True) as db:
        auto = await db.scalar(
            select(models.Workspace.auto_enabled_at).where(
                models.Workspace.tenant_id == models.scoped_tenant_id(),
                models.Workspace.name == workspace_name,
            )
        )
        if auto is None:
            logger.info("Not organizing %s: no such auto workspace", workspace_name)
            return
        inputs = await load_inputs(db, workspace_name)

    clock = time.monotonic()
    found = suggest(inputs.accounts, inputs.posts)
    elapsed = time.monotonic() - clock

    async with tracked_db("organize_write") as db:
        written = await save_suggestions(
            db, workspace_name, found, inputs.snapshots, started
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

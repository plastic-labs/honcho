"""The end of an auto workspace's dry run.

An auto workspace starts in its dry run: records can be imported, reviewed,
edited and deleted, and nothing reads them to build memory. The dry run ends the
first time something starts deriving the workspace:

- `POST /ingest/derive` releasing records;
- an ingest batch of records sent with `derive: "now"` (deletes alone derive
  nothing);
- a message written through the regular message routes;
- a dream scheduled by hand.

It ends once and for good: `workspaces.ingest_derived_at` is set and never
cleared, since a reset doesn't undo memory already built. After it, edits and
deletes of imported records are refused, and `derive: "auto"` queues.

`end_dry_run` runs in its own transaction holding the workspace fence
exclusive, before the work that ends the dry run is queued. An ingest batch reads
the marker under the shared fence, so it either finishes before the end or sees
it: an edit can't land after memory work has been queued, and a batch that
resolved `auto` to "later" commits its records before a derive call looks for
them.

Experimental, like the rest of ingest.
"""

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.ingest_fence import take_workspace_fence


async def end_dry_run(db: AsyncSession, workspace_name: str) -> None:
    """End the workspace's dry run if it's an auto workspace still in one.

    Call at the start of a transaction, holding no locks. A workspace that isn't
    auto, is gone, or has already ended its dry run is only read: the caller's
    transaction carries on. Otherwise this commits that transaction, ends the dry
    run in one of its own holding the exclusive workspace fence, and commits.
    """
    tenant_id = models.scoped_tenant_id()
    workspace = models.Workspace
    row = (
        await db.execute(
            select(workspace.auto_enabled_at, workspace.ingest_derived_at).where(
                workspace.tenant_id == tenant_id, workspace.name == workspace_name
            )
        )
    ).one_or_none()
    if row is None or row.auto_enabled_at is None or row.ingest_derived_at is not None:
        return
    await db.commit()
    await take_workspace_fence(db, workspace_name, mode="exclusive")
    await db.execute(
        update(workspace)
        .where(
            workspace.tenant_id == tenant_id,
            workspace.name == workspace_name,
            workspace.auto_enabled_at.is_not(None),
            workspace.ingest_derived_at.is_(None),
        )
        .values(ingest_derived_at=func.statement_timestamp())
        .execution_options(synchronize_session=False)
    )
    await db.commit()


__all__ = ["end_dry_run"]

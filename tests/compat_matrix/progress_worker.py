"""Controlled worker operations, sent unchanged to each image with ``python -c``.

The driver supplies fixtures and marks outcomes; this is not a model pipeline.
Imports resolve to the selected image's production code, without source mounts.
"""

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from src import models
from src.crud.deriver import cleanup_stale_work_units
from src.db import SessionLocal, engine
from src.deriver.queue_manager import QueueManager
from src.utils.work_unit import construct_work_unit_key


async def main(request: dict[str, Any]) -> Any:
    manager = QueueManager()
    mode = request["mode"]
    async with SessionLocal() as db:
        if mode == "seed":
            session = (
                await db.scalars(
                    select(models.Session).where(
                        models.Session.workspace_name == request["workspace"],
                        models.Session.name == request["session"],
                    )
                )
            ).one()
            payload = {
                "task_type": request["task"],
                "session_name": session.name,
                "observer": "alice",
                "observed": "bob",
            }
            if request["task"] == "dream":
                payload["dream_type"] = "omni"
            key = construct_work_unit_key(request["workspace"], payload)
            item_ids: list[int] = []
            for _ in range(request["count"]):
                item = models.QueueItem(
                    work_unit_key=key,
                    workspace_name=request["workspace"],
                    session_id=None if request["task"] == "dream" else session.id,
                    task_type=request["task"],
                    payload=payload,
                    processed=False,
                )
                db.add(item)
                await db.flush()
                item_ids.append(item.id)
            claims = (
                await manager.claim_work_units(db, [key])
                if request.get("claim", True)
                else {}
            )
            await db.commit()
            claim = (
                await db.get(models.ActiveQueueSession, claims[key])
                if key in claims
                else None
            )
            return {
                "key": key,
                "claim": claims.get(key),
                "items": item_ids,
                "workspace": request["workspace"],
                "session": session.name,
                "task": request["task"],
                "columns": list(models.ActiveQueueSession.__table__.columns.keys()),
                "progress_before": {
                    name: getattr(claim, name)
                    for name in (
                        "started_at",
                        "last_activity_at",
                        "stage",
                        "current_queue_item_id",
                    )
                    if claim is not None and hasattr(claim, name)
                },
            }
        if mode == "claim":
            result = await manager.claim_work_units(db, [request["key"]])
            await db.commit()
            return result
        if mode == "read":
            claim = await db.get(models.ActiveQueueSession, request["claim"])
            assert claim is not None
            item = await manager.get_next_queue_item(
                request["task"], request["key"], request["claim"]
            )
            return {
                "claim": claim.id,
                "item": item.id if item else None,
                "columns": list(models.ActiveQueueSession.__table__.columns.keys()),
            }
        if mode == "cleanup":
            return await manager._cleanup_work_unit(request["claim"], request["key"])  # pyright: ignore[reportPrivateUsage]
        if mode == "stale":
            await db.execute(
                update(models.ActiveQueueSession)
                .where(models.ActiveQueueSession.id == request["claim"])
                .values(last_updated=datetime.now(UTC) - timedelta(days=1))
            )
            await db.commit()
            await cleanup_stale_work_units(db)
            return await db.get(models.ActiveQueueSession, request["claim"]) is None
        if mode == "report":
            from src.utils.work_unit_progress import work_unit_progress

            await work_unit_progress(
                request["key"], request["claim"], request["items"][0]
            )(request["stage"])
            return True
        if mode in {"mark", "finish"}:
            claim = request["claim"]
            if claim is None:
                claim = (await manager.claim_work_units(db, [request["key"]]))[
                    request["key"]
                ]
                await db.commit()
            items = list(
                await db.scalars(
                    select(models.QueueItem).where(
                        models.QueueItem.id.in_(request["items"])
                    )
                )
            )
            await manager.mark_queue_items_as_processed(items, request["key"])
            if mode == "finish":
                manager.track_worker_work_unit("cmx-progress", request["key"], claim)
                await manager.process_work_unit(request["key"], "cmx-progress")
                # Completed queue rows hide claims from the public projection;
                # inspect the actual ownership table as well as the API ledger.
                assert (
                    await db.scalar(
                        select(models.ActiveQueueSession.id).where(
                            models.ActiveQueueSession.work_unit_key == request["key"]
                        )
                    )
                    is None
                )
            return True
    raise ValueError(mode)


async def run(request: dict[str, Any]) -> Any:
    try:
        return await main(request)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    result = asyncio.run(run(json.loads(sys.argv[1])))
    print("CMX_PROGRESS " + json.dumps(result, default=lambda value: value.isoformat()))

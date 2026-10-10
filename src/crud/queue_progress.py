"""Bounded reads of active memory work, without scanning completed queue history."""

from typing import Any, Literal

from sqlalchemy import Select, case, func, or_, select, true

from src import models


def get_queue_work_units(
    workspace_name: str,
    *,
    task_type: Literal["representation", "summary", "dream"] | None = None,
    observer: str | None = None,
    observed: str | None = None,
    session_name: str | None = None,
) -> Select[Any]:
    """Select one row per claim, attributed to its current processing item.

    Before processing starts, use the oldest pending item. Scope filters
    intersect and are applied after selecting that item: otherwise
    a filter could attribute the running stage to a different queued batch.
    The lateral lookup uses the existing (work_unit_key, processed, id) index.
    Workspace scope is applied inside the lookup as well as in session joins.
    """
    claim = models.ActiveQueueSession
    queue = models.QueueItem
    item = (
        select(queue.task_type, queue.session_id, queue.payload)
        .where(
            queue.work_unit_key == claim.work_unit_key,
            queue.workspace_name == workspace_name,
            queue.processed.is_(False),
            or_(
                claim.current_queue_item_id.is_(None),
                queue.id == claim.current_queue_item_id,
            ),
            queue.task_type.in_(("representation", "summary", "dream")),
        )
        .order_by(queue.id)
        .limit(1)
        .correlate(claim)
        .lateral()
    )
    # Dream records intentionally have no Session FK: their optional session
    # scope is carried in the validated payload, within this workspace.
    session_scope = case(
        (item.c.task_type == "dream", item.c.payload["session_name"].astext),
        else_=models.Session.name,
    )
    stmt = (
        select(
            claim.id,
            item.c.task_type,
            session_scope.label("session_id"),
            item.c.payload["observer"].astext.label("observer"),
            # ->> also maps explicit JSON null to SQL NULL, matching the
            # worker's legacy observer fallback for both absent and null keys.
            case(
                (
                    item.c.payload["observers"].astext.is_(None),
                    func.jsonb_build_array(),
                ),
                else_=item.c.payload["observers"],
            ).label("observers"),
            item.c.payload["observed"].astext.label("observed"),
            claim.started_at,
            claim.stage,
            claim.last_activity_at,
        )
        .select_from(claim)
        .join(item, true())
        .outerjoin(
            models.Session,
            (models.Session.id == item.c.session_id)
            & (models.Session.workspace_name == workspace_name),
        )
        .order_by(claim.id)
    )
    if task_type is not None:
        stmt = stmt.where(item.c.task_type == task_type)
    if observer is not None:
        stmt = stmt.where(
            or_(
                item.c.payload["observer"].astext == observer,
                item.c.payload["observers"].contains([observer]),
            )
        )
    if observed is not None:
        stmt = stmt.where(item.c.payload["observed"].astext == observed)
    if session_name is not None:
        stmt = stmt.where(session_scope == session_name)
    return stmt

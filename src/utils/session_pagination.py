"""Null-aware keyset pagination for session activity, retaining the activity index."""

import datetime

from sqlakeyset import InvalidPage, Paging, serialize_bookmark, unserialize_bookmark
from sqlalchemy import Select, and_, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Session
from src.utils.pagination import CursorPage, CursorParams


async def paginate_session_activity(
    db: AsyncSession,
    stmt: Select[tuple[Session]],
    params: CursorParams,
    *,
    reverse: bool,
) -> CursorPage[Session]:
    """Seek by nullable activity and ID without replacing the indexed ORDER BY.

    sqlakeyset's query builder cannot compare null ordering values. Its public
    bookmark and page metadata still apply, so only the seek predicate needs a
    session-specific implementation. Null activity remains last in either sort
    direction, and a backwards page reverses both that placement and the order.
    """
    raw_cursor = params.to_raw_params().cursor
    if raw_cursor is not None and not isinstance(raw_cursor, str):
        raise InvalidPage("Invalid session activity cursor")
    marker = unserialize_bookmark(raw_cursor or "")
    activity = Session.last_message_at
    session_id = Session.id
    descending = reverse != marker.backwards

    if marker.backwards:
        activity_order = activity.desc() if descending else activity.asc()
        id_order = session_id.desc() if descending else session_id.asc()
        stmt = stmt.order_by(None).order_by(activity_order.nulls_first(), id_order)

    limit = params.size + 1
    if marker.place is not None:
        if len(marker.place) != 2:
            raise InvalidPage("Invalid session activity cursor")
        timestamp, row_id = marker.place
        if not isinstance(row_id, str) or len(row_id) != 21:
            raise InvalidPage("Invalid session activity cursor")
        if timestamp is not None and (
            not isinstance(timestamp, datetime.datetime) or timestamp.tzinfo is None
        ):
            raise InvalidPage("Invalid session activity cursor")
        id_after = session_id < row_id if descending else session_id > row_id
        if timestamp is None:
            after = and_(activity.is_(None), id_after)
            remainder = activity.is_not(None) if marker.backwards else None
        else:
            key = tuple_(activity, session_id)
            after = (
                key < (timestamp, row_id) if descending else key > (timestamp, row_id)
            )
            remainder = activity.is_(None) if not marker.backwards else None
        # Separate bounded seeks keep the timestamp/ID range in Index Cond.
        # Combining that range with OR activity IS NULL scans from the index
        # head on deep pages instead of seeking to the requested position.
        stmt = stmt.offset(None)
        rows = list((await db.execute(stmt.where(after).limit(limit))).all())
        if remainder is not None and len(rows) < limit:
            rows.extend(
                (await db.execute(stmt.where(remainder).limit(limit - len(rows)))).all()
            )
    else:
        rows = list((await db.execute(stmt.limit(limit))).all())
    paging = Paging(
        rows=rows,
        per_page=params.size,
        backwards=marker.backwards,
        current_place=marker.place,
        places=[(row[0].last_message_at, row[0].id) for row in rows],
    )
    return CursorPage[Session].create(
        [row[0] for row in paging.rows],
        params=params,
        total=None,
        current=serialize_bookmark(paging.current),
        current_backwards=serialize_bookmark(paging.current_backwards),
        next_=serialize_bookmark(paging.next) if paging.has_next else None,
        previous=serialize_bookmark(paging.previous) if paging.has_previous else None,
    )

"""Read-only lookups of unprocessed queue work by the peer, session or message it touches.

Each builder returns a ``Select`` of pending ``QueueItem`` rows in one workspace,
ordered by id, to execute with ``db.scalars`` or compose into a larger statement.
Tenant-less reconciler rows (``workspace_name IS NULL``) never match.
"""

from typing import Any

from sqlalchemy import ColumnElement, Select, Text, false, literal_column, or_, select
from sqlalchemy.dialects.postgresql import JSONB

from src.models import QueueItem

__all__ = [
    "pending_work_for_message",
    "pending_work_for_peer",
    "pending_work_for_session",
]


def _payload_json(key: str) -> ColumnElement[Any]:
    # Literal key in ``->`` form; must match the ix_queue_pending_* expressions
    return QueueItem.payload.op("->", return_type=JSONB)(literal_column(f"'{key}'"))


def _payload_text(key: str) -> ColumnElement[str]:
    # Literal key in ``->>`` form; must match the ix_queue_pending_* expressions
    return QueueItem.payload.op("->>", return_type=Text)(literal_column(f"'{key}'"))


def _pending_work(
    workspace_name: str, condition: ColumnElement[bool]
) -> Select[tuple[QueueItem]]:
    return (
        select(QueueItem)
        .where(
            QueueItem.workspace_name == workspace_name,
            QueueItem.processed == false(),
            condition,
        )
        .order_by(QueueItem.id)
    )


def pending_work_for_peer(
    workspace_name: str, peer_name: str
) -> Select[tuple[QueueItem]]:
    """Select pending work whose payload names the peer in any role.

    Matches representation tasks that fan out to the peer or observe it, dreams
    for it as observer or observed, and scope tasks backed by it.
    """
    return _pending_work(
        workspace_name,
        or_(
            _payload_json("observers").has_key(peer_name),
            _payload_text("observed") == peer_name,
            _payload_text("observer") == peer_name,
            _payload_text("scope_peer") == peer_name,
        ),
    )


def pending_work_for_session(
    workspace_name: str, session_name: str
) -> Select[tuple[QueueItem]]:
    """Select pending work whose payload names the session."""
    return _pending_work(workspace_name, _payload_text("session_name") == session_name)


def pending_work_for_message(
    workspace_name: str, message_id: int
) -> Select[tuple[QueueItem]]:
    """Select pending work queued for the message.

    ``message_id`` is the internal ``messages.id``, not the public id.
    """
    return _pending_work(workspace_name, QueueItem.message_id == message_id)

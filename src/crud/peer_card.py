from __future__ import annotations

import logging
from typing import Any, cast

from sqlalchemy import literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.cache.client import safe_cache_delete
from src.crud.lifecycle import LifecycleFence, check_fence
from src.crud.peer import (
    get_or_create_peers,
    peer_cache_key,
    reject_scope_observed,
)

logger = logging.getLogger(__name__)


async def get_peer_card(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
) -> list[str] | None:
    """
    Get peer card from internal_metadata.

    The peer card is returned for the observer/observed relationship. Read from
    the database, never the peer cache.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observed: Peer name of the peer described in the peer card
        observer: Peer name of the observer

    Returns:
        The peer's card text if present, otherwise None.

    Raises:
        ResourceNotFoundException: If the peer does not exist.
    """
    label = construct_peer_card_label(observer=observer, observed=observed)
    row = (
        await db.execute(
            select(
                models.Peer.internal_metadata.op("->", return_type=JSONB)(
                    literal(label)
                )
            ).where(
                models.Peer.workspace_name == workspace_name,
                models.Peer.name == observer,
            )
        )
    ).one_or_none()
    if row is None:
        raise exceptions.ResourceNotFoundException(
            f"Peer {observer} not found in workspace {workspace_name}"
        )
    return cast(list[str] | None, row[0])


async def set_peer_card(
    db: AsyncSession,
    workspace_name: str,
    peer_card: list[str],
    *,
    observer: str,
    observed: str,
    fence: LifecycleFence | None = None,
) -> None:
    """
    Set peer card for a peer.

    If observer_name is provided, the peer card is set for the observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_card: List of strings to set as the peer card
        observed: Peer name of the peer described in the peer card
        observer: Peer name of the observer
        fence: Generations an automatic writer captured with its input. A fenced
            write never creates the observer.

    Raises:
        StaleLifecycleError: If the representation was erased after the fence
            was captured.
    """
    # A scope may be the card's *observer* — the Dreamer writes (scope, observed)
    # cards — but never its subject. Authoritative here rather than only in the
    # route, so the Dreamer and agent-tool paths are covered too, and in the same
    # transaction as the JSONB write below.
    #
    # A *missing* reserved name is refused as well: only the observer is resolved
    # below, so a card keyed on `scope.future` would otherwise persist while that
    # peer does not exist and retroactively describe the scope once created.
    await reject_scope_observed(
        db,
        workspace_name,
        [observed],
        action="No peer card is ever formed about a scope.",
    )

    peers_result = None
    if fence is None:
        peers_result = await get_or_create_peers(
            db, workspace_name, [schemas.PeerSpec(name=observer)]
        )
    else:
        try:
            await check_fence(db, fence)
        except exceptions.StaleLifecycleError:
            await db.rollback()
            raise

    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == observer)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("||")(
                {
                    construct_peer_card_label(
                        observer=observer, observed=observed
                    ): peer_card
                }
            )
        )
    )
    result = cast(CursorResult[Any], await db.execute(stmt))
    if result.rowcount == 0:
        raise exceptions.ResourceNotFoundException(
            f"Peer {observer} not found in workspace {workspace_name}"
        )
    await db.commit()
    if peers_result is not None:
        await peers_result.post_commit()

    # Invalidate cache - read-through pattern
    cache_key = peer_cache_key(workspace_name, observer)
    await safe_cache_delete(cache_key)


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"

from __future__ import annotations

import logging
from typing import cast

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.cache.client import cache
from src.crud.peer import get_peer, peer_cache_key

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

    The peer card is returned for the observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observed: Peer name of the peer described in the peer card
        observer: Peer name of the observer

    Returns:
        The peer's card text if present, otherwise None (also None if peer not found).
    """
    try:
        peer = await get_peer(db, workspace_name, schemas.PeerCreate(name=observer))
        return cast(
            list[str] | None,
            peer.internal_metadata.get(
                construct_peer_card_label(observer=observer, observed=observed)
            ),
        )
    except exceptions.ResourceNotFoundException:
        return None


async def set_peer_card(
    db: AsyncSession,
    workspace_name: str,
    peer_card: list[str],
    *,
    observer: str,
    observed: str,
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

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
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
        .returning(models.Peer)
    )
    result = await db.execute(stmt)
    updated_peer = result.scalar_one_or_none()
    if updated_peer is None:
        raise exceptions.ResourceNotFoundException(
            f"Peer {observer} not found in workspace {workspace_name}"
        )
    await db.commit()

    # Invalidate cache - read-through pattern
    cache_key = peer_cache_key(workspace_name, observer)
    await cache.delete(cache_key)


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"

from __future__ import annotations

import logging
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.cache.client import safe_cache_delete
from src.crud.dream_history import DreamHistoryTarget, record_peer_card
from src.crud.lifecycle_barrier import (
    WriteSnapshot,
    capture_write_snapshot,
    final_write,
)
from src.crud.peer import (
    get_or_create_peers,
    get_peer,
    peer_cache_key,
    reject_scope_observed,
)
from src.utils.sanitization import strip_nul

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
        The peer's card text if present, otherwise None.

    Raises:
        ResourceNotFoundException: If the peer does not exist.
    """
    peer = await get_peer(db, workspace_name, observer)
    return cast(
        list[str] | None,
        peer.internal_metadata.get(
            construct_peer_card_label(observer=observer, observed=observed)
        ),
    )


async def set_peer_card(
    db: AsyncSession,
    workspace_name: str,
    peer_card: list[str],
    *,
    observer: str,
    observed: str,
    history: DreamHistoryTarget | None = None,
    barrier: WriteSnapshot | None = None,
) -> None:
    """
    Set peer card for a peer.

    If observer_name is provided, the peer card is set for the observer/observed relationship.

    The write runs behind the lifecycle barrier, in the transaction this
    function commits. A producer (the Dreamer) passes ``barrier``, what it
    captured before it read the conclusions the card is built from: the card
    is then written only to the captured observer, which is never created
    here. A public write passes none: the observer is created on first use as
    before, and the workspace, the observer and an existing observed peer must
    be live.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_card: List of strings to set as the peer card
        observed: Peer name of the peer described in the peer card
        observer: Peer name of the observer
        history: When set, record the card before and after to this dream run
            in the same transaction
        barrier: What the producer captured before it read its input

    Raises:
        LifecycleBarrierError: A parent was deleted, or a deletion made the
            producer's input stale. Nothing is written
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

    # jsonb rejects NUL in strings. Only the bytes are removed; what counts as a
    # valid entry is the caller's call (the API may set `[]` or free-form lines).
    peer_card = strip_nul(peer_card)

    peers_result = None
    try:
        if barrier is None:
            # A public write. Ensure the observer exists (get-or-create), then
            # capture in this same transaction: the barrier below locks what
            # was just read, so this is not an unlocked preflight.
            peers_result = await get_or_create_peers(
                db, workspace_name, [schemas.PeerSpec(name=observer)]
            )
            barrier = await _capture_public_card_write(
                db, workspace_name, observer=observer, observed=observed
            )
        observer_stamp = barrier.peer(observer)
        if observer_stamp is None:
            raise exceptions.ResourceNotFoundException(
                f"Peer {observer} not found in workspace {workspace_name}"
            )

        label = construct_peer_card_label(observer=observer, observed=observed)
        async with final_write(db, barrier):
            if history is not None:
                # Locked so a concurrent write can't land between this read and
                # the update, which would make `before` wrong. FOR NO KEY
                # UPDATE is what the update itself takes: it blocks another
                # card write, and not the FOR KEY SHARE every final write
                # holds on this row, which FOR UPDATE would deadlock against.
                before = await db.scalar(
                    select(models.Peer.internal_metadata[label])
                    .where(models.Peer.tenant_id == barrier.tenant_id)
                    .where(models.Peer.id == observer_stamp.uid)
                    .with_for_update(key_share=True)
                )
                await record_peer_card(
                    db,
                    history,
                    workspace_name,
                    before=cast(list[str] | None, before),
                    after=peer_card,
                )

            stmt = (
                update(models.Peer)
                .where(models.Peer.tenant_id == barrier.tenant_id)
                .where(models.Peer.id == observer_stamp.uid)
                .values(
                    internal_metadata=models.Peer.internal_metadata.op("||")(
                        {label: peer_card}
                    )
                )
            )
            result = cast(CursorResult[Any], await db.execute(stmt))
            if result.rowcount == 0:
                raise exceptions.ResourceNotFoundException(
                    f"Peer {observer} not found in workspace {workspace_name}"
                )
        await db.commit()
    except exceptions.LifecycleBarrierError:
        await db.rollback()
        raise
    if peers_result is not None:
        await peers_result.post_commit()

    # Invalidate cache - read-through pattern
    cache_key = peer_cache_key(workspace_name, observer)
    await safe_cache_delete(cache_key)


async def _capture_public_card_write(
    db: AsyncSession, workspace_name: str, *, observer: str, observed: str
) -> WriteSnapshot:
    """Capture a public card write: the workspace, the observer, a known observed peer.

    A card may be set about a peer that does not exist yet, as before. A card
    about a peer that exists and is tombstoned is refused.
    """
    names = [observer]
    if observed != observer:
        observed_row = (
            await db.execute(
                select(models.Peer.deleted_at).where(
                    models.Peer.tenant_id == models.scoped_tenant_id(),
                    models.Peer.workspace_name == workspace_name,
                    models.Peer.name == observed,
                )
            )
        ).one_or_none()
        if observed_row is not None:
            if observed_row.deleted_at is not None:
                raise exceptions.WriteParentGoneError(
                    "peer", f"Peer {observed} was deleted"
                )
            names.append(observed)
    return await capture_write_snapshot(db, workspace_name, peer_names=names)


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"

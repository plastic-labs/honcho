"""Message erasure.

``erase_message`` is the seam: it takes the tenant context and the message's
identity, and every storage-specific key predicate stays inside this module.

One transaction:

1. Lock the session row ``FOR UPDATE`` and the message.
2. Bump the session generation and the generation of every affected
   representation (locked in id order).
3. Remove queued work for the message and revoke the claims holding it.
4. Conservative invalidation: hard-delete the session's conclusions and their
   descendants across representations, clear the affected peer cards and both
   summary tiers.
5. Delete the message and its embeddings.

Cache eviction runs after commit and is best-effort; card and summary reads are
served from the database.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import ColumnElement, delete, literal, or_, select, tuple_, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.cache.client import safe_cache_delete
from src.crud.collection import collection_cache_key
from src.crud.peer import peer_cache_key
from src.crud.peer_card import construct_peer_card_label
from src.crud.pending_work import pending_work_for_message
from src.crud.session import session_cache_key
from src.exceptions import ResourceNotFoundException
from src.utils.summarizer import SUMMARIES_KEY

logger = logging.getLogger(__name__)

__all__ = [
    "MessageErasureResult",
    "MessageIdentity",
    "TenantContext",
    "erase_message",
]


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Where the primitive lives. A single-tenant database needs only the workspace."""

    workspace_name: str


@dataclass(frozen=True, slots=True)
class MessageIdentity:
    """Public identity of a message."""

    session_name: str
    message_id: str


@dataclass(slots=True)
class MessageErasureResult:
    message_id: str
    session_generation: int
    representations: list[tuple[str, str]] = field(default_factory=list)
    queue_items_deleted: int = 0
    claims_revoked: int = 0
    conclusions_deleted: int = 0
    cards_cleared: int = 0
    summaries_cleared: bool = False


async def erase_message(
    db: AsyncSession, tenant: TenantContext, identity: MessageIdentity
) -> MessageErasureResult:
    """Erase a message and everything derived from its session, in one transaction.

    Raises:
        ResourceNotFoundException: If the session or message does not exist.
    """
    workspace_name = tenant.workspace_name

    session_row = (
        await db.execute(
            select(
                models.Session.id,
                models.Session.deletion_generation,
                models.Session.internal_metadata.has_key(SUMMARIES_KEY),
            )
            .where(
                models.Session.workspace_name == workspace_name,
                models.Session.name == identity.session_name,
            )
            .with_for_update()
        )
    ).one_or_none()
    if session_row is None:
        raise ResourceNotFoundException(
            f"Session {identity.session_name} not found in workspace {workspace_name}"
        )
    _session_id, session_generation, had_summaries = session_row

    message_row = (
        await db.execute(
            select(models.Message.id)
            .where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == identity.session_name,
                models.Message.public_id == identity.message_id,
            )
            .with_for_update()
        )
    ).one_or_none()
    if message_row is None:
        raise ResourceNotFoundException(
            f"Message {identity.message_id} not found in session"
            + f" {identity.session_name}"
        )
    message_pk: int = message_row[0]

    pending = pending_work_for_message(workspace_name, message_pk).subquery()
    pending_payloads = (await db.execute(select(pending.c.payload))).scalars().all()

    conclusion_ids = await _session_conclusion_closure(
        db, workspace_name, identity.session_name
    )

    representations = await _affected_representations(
        db, workspace_name, conclusion_ids, pending_payloads
    )

    new_session_generation = session_generation + 1
    await db.execute(
        update(models.Session)
        .where(
            models.Session.workspace_name == workspace_name,
            models.Session.name == identity.session_name,
        )
        .values(
            deletion_generation=new_session_generation,
            internal_metadata=models.Session.internal_metadata.op("-")(
                literal(SUMMARIES_KEY)
            ),
        )
    )
    if representations:
        locked_collection_ids = (
            await db.scalars(
                select(models.Collection.id)
                .where(
                    models.Collection.workspace_name == workspace_name,
                    tuple_(models.Collection.observer, models.Collection.observed).in_(
                        representations
                    ),
                )
                .order_by(models.Collection.id)
                .with_for_update()
            )
        ).all()
        if locked_collection_ids:
            await db.execute(
                update(models.Collection)
                .where(models.Collection.id.in_(locked_collection_ids))
                .values(deletion_generation=models.Collection.deletion_generation + 1)
            )

    claims_revoked = _rowcount(
        await db.execute(
            delete(models.ActiveQueueSession).where(
                models.ActiveQueueSession.work_unit_key.in_(
                    select(pending.c.work_unit_key)
                )
            )
        )
    )
    # Processed rows too: queue.message_id is a foreign key with no cascade
    queue_items_deleted = _rowcount(
        await db.execute(
            delete(models.QueueItem).where(
                models.QueueItem.workspace_name == workspace_name,
                models.QueueItem.message_id == message_pk,
            )
        )
    )

    conclusions_deleted = 0
    if conclusion_ids:
        conclusions_deleted = _rowcount(
            await db.execute(
                delete(models.Document).where(
                    models.Document.id.in_(
                        select(models.Document.id)
                        .where(models.Document.id.in_(conclusion_ids))
                        .order_by(models.Document.id)
                        .with_for_update()
                    )
                )
            )
        )

    cards_cleared = await _clear_peer_cards(db, workspace_name, representations)

    await db.execute(
        delete(models.MessageEmbedding).where(
            models.MessageEmbedding.workspace_name == workspace_name,
            models.MessageEmbedding.message_id == identity.message_id,
        )
    )
    await db.execute(delete(models.Message).where(models.Message.id == message_pk))
    await db.commit()

    await _evict_cache(workspace_name, identity.session_name, representations)

    logger.info(
        "Erased message %s from %s/%s (conclusions=%d, cards=%d, queue=%d, claims=%d)",
        identity.message_id,
        workspace_name,
        identity.session_name,
        conclusions_deleted,
        cards_cleared,
        queue_items_deleted,
        claims_revoked,
    )
    return MessageErasureResult(
        message_id=identity.message_id,
        session_generation=new_session_generation,
        representations=representations,
        queue_items_deleted=queue_items_deleted,
        claims_revoked=claims_revoked,
        conclusions_deleted=conclusions_deleted,
        cards_cleared=cards_cleared,
        summaries_cleared=bool(had_summaries),
    )


async def _session_conclusion_closure(
    db: AsyncSession, workspace_name: str, session_name: str
) -> list[str]:
    """Ids of the session's conclusions plus every conclusion derived from them.

    The closure follows ``document_sources`` edges across representations, so a
    cross-session deduction or induction built on a session conclusion is
    included.
    """
    seed = (
        select(models.Document.id.label("id"))
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.session_name == session_name,
        )
        .cte("closure", recursive=True)
    )
    closure = seed.union(
        select(models.DocumentSource.derived_id).join(
            seed, models.DocumentSource.source_id == seed.c.id
        )
    )
    return list((await db.scalars(select(closure.c.id).distinct())).all())


async def _affected_representations(
    db: AsyncSession,
    workspace_name: str,
    conclusion_ids: list[str],
    pending_payloads: Sequence[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Representations whose conclusions are invalidated or whose queued work is removed."""
    pairs: set[tuple[str, str]] = set()
    if conclusion_ids:
        rows = await db.execute(
            select(models.Document.observer, models.Document.observed)
            .where(
                models.Document.workspace_name == workspace_name,
                models.Document.id.in_(conclusion_ids),
            )
            .distinct()
        )
        pairs.update((observer, observed) for observer, observed in rows)
    for payload in pending_payloads:
        observed = payload.get("observed")
        observers = payload.get("observers")
        if isinstance(observed, str) and isinstance(observers, list):
            pairs.update(
                (observer, observed)
                for observer in cast(list[object], observers)
                if isinstance(observer, str)
            )
    return sorted(pairs)


async def _clear_peer_cards(
    db: AsyncSession, workspace_name: str, representations: list[tuple[str, str]]
) -> int:
    """Remove the card each affected representation's observer holds about its observed."""
    labels_by_observer: dict[str, list[str]] = {}
    for observer, observed in representations:
        labels_by_observer.setdefault(observer, []).append(
            construct_peer_card_label(observer=observer, observed=observed)
        )
    if not labels_by_observer:
        return 0

    cleared = 0
    peer_ids = (
        await db.scalars(
            select(models.Peer.id)
            .where(
                models.Peer.workspace_name == workspace_name,
                models.Peer.name.in_(labels_by_observer),
            )
            .order_by(models.Peer.id)
            .with_for_update()
        )
    ).all()
    if not peer_ids:
        return 0
    for observer, labels in sorted(labels_by_observer.items()):
        result = await db.execute(
            update(models.Peer)
            .where(
                models.Peer.workspace_name == workspace_name,
                models.Peer.name == observer,
                or_(
                    *(models.Peer.internal_metadata.has_key(label) for label in labels)
                ),
            )
            .values(
                internal_metadata=_without_keys(models.Peer.internal_metadata, labels)
            )
        )
        cleared += _rowcount(result)
    return cleared


def _rowcount(result: Any) -> int:
    return cast(CursorResult[Any], result).rowcount


def _without_keys(column: Any, keys: list[str]) -> ColumnElement[dict[str, Any]]:
    expression: ColumnElement[dict[str, Any]] = column
    for key in keys:
        expression = expression.op("-")(literal(key))
    return expression


async def _evict_cache(
    workspace_name: str,
    session_name: str,
    representations: list[tuple[str, str]],
) -> None:
    await safe_cache_delete(session_cache_key(workspace_name, session_name))
    for observer in sorted({observer for observer, _ in representations}):
        await safe_cache_delete(peer_cache_key(workspace_name, observer))
    for observer, observed in representations:
        await safe_cache_delete(
            collection_cache_key(workspace_name, observer, observed)
        )

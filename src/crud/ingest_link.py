"""Linking imported accounts into one person, and resetting an import.

A link changes which person an account maps to; it never moves messages that
were already imported. The merged people must have nothing imported yet (no
messages, no session memberships, nothing derived), so a dry run is reset first
(`reset_import`), then linked, then imported again: the next import follows the
link, because ingest follows a confirmed binding even to a peer it didn't create.

All three take the exclusive workspace fence before any row lock, which
serializes them against every ingest transaction, the ingest enqueuer and
session deletion in the workspace (`crud.ingest_fence`).

Every statement filters each table on `tenant_id` itself: several tables are
named only in subqueries, where the ORM tenant hook doesn't reach.
"""

import datetime
from collections.abc import Sequence
from typing import Any

from sqlalchemy import case, exists, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.ingest_fence import take_workspace_fence
from src.crud.peer import reject_scope_peers
from src.crud.peer_card import construct_peer_card_label
from src.deriver.enqueue import enqueue_deletion
from src.exceptions import ConflictException, ResourceNotFoundException

RESET_FIRST = (
    " Reset the import with POST /ingest/reset (`honcho ingest reset`), link,"
    + " then import again."
)


def _ingest_metadata(peer: models.Peer) -> dict[str, Any]:
    ingest: Any = (peer.internal_metadata or {}).get("ingest")
    return dict(ingest) if isinstance(ingest, dict) else {}  # pyright: ignore[reportUnknownArgumentType]


async def _peers(
    db: AsyncSession, workspace_name: str, names: Sequence[str]
) -> dict[str, models.Peer]:
    """The named peers, locked in name order."""
    found = {
        p.name: p
        for p in await db.scalars(
            select(models.Peer)
            .where(
                models.Peer.tenant_id == models.scoped_tenant_id(),
                models.Peer.workspace_name == workspace_name,
                models.Peer.name.in_(list(names)),
            )
            .order_by(models.Peer.name)
            .with_for_update()
        )
    }
    missing = sorted(set(names) - set(found))
    if missing:
        raise ResourceNotFoundException(f"Peer(s) {missing} not found")
    return found


async def _merged_into(
    db: AsyncSession, workspace_name: str, peer_name: str
) -> list[str]:
    """The peers linked into `peer_name`, in name order."""
    peer = models.Peer
    return list(
        await db.scalars(
            select(peer.name)
            .where(
                peer.tenant_id == models.scoped_tenant_id(),
                peer.workspace_name == workspace_name,
                peer.internal_metadata["ingest"]["merged_into"].astext == peer_name,
            )
            .order_by(peer.name)
        )
    )


def _trusted_bindings(workspace_name: str, peer: models.Peer) -> Any:
    """The peer's bindings ingest trusts: it created the peer, or a person confirmed it."""
    binding = models.IngestPeerBinding
    owner = _ingest_metadata(peer).get("binding_id")
    return select(binding).where(
        binding.tenant_id == models.scoped_tenant_id(),
        binding.workspace_name == workspace_name,
        binding.peer_name == peer.name,
        or_(binding.id == owner, binding.confirmed_at.is_not(None)),
    )


async def _refuse_if_imported(db: AsyncSession, workspace_name: str, name: str) -> None:
    """Refuse a person who already has imported messages or derived state."""
    tenant_id = models.scoped_tenant_id()
    message = models.Message
    member = models.SessionPeer
    collection = models.Collection
    peer = models.Peer
    has_messages = await db.scalar(
        select(
            exists().where(
                message.tenant_id == tenant_id,
                message.workspace_name == workspace_name,
                message.peer_name == name,
            )
        )
    )
    if has_messages:
        raise ConflictException(f"Peer {name} already has messages." + RESET_FIRST)
    in_sessions = await db.scalar(
        select(
            exists().where(
                member.tenant_id == tenant_id,
                member.workspace_name == workspace_name,
                member.peer_name == name,
            )
        )
    )
    if in_sessions:
        raise ConflictException(f"Peer {name} is in sessions." + RESET_FIRST)
    derived = await db.scalar(
        select(
            exists().where(
                collection.tenant_id == tenant_id,
                collection.workspace_name == workspace_name,
                or_(collection.observer == name, collection.observed == name),
            )
        )
    )
    # A peer card lives on its observer: the peer's own cards, or another
    # peer's card about it.
    card_about = await db.scalar(
        select(
            exists().where(
                peer.tenant_id == tenant_id,
                peer.workspace_name == workspace_name,
                peer.internal_metadata.has_key(
                    construct_peer_card_label(observer="", observed=name)
                ),
            )
        )
    )
    if derived or card_about:
        raise ConflictException(
            f"Something has already been derived about peer {name}; linking after"
            + " derivation isn't supported yet."
        )


def _has_own_cards(peer: models.Peer) -> bool:
    return any(
        key == "peer_card" or key.endswith("_peer_card")
        for key in (peer.internal_metadata or {})
    )


async def _accounts(
    db: AsyncSession, workspace_name: str, peer: models.Peer
) -> list[schemas.IngestAccount]:
    binding = models.IngestPeerBinding
    owner = _ingest_metadata(peer).get("binding_id")
    rows = (
        await db.scalars(
            _trusted_bindings(workspace_name, peer).order_by(
                # The person's own account first, as the status lists it.
                case((binding.id == owner, 0), else_=1),
                binding.created_at,
                binding.id,
            )
        )
    ).all()
    return [
        schemas.IngestAccount(
            source=b.source,
            source_instance=b.source_instance,
            id=b.key_value if b.key_kind == "external_id" else None,
            display_name=b.display_name,
        )
        for b in rows
    ]


async def _set_merged_into(
    db: AsyncSession, workspace_name: str, peer_name: str, into: str | None
) -> None:
    """Set or clear `internal_metadata.ingest.merged_into`, keeping other keys."""
    peer = models.Peer
    current = func.coalesce(peer.internal_metadata["ingest"], literal({}, JSONB))
    ingest = (
        current.op("||")(func.jsonb_build_object("merged_into", into))
        if into is not None
        else current.op("-")("merged_into")
    )
    await db.execute(
        update(peer)
        .where(
            peer.tenant_id == models.scoped_tenant_id(),
            peer.workspace_name == workspace_name,
            peer.name == peer_name,
        )
        .values(
            # Replaces only the "ingest" key, with its other entries kept.
            internal_metadata=func.coalesce(
                peer.internal_metadata, literal({}, JSONB)
            ).op("||")(func.jsonb_build_object("ingest", ingest))
        )
    )


async def link_peers(
    db: AsyncSession, workspace_name: str, names: Sequence[str]
) -> schemas.IngestLinkResult:
    """Map every account of `names[1:]` to `names[0]`, for the next import.

    Raises:
        ResourceNotFoundException: A peer doesn't exist.
        ConflictException: A merged person already has imported messages,
            memberships or derived state, isn't an imported person, or a peer
            is named twice or already merged.
        ValidationException: A peer is a scope.
    """
    keep_name, merge_names = names[0], list(names[1:])
    if len(set(names)) != len(names):
        raise ConflictException("Each peer can be named only once")
    await take_workspace_fence(db, workspace_name, mode="exclusive")
    await reject_scope_peers(
        db, workspace_name, names, action="Scopes can't be linked."
    )

    # People already linked into a merged one move to the kept person too. Only
    # a link sets that marker, and links are serialized by the fence above, so
    # it can be read before locking.
    chained = {
        name: await _merged_into(db, workspace_name, name) for name in merge_names
    }
    # Every peer this link writes is locked in name order, the order other
    # writers use.
    peers = await _peers(
        db, workspace_name, [*names, *(e for c in chained.values() for e in c)]
    )
    keep = peers[keep_name]
    if _ingest_metadata(keep).get("merged_into"):
        raise ConflictException(
            f"Peer {keep_name} was merged into another peer; link into that one"
        )
    binding = models.IngestPeerBinding
    for name in sorted(merge_names):
        merged = peers[name]
        owner = _ingest_metadata(merged).get("binding_id")
        bindings = (
            await db.scalars(
                _trusted_bindings(workspace_name, merged)
                .order_by(binding.id)
                .with_for_update()
            )
        ).all()
        if owner is None or not bindings:
            raise ConflictException(
                f"Peer {name} isn't an imported person: only people an import"
                + " created can be linked into another"
            )
        unnamed = [b for b in bindings if b.key_kind != "external_id"]
        if unnamed:
            raise ConflictException(
                f"Peer {name} has accounts without an ID in their source (matched"
                + " by name), which can't be linked or unlinked"
            )
        await _refuse_if_imported(db, workspace_name, name)
        if _has_own_cards(merged):
            raise ConflictException(
                f"Something has already been derived about peer {name}; linking"
                + " after derivation isn't supported yet."
            )
        now = datetime.datetime.now(datetime.UTC)
        await db.execute(
            update(binding)
            .where(
                binding.tenant_id == models.scoped_tenant_id(),
                binding.id.in_([b.id for b in bindings]),
            )
            .values(peer_name=keep_name, confirmed_at=now)
        )
        # People already linked into this one now belong to the kept person,
        # so unlinking one of their accounts finds where it came from.
        for earlier in chained[name]:
            await _set_merged_into(db, workspace_name, earlier, keep_name)
        await _set_merged_into(db, workspace_name, name, keep_name)

    return schemas.IngestLinkResult(
        peer_id=keep_name,
        merged=merge_names,
        accounts=await _accounts(db, workspace_name, keep),
    )


async def unlink_account(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    account: schemas.IngestAccountRef,
) -> schemas.IngestUnlinkResult:
    """Map a linked account back to the person it came from, for the next import.

    Raises:
        ResourceNotFoundException: The peer, or the account under it, doesn't exist.
        ConflictException: The account isn't linked (it's the peer's own), its
            person is gone, or it already has imported records.
    """
    await take_workspace_fence(db, workspace_name, mode="exclusive")
    tenant_id = models.scoped_tenant_id()
    binding_model = models.IngestPeerBinding
    binding = await db.scalar(
        select(binding_model)
        .where(
            binding_model.tenant_id == tenant_id,
            binding_model.workspace_name == workspace_name,
            binding_model.source == account.source,
            binding_model.source_instance == account.source_instance,
            binding_model.key_kind == "external_id",
            binding_model.key_value == account.id,
            binding_model.peer_name == peer_name,
        )
        .with_for_update()
    )
    if binding is None:
        raise ResourceNotFoundException(
            f"Peer {peer_name} has no account {account.source}/"
            + f"{account.source_instance}/{account.id}"
        )
    if binding.confirmed_at is None:
        raise ConflictException(
            f"That account is {peer_name}'s own, not one linked into it"
        )
    peer = models.Peer
    origin = await db.scalar(
        select(peer)
        .where(
            peer.tenant_id == tenant_id,
            peer.workspace_name == workspace_name,
            peer.internal_metadata["ingest"]["binding_id"].astext == binding.id,
            peer.internal_metadata["ingest"]["merged_into"].astext == peer_name,
        )
        .with_for_update()
    )
    if origin is None:
        raise ConflictException(
            "The person this account was linked from no longer exists"
        )
    record = models.IngestRecord
    imported = await db.scalar(
        select(
            exists().where(
                record.tenant_id == tenant_id,
                record.workspace_name == workspace_name,
                or_(
                    record.author_binding_id == binding.id,
                    record.participant_binding_ids.contains([binding.id]),
                ),
            )
        )
    )
    if imported:
        raise ConflictException(
            "That account already has imported records."
            + RESET_FIRST.replace("link", "unlink")
        )
    await db.execute(
        update(binding_model)
        .where(binding_model.tenant_id == tenant_id, binding_model.id == binding.id)
        .values(peer_name=origin.name, confirmed_at=None)
    )
    await _set_merged_into(db, workspace_name, origin.name, None)
    return schemas.IngestUnlinkResult(
        peer_id=origin.name,
        account=schemas.IngestAccount(
            source=binding.source,
            source_instance=binding.source_instance,
            id=binding.key_value,
            display_name=binding.display_name,
        ),
    )


async def reset_import(db: AsyncSession, workspace_name: str) -> list[str]:
    """Delete every active session holding imported records.

    Marks each inactive and enqueues its deletion, as `DELETE /sessions/{id}`
    does; the deletion removes its messages, memberships, ingest records and
    session bindings. People, their configuration and their account bindings
    stay. Returns the sessions it enqueued.
    """
    await take_workspace_fence(db, workspace_name, mode="exclusive")
    tenant_id = models.scoped_tenant_id()
    session = models.Session
    record = models.IngestRecord
    sessions = (
        await db.scalars(
            select(session)
            .where(
                session.tenant_id == tenant_id,
                session.workspace_name == workspace_name,
                session.is_active.is_(True),
                session.name.in_(
                    select(record.session_name).where(
                        record.tenant_id == tenant_id,
                        record.workspace_name == workspace_name,
                    )
                ),
            )
            .order_by(session.name)
            .with_for_update()
        )
    ).all()
    for s in sessions:
        s.is_active = False
        await enqueue_deletion(
            workspace_name=workspace_name,
            deletion_type="session",
            resource_id=s.name,
            db_session=db,
        )
    return [s.name for s in sessions]


__all__ = ["link_peers", "reset_import", "unlink_account"]

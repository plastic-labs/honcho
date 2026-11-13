"""Ingest: turn source records into messages, mapped and deduplicated by the server.

This module is the ingest write transaction. In order:

1. Work out each record's identity and the sessions and peers it maps to (pure),
   and refuse the request over an explicit peer it can't use.
2. Classify records already ingested (duplicate, or rejected with a reason).
3. Take the ingest fence (src/crud/ingest_fence.py) for every session the batch
   writes, before any row lock, resolving session names under it.
4. Per session, in sorted order and inside a savepoint: bind the session and
   create it if missing.
5. Resolve every peer the batch names, in one sorted order. Sessions before
   peers is the message route's order too.
6. Per session, in sorted order and inside a savepoint: claim the records and add
   memberships. A refusal rolls back that savepoint and undoes the session step 4
   created, so it rejects only that session's records.
7. Insert the messages under each session's sequence lock, mark the records
   `pending` (or `deferred`), and commit.

Every set of row locks is taken in a sorted order, so two batches can't deadlock
each other. Turning `pending` records into queue work is a separate step (the
enqueuer), so this transaction never touches a queue table.
"""

import datetime
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Literal, NamedTuple, cast

from sqlalchemy import and_, delete, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.cache.client import safe_cache_delete
from src.crud.ingest_fence import take_session_fences, take_workspace_fence
from src.crud.message import insert_messages_locked
from src.crud.peer import peer_cache_key, reject_scope_observed
from src.crud.session import session_cache_key, upsert_session_peers
from src.exceptions import (
    ConflictException,
    ObserverException,
    ResourceNotFoundException,
    ValidationException,
)
from src.telemetry.events import PeerCreatedEvent, SessionCreatedEvent, emit

logger = getLogger(__name__)

# Suffix lengths, in hex characters, for generated session and peer names. The
# longer one is used when the shorter name already belongs to a resource that
# this binding didn't create.
_SHORT_SUFFIX = 10
_LONG_SUFFIX = 16
_SLUG_MAX = 48
_SLUG_INVALID = re.compile(r"[^a-z0-9_-]+")
_NAME_PATTERN = re.compile(schemas.RESOURCE_NAME_PATTERN)

IdentityKind = Literal["source_id", "content_hash"]
_Identity = tuple[str, str, str, str]  # (source, source_instance, kind, identity)


# ---------------------------------------------------------------------------
# Pure mapping
# ---------------------------------------------------------------------------


def slug(text: str | None, fallback: str) -> str:
    """Lowercase, runs outside [a-z0-9_-] collapsed to '-', trimmed, capped."""
    value = _SLUG_INVALID.sub("-", (text or "").lower()).strip("-")
    value = value[:_SLUG_MAX].strip("-")
    return value or fallback


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _digest(*parts: str) -> str:
    # Length-prefix each part so no two different part lists hash the same input.
    joined = "".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(joined.encode()).hexdigest()


def record_identity(record: schemas.IngestRecordCreate) -> tuple[IdentityKind, str]:
    """The record's dedup identity: its `source_id`, or a hash of its content."""
    if record.source_id is not None:
        return "source_id", record.source_id
    author = record.author
    return "content_hash", _digest(
        record.source,
        record.source_instance,
        record.container.id if record.container else "",
        # The same instant hashes the same whatever offset the source wrote.
        _comparable(record.created_at).astimezone(datetime.UTC).isoformat(),
        (author.id or author.name) if author else "",
        record.content,
    )


@dataclass(frozen=True, order=True)
class BindingKey:
    """An external identity that a binding maps to a session or peer name.

    `kind` is typed (container, standalone, external_id, provisional_name,
    source_peer), so an external value can never collide with a synthetic one.
    """

    source: str
    source_instance: str
    kind: str
    value: str
    # The display name the generated name is built from; not part of identity.
    first_name: str = field(default="", compare=False)

    def name(self, suffix: int) -> str:
        digest = _digest(self.source, self.source_instance, self.kind, self.value)
        return f"{slug(self.first_name, self.source)}-{digest[:suffix]}"

    def candidates(self) -> list[str]:
        return [self.name(_SHORT_SUFFIX), self.name(_LONG_SUFFIX)]


def session_key_for(
    record: schemas.IngestRecordCreate, identity: tuple[IdentityKind, str]
) -> BindingKey | str:
    """A record's session: an explicit name, or the binding key to map."""
    if record.session_id is not None:
        return record.session_id
    if record.container is not None:
        return BindingKey(
            record.source,
            record.source_instance,
            "container",
            record.container.id,
            record.container.name or record.container.id,
        )
    return BindingKey(
        record.source,
        record.source_instance,
        "standalone",
        f"{identity[0]}:{identity[1]}",
        record.source,
    )


def peer_key_for(
    record: schemas.IngestRecordCreate, person: schemas.IngestParticipant
) -> BindingKey:
    if person.id is not None:
        return BindingKey(
            record.source, record.source_instance, "external_id", person.id, person.name
        )
    # No ID: provisional, scoped to the container so the same name in two places
    # is never merged. The container ID is length-prefixed, so no container and
    # name pair can spell another pair's key.
    container = (
        f"container:{len(record.container.id)}:{record.container.id}"
        if record.container
        else "standalone"
    )
    return BindingKey(
        record.source,
        record.source_instance,
        "provisional_name",
        f"{container}:{person.name.casefold()}",
        person.name,
    )


def author_key_for(record: schemas.IngestRecordCreate) -> BindingKey | str:
    """A record's author: an explicit peer name, or the binding key to map."""
    if record.peer_id is not None:
        return record.peer_id
    if record.author is not None:
        return peer_key_for(record, record.author)
    return BindingKey(
        record.source,
        record.source_instance,
        "source_peer",
        "",
        f"{record.source}-source",
    )


def _comparable(value: datetime.datetime | None) -> datetime.datetime:
    # A null revision is the original, older than every timestamp.
    if value is None:
        return datetime.datetime.min.replace(tzinfo=datetime.UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value


def classify(
    stored_revised_at: datetime.datetime | None,
    stored_sha256: str,
    incoming_revised_at: datetime.datetime | None,
    incoming_sha256: str,
) -> tuple[Literal["duplicate", "rejected"], str | None]:
    """Classify a record against one already stored with the same identity."""
    stored = _comparable(stored_revised_at)
    incoming = _comparable(incoming_revised_at)
    if incoming < stored:
        return "rejected", "stale_revision"
    if incoming > stored:
        return "rejected", "edit_not_supported"
    if incoming_sha256 == stored_sha256:
        return "duplicate", None
    return "rejected", "content_conflict"


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


@dataclass
class _Record:
    index: int
    record: schemas.IngestRecordCreate
    identity_kind: IdentityKind
    identity: str
    sha256: str
    session_key: BindingKey | str
    author_key: BindingKey | str
    participant_keys: list[BindingKey]
    session_name: str = ""
    author_name: str = ""
    record_id: str = ""

    @property
    def identity_tuple(self) -> _Identity:
        return (
            self.record.source,
            self.record.source_instance,
            self.identity_kind,
            self.identity,
        )

    def peer_keys(self) -> list[BindingKey | str]:
        return [self.author_key, *self.participant_keys]


@dataclass
class IngestResult:
    """The per-record outcomes, plus what the caller does after commit."""

    results: list[schemas.IngestRecordResult]
    # Message public ids and token totals created, per session.
    created_messages: dict[str, list[str]] = field(default_factory=dict)
    created_token_counts: dict[str, int] = field(default_factory=dict)
    created_sessions: list[str] = field(default_factory=list)
    created_peers: list[str] = field(default_factory=list)
    # Authors of the created messages, whose pending dreams are cancelled.
    authors: set[str] = field(default_factory=set)


class IngestRaced(Exception):
    """A concurrent batch bound one of this batch's sources first; retry it whole."""


class _Refused(Exception):
    """A record, a peer or a session group can't be written, with the reason."""

    reason: str

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _StoredRecord:
    id: str
    revised_at: datetime.datetime | None
    content_sha256: str
    derivation: str
    message_public_id: str | None
    session_name: str
    peer_name: str | None


def _prepare(batch: schemas.IngestBatch) -> list[_Record]:
    # One binding key per external identity across the batch: the first record
    # that names it supplies the display name, so a rename inside one batch still
    # maps every record to the same session or peer.
    canonical: dict[BindingKey, BindingKey] = {}

    def first_seen(key: BindingKey) -> BindingKey:
        return canonical.setdefault(key, key)

    def first_seen_any(key: BindingKey | str) -> BindingKey | str:
        return first_seen(key) if isinstance(key, BindingKey) else key

    prepared: list[_Record] = []
    for index, record in enumerate(batch.records):
        identity_kind, identity = record_identity(record)
        prepared.append(
            _Record(
                index=index,
                record=record,
                identity_kind=identity_kind,
                identity=identity,
                sha256=content_sha256(record.content),
                session_key=first_seen_any(
                    session_key_for(record, (identity_kind, identity))
                ),
                author_key=first_seen_any(author_key_for(record)),
                participant_keys=[
                    first_seen(peer_key_for(record, p)) for p in record.participants
                ],
            )
        )
    return prepared


def _rejected(index: int, reason: str) -> schemas.IngestRecordResult:
    return schemas.IngestRecordResult(index=index, outcome="rejected", reason=reason)


def _reject_all(
    results: dict[int, schemas.IngestRecordResult],
    records: Sequence[_Record],
    reason: str,
) -> None:
    # A record that already has an outcome (a concurrent duplicate) keeps it.
    for r in records:
        results.setdefault(r.index, _rejected(r.index, reason))


async def ingest_batch(
    db: AsyncSession, workspace_name: str, batch: schemas.IngestBatch
) -> IngestResult:
    """Run the ingest write transaction for one batch and commit it.

    Raises:
        ValidationException: An explicit `peer_id` is a scope, is a reserved scope
            name that doesn't exist, or doesn't exist and isn't a valid new name.
        ResourceNotFoundException: The workspace doesn't exist.
        ConflictException: The workspace isn't an auto workspace.
        DBAPIError: A deadlock, serialization failure or lock timeout.
        IngestRaced: A concurrent batch bound one of this batch's sources first.
            For these two, nothing was committed and the caller retries the whole
            batch, which is safe because records dedup on their identity.
    """
    tenant_id = models.scoped_tenant_id()
    prepared = _prepare(batch)
    await _validate_explicit_peers(db, tenant_id, workspace_name, prepared)
    results: dict[int, schemas.IngestRecordResult] = {}
    outcome = IngestResult(results=[])
    # The stored row each identity was classified against, so a repeat inside the
    # batch is classified against that row too.
    basis: dict[_Identity, _StoredRecord] = {}

    # A repeated identity inside the batch: the first occurrence is processed and
    # later ones are classified once it has an outcome.
    first_by_identity: dict[_Identity, _Record] = {}
    repeats: list[tuple[_Record, _Record]] = []
    unique: list[_Record] = []
    for r in prepared:
        first = first_by_identity.setdefault(r.identity_tuple, r)
        if first is r:
            unique.append(r)
        else:
            repeats.append((r, first))

    # Records already stored are classified and take no further part. Read before
    # the fence only to narrow it, and again under it.
    remaining = await _classify_stored(
        db, tenant_id, workspace_name, unique, results, basis
    )

    # The fence, before any row lock: shared on the workspace, then every session
    # this batch writes.
    await take_workspace_fence(db, workspace_name, mode="shared")
    session_names = await _fence_sessions(db, tenant_id, workspace_name, remaining)

    # Re-read under the fence: a workspace deleted while we waited is gone now.
    workspace = (
        await db.execute(
            select(models.Workspace.id, models.Workspace.auto_enabled_at).where(
                models.Workspace.tenant_id == tenant_id,
                models.Workspace.name == workspace_name,
            )
        )
    ).one_or_none()
    if workspace is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")
    if workspace.auto_enabled_at is None:
        raise ConflictException(
            f"Workspace {workspace_name} isn't an auto workspace. Ingest writes"
            + " only to auto workspaces: create one with auto: true."
        )
    remaining = await _classify_stored(
        db, tenant_id, workspace_name, remaining, results, basis
    )

    groups: dict[str, list[_Record]] = {}
    for r in remaining:
        r.session_name = session_names[r.session_key]
        groups.setdefault(r.session_name, []).append(r)

    newest_names = _newest_display_names(remaining)
    parents = _first_parents(remaining)

    # 1. Sessions and their bindings, in sorted order, each in a savepoint. Sessions
    # come before peers, as on the message route, so the two can't cycle on a new
    # session and a new peer.
    created_here: dict[str, _CreatedSession] = {}
    for session_name in sorted(groups):
        group = groups[session_name]
        keys = list(
            dict.fromkeys(
                r.session_key for r in group if isinstance(r.session_key, BindingKey)
            )
        )
        await db.flush()
        savepoint = await db.begin_nested()
        try:
            created = await _ensure_session(
                db, tenant_id, workspace_name, session_name, keys
            )
            if created.untrusted:
                group = await _drop_untrusted(db, tenant_id, created, group, results)
                groups[session_name] = group
            for key, binding_id in created.bound.items():
                if key in newest_names:
                    await _note_display_name(
                        db,
                        models.IngestSessionBinding,
                        tenant_id,
                        binding_id,
                        newest_names[key],
                    )
                if key in parents:
                    await _note_parent(db, tenant_id, binding_id, parents[key])
            created_here[session_name] = created
        except _Refused as refused:
            await savepoint.rollback()
            _reject_all(results, group, refused.reason)
            del groups[session_name]
            continue
        await savepoint.commit()

    # 2. Every peer the remaining records name, in one sorted order. A peer that
    # can't be used rejects the records that name it.
    peer_names: dict[BindingKey | str, str | _Refused] = {}
    peer_bindings: dict[BindingKey, str] = {}
    for key in sorted(
        {k for group in groups.values() for r in group for k in r.peer_keys()},
        key=_peer_sort_key,
    ):
        try:
            peer_names[key], binding_id = await _ensure_peer(
                db, tenant_id, workspace_name, key, outcome
            )
        except _Refused as refused:
            peer_names[key] = refused
            continue
        if isinstance(key, BindingKey) and binding_id is not None:
            peer_bindings[key] = binding_id
            if key in newest_names:
                await _note_display_name(
                    db,
                    models.IngestPeerBinding,
                    tenant_id,
                    binding_id,
                    newest_names[key],
                )

    # 3. Per session, in sorted order and in a savepoint: claim the records and add
    # memberships. A session that ends up with no records is undone.
    accepted: dict[str, list[_Record]] = {}
    for session_name in sorted(groups):
        group: list[_Record] = []
        for r in groups[session_name]:
            refusal = next(
                (
                    n
                    for n in (peer_names[k] for k in r.peer_keys())
                    if isinstance(n, _Refused)
                ),
                None,
            )
            if refusal is not None:
                results[r.index] = _rejected(r.index, refusal.reason)
                continue
            author = peer_names[r.author_key]
            assert isinstance(author, str)
            r.author_name = author
            group.append(r)
        await db.flush()
        savepoint = await db.begin_nested()
        try:
            live = await _write_group(
                db,
                tenant_id,
                workspace_name,
                session_name,
                group,
                peer_names,
                peer_bindings,
                results,
                basis,
            )
        except _Refused as refused:
            await savepoint.rollback()
            _reject_all(results, group, refused.reason)
            live = []
        else:
            await savepoint.commit()
        created = created_here[session_name]
        if live:
            accepted[session_name] = live
            if created.session:
                outcome.created_sessions.append(session_name)
        else:
            await _undo_session(db, tenant_id, workspace_name, session_name, created)

    # 4. Messages, under each session's sequence lock, in sorted session order.
    derivation: Literal["pending", "deferred"] = (
        "deferred" if batch.derive == "later" else "pending"
    )
    for session_name in sorted(accepted):
        group = sorted(
            accepted[session_name],
            key=lambda r: (_comparable(r.record.created_at), r.identity),
        )
        created = await insert_messages_locked(
            db,
            [
                schemas.MessageCreate.model_validate(
                    {
                        "content": r.record.content,
                        "peer_id": r.author_name,
                        "metadata": r.record.metadata,
                        "configuration": r.record.configuration,
                        "created_at": r.record.created_at,
                    }
                )
                for r in group
            ],
            workspace_name,
            session_name,
        )
        await db.flush()
        for r, message in zip(group, created, strict=True):
            await db.execute(
                update(models.IngestRecord)
                .where(
                    models.IngestRecord.tenant_id == tenant_id,
                    models.IngestRecord.id == r.record_id,
                )
                .values(message_public_id=message.public_id, derivation=derivation)
            )
            results[r.index] = schemas.IngestRecordResult(
                index=r.index,
                outcome="created",
                record_id=r.record_id,
                message_id=message.public_id,
                session_id=session_name,
                peer_id=r.author_name,
                derivation=derivation,
            )
            outcome.authors.add(r.author_name)
        outcome.created_messages[session_name] = [m.public_id for m in created]
        outcome.created_token_counts[session_name] = sum(m.token_count for m in created)

    await db.commit()

    for r, first in repeats:
        results[r.index] = _classified_repeat(r, first, results[first.index], basis)
    outcome.results = [results[i] for i in range(len(prepared))]
    return outcome


async def _validate_explicit_peers(
    db: AsyncSession, tenant_id: str, workspace_name: str, records: Sequence[_Record]
) -> None:
    """Refuse the whole request over an explicit `peer_id` ingest can't use.

    As on the message route: a scope can't author, and a reserved scope name that
    doesn't exist yet can't be created (it may become a scope later). A peer that
    merely has a reserved-looking name and isn't a scope works as usual.
    """
    names = sorted({r.author_key for r in records if isinstance(r.author_key, str)})
    if not names:
        return
    await reject_scope_observed(
        db, workspace_name, names, action="Scope peers can't author ingested records."
    )
    existing = set(
        (
            await db.execute(
                select(models.Peer.name).where(
                    models.Peer.tenant_id == tenant_id,
                    models.Peer.workspace_name == workspace_name,
                    models.Peer.name.in_(names),
                )
            )
        ).scalars()
    )
    invalid = [n for n in names if n not in existing and not _NAME_PATTERN.fullmatch(n)]
    if invalid:
        raise ValidationException(
            f"Peer id(s) {invalid} don't exist and can't be created: a new peer id"
            + " may contain only letters, digits, '-' and '_'."
        )


async def _classify_stored(
    db: AsyncSession,
    tenant_id: str,
    workspace_name: str,
    records: Sequence[_Record],
    results: dict[int, schemas.IngestRecordResult],
    basis: dict[_Identity, _StoredRecord],
) -> list[_Record]:
    """Classify the records already stored; return the ones that aren't."""
    stored = await _stored_records(db, tenant_id, workspace_name, records)
    remaining: list[_Record] = []
    for r in records:
        row = stored.get(r.identity_tuple)
        if row is None:
            remaining.append(r)
        else:
            basis[r.identity_tuple] = row
            results[r.index] = _classified(r, row)
    return remaining


async def _fence_sessions(
    db: AsyncSession, tenant_id: str, workspace_name: str, records: Sequence[_Record]
) -> dict[BindingKey | str, str]:
    """Fence every session the records write; return each session key's name.

    A key's session is resolved under the fence, since a binding can change while
    the batch waits for it. Before the fence, read without locks, every name a key
    could resolve to is fenced: its bound name and both candidates. If a name
    resolved under the fence isn't fenced yet, it is fenced and every key is
    resolved again, until nothing changes.
    """
    keys = list(dict.fromkeys(r.session_key for r in records))
    wanted: set[str] = set()
    for key in keys:
        if isinstance(key, str):
            wanted.add(key)
            continue
        wanted.update(key.candidates())
        bound = await _binding(
            db, models.IngestSessionBinding, tenant_id, workspace_name, key
        )
        if bound is not None:
            wanted.add(bound[1])
    fenced: set[str] = set()
    while True:
        await take_session_fences(db, workspace_name, wanted - fenced)
        fenced |= wanted
        names = _separate_collisions(
            {
                key: await _session_name(db, tenant_id, workspace_name, key)
                for key in keys
            }
        )
        unfenced = set(names.values()) - fenced
        if not unfenced:
            return names
        wanted = unfenced


_BindingModel = type[models.IngestSessionBinding] | type[models.IngestPeerBinding]


# The same display name is stamped again only when this much newer, so a busy
# author's binding isn't rewritten (and row-locked) by every batch. The cost: a
# record carrying an older name, arriving late and less than this much older than
# the newest record seen, can bring the older name back until the next one
# arrives. Display names are only shown, never matched on, so that is accepted.
_RESTAMP_AFTER = datetime.timedelta(days=1)


def _newest_display_names(
    records: Sequence[_Record],
) -> dict[BindingKey, tuple[datetime.datetime, str]]:
    """Per binding key, the display name of the newest record that carried one."""
    newest: dict[BindingKey, tuple[datetime.datetime, str]] = {}

    def see(key: BindingKey | str, name: str | None, at: datetime.datetime) -> None:
        if not isinstance(key, BindingKey) or not name:
            return
        if key not in newest or newest[key][0] < at:
            newest[key] = (at, name)

    for r in records:
        at = _comparable(r.record.created_at)
        if r.record.container is not None:
            see(r.session_key, r.record.container.name, at)
        if r.record.author is not None:
            see(r.author_key, r.record.author.name, at)
        for key, person in zip(r.participant_keys, r.record.participants, strict=True):
            see(key, person.name, at)
    return newest


def _source_context(record: schemas.IngestRecordCreate) -> dict[str, Any]:
    """A record's container, author and participants as received."""
    return {
        "container": record.container.model_dump(mode="json")
        if record.container is not None
        else None,
        "author": record.author.model_dump(mode="json")
        if record.author is not None
        else None,
        "participants": [p.model_dump(mode="json") for p in record.participants],
    }


def _first_parents(records: Sequence[_Record]) -> dict[BindingKey, str]:
    """Per container binding key, the parent ID of its oldest record that sent one."""
    first: dict[BindingKey, tuple[datetime.datetime, str]] = {}
    for r in records:
        container = r.record.container
        if (
            container is None
            or container.parent_id is None
            or not isinstance(r.session_key, BindingKey)
        ):
            continue
        at = _comparable(r.record.created_at)
        if r.session_key not in first or at < first[r.session_key][0]:
            first[r.session_key] = (at, container.parent_id)
    return {key: parent for key, (_, parent) in first.items()}


async def _note_parent(
    db: AsyncSession, tenant_id: str, binding_id: str, parent: str
) -> None:
    """Record a container's parent if its binding has none yet.

    Set once and never overwritten. A row that already has one isn't written,
    so it isn't locked.
    """
    await db.execute(
        update(models.IngestSessionBinding)
        .where(
            models.IngestSessionBinding.tenant_id == tenant_id,
            models.IngestSessionBinding.id == binding_id,
            models.IngestSessionBinding.parent_key_value.is_(None),
        )
        .values(parent_key_value=parent)
    )


async def _note_display_name(
    db: AsyncSession,
    model: _BindingModel,
    tenant_id: str,
    binding_id: str,
    newest: tuple[datetime.datetime, str],
) -> None:
    """Record a binding's display name if it is newer than the one stored.

    A row the predicate doesn't match isn't written, so it isn't locked.
    """
    at, name = newest
    await db.execute(
        update(model)
        .where(
            model.tenant_id == tenant_id,
            model.id == binding_id,
            or_(
                model.display_name_at.is_(None),
                and_(
                    model.display_name_at < at,
                    model.display_name.is_distinct_from(name),
                ),
                model.display_name_at < at - _RESTAMP_AFTER,
            ),
        )
        .values(display_name=name, display_name_at=at)
    )


def _peer_sort_key(key: BindingKey | str) -> tuple[int, str, str, str, str]:
    """Explicit names first, in name order; then binding keys, by identity.

    Explicit names are the ones another route can also create, so they go in
    the name order every multi-peer insert uses (CLAUDE.md). Binding keys are
    ordered by their identity, not the names they generate: a generated name
    contains a display name that isn't part of the identity, so two batches
    could order the same keys differently and invert their binding-row locks.
    No other route can name a generated peer before ingest creates it.
    """
    if isinstance(key, str):
        return (0, key, "", "", "")
    return (1, key.source, key.source_instance, key.kind, key.value)


async def _stored_records(
    db: AsyncSession, tenant_id: str, workspace_name: str, records: Sequence[_Record]
) -> dict[_Identity, _StoredRecord]:
    if not records:
        return {}
    table = models.IngestRecord
    rows = await db.execute(
        select(
            table.source,
            table.source_instance,
            table.identity_kind,
            table.identity,
            table.id,
            table.revised_at,
            table.content_sha256,
            table.derivation,
            table.message_public_id,
            table.session_name,
            models.Message.peer_name,
        )
        .outerjoin(
            models.Message,
            (models.Message.tenant_id == table.tenant_id)
            & (models.Message.public_id == table.message_public_id),
        )
        .where(
            table.tenant_id == tenant_id,
            table.workspace_name == workspace_name,
            tuple_(
                table.source, table.source_instance, table.identity_kind, table.identity
            ).in_([r.identity_tuple for r in records]),
        )
    )
    return {
        (row[0], row[1], row[2], row[3]): _StoredRecord(*row[4:]) for row in rows.all()
    }


def _classified(r: _Record, row: _StoredRecord) -> schemas.IngestRecordResult:
    verdict, reason = classify(
        row.revised_at, row.content_sha256, r.record.revised_at, r.sha256
    )
    if verdict == "rejected":
        return _rejected(r.index, reason or "rejected")
    return schemas.IngestRecordResult(
        index=r.index,
        outcome="duplicate",
        record_id=row.id,
        message_id=row.message_public_id,
        session_id=row.session_name,
        peer_id=row.peer_name,
        derivation=row.derivation,  # pyright: ignore[reportArgumentType]
    )


def _classified_repeat(
    r: _Record,
    first: _Record,
    first_result: schemas.IngestRecordResult,
    basis: dict[_Identity, _StoredRecord],
) -> schemas.IngestRecordResult:
    # The first occurrence was classified against a stored row: so is the repeat.
    row = basis.get(r.identity_tuple)
    if row is not None:
        return _classified(r, row)
    verdict, reason = classify(
        first.record.revised_at, first.sha256, r.record.revised_at, r.sha256
    )
    if verdict == "rejected":
        return _rejected(r.index, reason or "rejected")
    if first_result.outcome == "rejected":
        return _rejected(r.index, first_result.reason or "rejected")
    return first_result.model_copy(update={"index": r.index, "outcome": "duplicate"})


# ---------------------------------------------------------------------------
# Bindings, sessions, peers
# ---------------------------------------------------------------------------


def _provenance(binding_id: str, key: BindingKey) -> dict[str, Any]:
    return {
        "ingest": {
            "binding_id": binding_id,
            "source": key.source,
            "source_instance": key.source_instance,
            "key_kind": key.kind,
            "key_value": key.value,
            "provisional": key.kind == "provisional_name",
        }
    }


def _owned_by(internal_metadata: dict[str, Any] | None, binding_id: str) -> bool:
    ingest: Any = (internal_metadata or {}).get("ingest")
    return (
        isinstance(ingest, dict)
        and cast(dict[str, Any], ingest).get("binding_id") == binding_id
    )


class _Bound(NamedTuple):
    """A binding: its id, the name it points at, and whether a person confirmed it."""

    id: str
    name: str
    confirmed: bool


def _trusted(
    bound: _Bound, target_metadata: dict[str, Any] | None, target_exists: bool
) -> bool:
    """Whether a binding's target is really its own.

    Ingest created it for this binding, or a person pointed the binding at it (a
    confirmed link or move). Anything else is stale.
    """
    return target_exists and (bound.confirmed or _owned_by(target_metadata, bound.id))


async def _binding(
    db: AsyncSession,
    model: _BindingModel,
    tenant_id: str,
    workspace_name: str,
    key: BindingKey,
    *,
    for_update: bool = False,
) -> _Bound | None:
    """The existing binding for a key, or None."""
    name_column = (
        models.IngestSessionBinding.session_name
        if model is models.IngestSessionBinding
        else models.IngestPeerBinding.peer_name
    )
    statement = select(model.id, name_column, model.confirmed_at).where(
        model.tenant_id == tenant_id,
        model.workspace_name == workspace_name,
        model.source == key.source,
        model.source_instance == key.source_instance,
        model.key_kind == key.kind,
        model.key_value == key.value,
    )
    if for_update:
        statement = statement.with_for_update()
    row = (await db.execute(statement)).one_or_none()
    return _Bound(row[0], row[1], row[2] is not None) if row is not None else None


async def _bind(
    db: AsyncSession,
    model: _BindingModel,
    tenant_id: str,
    workspace_name: str,
    key: BindingKey,
    name: str,
) -> tuple[_Bound, bool]:
    """The binding for a key, created pointing at `name` if missing.

    Returns the binding and whether this call created it.
    """
    if model is models.IngestSessionBinding:
        constraint, name_field = "uq_ingest_session_bindings_key", "session_name"
    else:
        constraint, name_field = "uq_ingest_peer_bindings_key", "peer_name"
    inserted = await db.scalar(
        pg_insert(model)
        .values(
            workspace_name=workspace_name,
            source=key.source,
            source_instance=key.source_instance,
            key_kind=key.kind,
            key_value=key.value,
            **{name_field: name},
        )
        .on_conflict_do_nothing(constraint=constraint)
        .returning(model.id)
    )
    bound = await _binding(db, model, tenant_id, workspace_name, key)
    assert bound is not None
    return bound, inserted is not None


async def _rebind(
    db: AsyncSession, model: _BindingModel, tenant_id: str, binding_id: str, name: str
) -> None:
    """Point a binding at another name. Ingest's choice, so not confirmed."""
    values: dict[str, Any] = {"confirmed_at": None}
    if model is models.IngestSessionBinding:
        values["session_name"] = name
    else:
        values["peer_name"] = name
    await db.execute(
        update(model)
        .where(model.tenant_id == tenant_id, model.id == binding_id)
        .values(**values)
    )


@dataclass
class _CreatedSession:
    """What step 1 created for a session, so it can be undone."""

    session: bool = False
    binding_ids: list[str] = field(default_factory=list)
    # Every binding of the group's keys, created or not.
    bound: dict[BindingKey, str] = field(default_factory=dict)
    # Keys whose binding can't use the existing session: ingest didn't create the
    # session for that binding and no person pointed it there.
    untrusted: set[BindingKey] = field(default_factory=set)


async def _session_row(
    db: AsyncSession, tenant_id: str, workspace_name: str, session_name: str
) -> tuple[bool, dict[str, Any] | None] | None:
    """(is_active, internal_metadata) of a session, or None if it doesn't exist."""
    row = (
        await db.execute(
            select(models.Session.is_active, models.Session.internal_metadata).where(
                models.Session.tenant_id == tenant_id,
                models.Session.workspace_name == workspace_name,
                models.Session.name == session_name,
            )
        )
    ).one_or_none()
    return (row[0], row[1]) if row is not None else None


def _separate_collisions(
    resolved: dict[BindingKey | str, tuple[str, bool]],
) -> dict[BindingKey | str, str]:
    """Give different new containers that resolve to one short name their own.

    `resolved` maps each key to (name, whether the name was newly generated). Two
    keys get the same new short name only when their digests collide and neither
    session exists yet (an existing one sends the other key to its long name).
    The first key in sorted order keeps it; the others take their long name. A
    name a key is already bound to (its own session, or one a person confirmed,
    which several keys may share) is never moved.
    """
    by_name: dict[str, list[BindingKey]] = {}
    for key, (name, new) in resolved.items():
        if isinstance(key, BindingKey) and new and name == key.candidates()[0]:
            by_name.setdefault(name, []).append(key)
    names = {key: name for key, (name, _) in resolved.items()}
    for colliding in by_name.values():
        for key in sorted(colliding)[1:]:
            names[key] = key.candidates()[1]
    return names


async def _session_name(
    db: AsyncSession, tenant_id: str, workspace_name: str, key: BindingKey | str
) -> tuple[str, bool]:
    """The session a record goes to, and whether the name is newly generated.

    Read-only; step 1 binds it.
    """
    if isinstance(key, str):
        return key, False
    bound = await _binding(
        db, models.IngestSessionBinding, tenant_id, workspace_name, key
    )
    binding_id: str | None = None
    if bound is not None:
        binding_id = bound.id
        row = await _session_row(db, tenant_id, workspace_name, bound.name)
        if _trusted(bound, row[1] if row else None, row is not None):
            return bound.name, False
        # Stale: the bound session is gone (or its name now belongs to another
        # session). Resolved afresh, as for a first sighting; step 4 repoints it.
    # The short name, unless a session this binding didn't create already has it.
    short, long = key.candidates()
    row = await _session_row(db, tenant_id, workspace_name, short)
    if row is None or (binding_id is not None and _owned_by(row[1], binding_id)):
        return short, True
    return long, True


async def _ensure_session(
    db: AsyncSession,
    tenant_id: str,
    workspace_name: str,
    session_name: str,
    keys: list[BindingKey],
) -> _CreatedSession:
    """Bind the group's keys and create its session if missing."""
    created = _CreatedSession()
    binding_id: str | None = None
    provenance: dict[str, Any] = {}
    bindings: list[_Bound] = []
    ordered_keys = sorted(keys)
    for key in ordered_keys:
        bound, inserted = await _bind(
            db,
            models.IngestSessionBinding,
            tenant_id,
            workspace_name,
            key,
            session_name,
        )
        created.bound[key] = bound.id
        if inserted:
            created.binding_ids.append(bound.id)
        elif bound.name != session_name:
            row = await _session_row(db, tenant_id, workspace_name, bound.name)
            if _trusted(bound, row[1] if row else None, row is not None):
                # Bound (or confirmed) by someone else after this batch resolved
                # its names.
                raise IngestRaced
            # Stale: its session is gone. Point it at this one.
            await _rebind(
                db, models.IngestSessionBinding, tenant_id, bound.id, session_name
            )
            bound = _Bound(bound.id, session_name, False)
        bindings.append(bound)
        if binding_id is None:
            binding_id, provenance = bound.id, _provenance(bound.id, key)
    inserted_id = await db.scalar(
        pg_insert(models.Session)
        .values(
            workspace_name=workspace_name,
            name=session_name,
            internal_metadata=provenance,
        )
        .on_conflict_do_nothing()
        .returning(models.Session.id)
    )
    if inserted_id is not None:
        created.session = True
        internal_metadata: dict[str, Any] | None = provenance
    else:
        row = await _session_row(db, tenant_id, workspace_name, session_name)
        assert row is not None
        is_active, internal_metadata = row
        if not is_active:
            raise _Refused("session_inactive")
    # Each binding must be able to use the session on its own: ingest created the
    # session for that binding (the one named in its provenance), or a person
    # pointed that binding at it. Another binding's ownership or confirmation
    # doesn't extend to it, so ingest never adopts a session by name.
    created.untrusted = {
        key
        for key, bound in zip(ordered_keys, bindings, strict=True)
        if not _trusted(bound, internal_metadata, True)
    }
    return created


async def _drop_untrusted(
    db: AsyncSession,
    tenant_id: str,
    created: _CreatedSession,
    group: list[_Record],
    results: dict[int, schemas.IngestRecordResult],
) -> list[_Record]:
    """Reject the records whose binding can't use the session; return the rest.

    Bindings this batch created for them are removed: they point at a session
    their records can't go to.

    Raises:
        _Refused: No record of the group is left.
    """
    kept = [r for r in group if r.session_key not in created.untrusted]
    if not kept:
        raise _Refused("name_conflict")
    for r in group:
        if r.session_key in created.untrusted:
            results[r.index] = _rejected(r.index, "name_conflict")
    dropped = [
        created.bound.pop(key)
        for key in sorted(created.untrusted)
        if created.bound[key] in created.binding_ids
    ]
    for key in created.untrusted:
        created.bound.pop(key, None)
    if dropped:
        await db.execute(
            delete(models.IngestSessionBinding).where(
                models.IngestSessionBinding.tenant_id == tenant_id,
                models.IngestSessionBinding.id.in_(dropped),
            )
        )
        created.binding_ids = [b for b in created.binding_ids if b not in dropped]
    return kept


async def _undo_session(
    db: AsyncSession,
    tenant_id: str,
    workspace_name: str,
    session_name: str,
    created: _CreatedSession,
) -> None:
    """Delete the session and bindings step 1 created, when nothing landed in it.

    Only this transaction's own uncommitted rows, so this takes no new locks.
    """
    if created.binding_ids:
        await db.execute(
            delete(models.IngestSessionBinding).where(
                models.IngestSessionBinding.tenant_id == tenant_id,
                models.IngestSessionBinding.id.in_(created.binding_ids),
            )
        )
    if created.session:
        await db.execute(
            delete(models.Session).where(
                models.Session.tenant_id == tenant_id,
                models.Session.workspace_name == workspace_name,
                models.Session.name == session_name,
            )
        )


async def _peer_metadata(
    db: AsyncSession, tenant_id: str, workspace_name: str, peer_name: str
) -> tuple[bool, dict[str, Any] | None]:
    """(exists, internal_metadata) of a peer."""
    row = (
        await db.execute(
            select(models.Peer.internal_metadata).where(
                models.Peer.tenant_id == tenant_id,
                models.Peer.workspace_name == workspace_name,
                models.Peer.name == peer_name,
            )
        )
    ).one_or_none()
    return (row is not None, row[0] if row is not None else None)


async def _ensure_peer(
    db: AsyncSession,
    tenant_id: str,
    workspace_name: str,
    key: BindingKey | str,
    outcome: IngestResult,
) -> tuple[str, str | None]:
    """The peer for a key, created if needed. Returns its name and binding."""
    if isinstance(key, str):
        # Checked up front (_validate_explicit_peers): not a scope, and a name
        # that doesn't exist yet is a valid new one.
        created = await db.scalar(
            pg_insert(models.Peer)
            .values(workspace_name=workspace_name, name=key)
            .on_conflict_do_nothing()
            .returning(models.Peer.id)
        )
        if created is not None:
            outcome.created_peers.append(key)
        return key, None

    short, long = key.candidates()
    bound = await _binding(db, models.IngestPeerBinding, tenant_id, workspace_name, key)
    if bound is not None:
        exists, metadata = await _peer_metadata(
            db, tenant_id, workspace_name, bound.name
        )
        if _trusted(bound, metadata, exists):
            return bound.name, bound.id
        # Stale: the bound peer is gone (or its name now belongs to another peer).
        # Lock the binding and look again, so two batches recovering it one after
        # the other agree on one peer; then resolve afresh, as for a first sighting.
        bound = await _binding(
            db,
            models.IngestPeerBinding,
            tenant_id,
            workspace_name,
            key,
            for_update=True,
        )
        assert bound is not None
        exists, metadata = await _peer_metadata(
            db, tenant_id, workspace_name, bound.name
        )
        if _trusted(bound, metadata, exists):
            return bound.name, bound.id
    else:
        bound, inserted = await _bind(
            db, models.IngestPeerBinding, tenant_id, workspace_name, key, short
        )
        if not inserted:
            # A concurrent batch bound this key first and has committed.
            exists, metadata = await _peer_metadata(
                db, tenant_id, workspace_name, bound.name
            )
            if _trusted(bound, metadata, exists):
                return bound.name, bound.id
            raise IngestRaced
    binding_id, name = bound.id, bound.name

    # The first candidate that is free, or already this binding's own peer.
    for candidate in (short, long):
        exists, metadata = await _peer_metadata(
            db, tenant_id, workspace_name, candidate
        )
        if exists and not _owned_by(metadata, binding_id):
            continue
        if candidate != name:
            await _rebind(
                db, models.IngestPeerBinding, tenant_id, binding_id, candidate
            )
            name = candidate
        if exists:
            return candidate, binding_id
        created = await db.scalar(
            pg_insert(models.Peer)
            .values(
                workspace_name=workspace_name,
                name=candidate,
                internal_metadata=_provenance(binding_id, key),
            )
            .on_conflict_do_nothing()
            .returning(models.Peer.id)
        )
        if created is not None:
            outcome.created_peers.append(candidate)
            return candidate, binding_id
        # Created meanwhile: by a concurrent batch recovering the same stale
        # binding (then it is this binding's peer), or by someone else.
        exists, metadata = await _peer_metadata(
            db, tenant_id, workspace_name, candidate
        )
        if exists and _owned_by(metadata, binding_id):
            return candidate, binding_id
    raise _Refused("name_conflict")


async def _write_group(
    db: AsyncSession,
    tenant_id: str,
    workspace_name: str,
    session_name: str,
    group: list[_Record],
    peer_names: dict[BindingKey | str, str | _Refused],
    peer_bindings: dict[BindingKey, str],
    results: dict[int, schemas.IngestRecordResult],
    basis: dict[_Identity, _StoredRecord],
) -> list[_Record]:
    """Claim a session group's records and add its memberships.

    Returns the group's records that are still to be written.
    """
    # Claim, in sorted identity order. A conflict is a concurrent duplicate.
    for r in sorted(group, key=lambda r: r.identity_tuple):
        configuration = r.record.configuration
        claimed = await db.scalar(
            pg_insert(models.IngestRecord)
            .values(
                workspace_name=workspace_name,
                source=r.record.source,
                source_instance=r.record.source_instance,
                identity_kind=r.identity_kind,
                identity=r.identity,
                session_name=session_name,
                content_sha256=r.sha256,
                revised_at=r.record.revised_at,
                configuration=configuration.model_dump(exclude_none=True)
                if configuration is not None
                else None,
                reasoning_disabled=bool(
                    configuration is not None
                    and configuration.reasoning is not None
                    and configuration.reasoning.enabled is False
                ),
                derivation="pending",
                author_binding_id=peer_bindings.get(r.author_key)
                if isinstance(r.author_key, BindingKey)
                else None,
                participant_binding_ids=list(
                    dict.fromkeys(peer_bindings[k] for k in r.participant_keys)
                )
                or None,
                source_context=_source_context(r.record),
            )
            .on_conflict_do_nothing(constraint="uq_ingest_records_identity")
            .returning(models.IngestRecord.id)
        )
        if claimed is None:
            stored = await _stored_records(db, tenant_id, workspace_name, [r])
            row = stored[r.identity_tuple]
            basis[r.identity_tuple] = row
            results[r.index] = _classified(r, row)
        else:
            r.record_id = claimed
    live = [r for r in group if r.index not in results]
    if not live:
        return []

    members = sorted(
        {
            name
            for r in live
            for name in (peer_names[k] for k in r.peer_keys())
            if isinstance(name, str)
        }
    )
    try:
        await upsert_session_peers(
            db,
            workspace_name,
            session_name,
            {name: schemas.SessionPeerConfig() for name in members},
            fetch_after_upsert=False,
        )
    except ObserverException:
        raise _Refused("observers_limit") from None
    return live


async def after_commit(workspace_name: str, outcome: IngestResult) -> None:
    """Cache invalidation and creation events, once the batch has committed."""
    for session_name in outcome.created_sessions:
        await safe_cache_delete(session_cache_key(workspace_name, session_name))
        emit(
            SessionCreatedEvent(
                workspace_name=workspace_name, session_name=session_name
            )
        )
    for peer_name in dict.fromkeys(outcome.created_peers):
        await safe_cache_delete(peer_cache_key(workspace_name, peer_name))
        emit(PeerCreatedEvent(workspace_name=workspace_name, peer_name=peer_name))

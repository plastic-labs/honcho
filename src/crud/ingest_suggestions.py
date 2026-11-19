"""Suggested identity links for an import: read what the rules need, store what they say.

The organizer (`src/organizer/run.py`) reads a workspace's imported accounts
and records here, runs the rules (`src/organizer/identity.py`) with no
session held, and stores the result here. A person then lists the
suggestions, accepts one (which links the two accounts' people through
`crud.ingest_link.link_peers`, exactly as `POST /ingest/link` does) or
dismisses it.

An account is a peer binding ingest trusts (it created the binding's peer, or
a person confirmed it), keyed by the binding's ID, or an explicit `peer_id`
author, keyed `peer:<name>`. What a source said about an account (its handle,
aliases, email digest, bot flag) comes from the newest record that carried
it: records keep their people as received (`ingest_records.source_context`).

Suggestions outlive a reset, which deletes the records they were made from:
a refresh only adds or updates `open` suggestions, never deletes one, and each
keeps a snapshot of its two accounts, with how many records each had.

Every statement filters each table on `tenant_id` itself: several tables are
named only in joins, where the ORM tenant hook doesn't reach.
"""

import datetime
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.ingest_fence import take_workspace_fence
from src.crud.ingest_link import link_peers
from src.exceptions import ConflictException, ResourceNotFoundException
from src.organizer.identity import (
    RULES_VERSION,
    Account,
    AccountKind,
    Post,
    Suggestion,
)

_KINDS: dict[str, AccountKind] = {
    "external_id": "external",
    "provisional_name": "provisional",
    "source_peer": "source",
}
EXPLICIT = "peer:"
"""The key prefix of an explicit peer: it has no binding."""


@dataclass(frozen=True)
class Inputs:
    """What the rules read, and how to show each account later."""

    accounts: list[Account]
    posts: list[Post]
    snapshots: dict[str, dict[str, Any]]


def _active(record: Any) -> Any:
    session = models.Session
    return (
        select(session.name)
        .where(
            session.tenant_id == record.tenant_id,
            session.workspace_name == record.workspace_name,
            session.name == record.session_name,
            session.is_active.is_(True),
        )
        .exists()
    )


def _records_with_messages(workspace_name: str) -> Select[Any]:
    """Imported records in active sessions, joined to their live messages."""
    tenant_id = models.scoped_tenant_id()
    record, message = models.IngestRecord, models.Message
    return (
        select()
        .select_from(record)
        .join(
            message,
            (message.tenant_id == record.tenant_id)
            & (message.workspace_name == record.workspace_name)
            & (message.public_id == record.message_public_id),
        )
        .where(
            record.tenant_id == tenant_id,
            record.workspace_name == workspace_name,
            message.tenant_id == tenant_id,
            message.workspace_name == workspace_name,
            message.deleted_at.is_(None),
            _active(record),
        )
    )


def _participant_key(
    source: str,
    instance: str,
    container: Mapping[str, Any] | None,
    person: Mapping[str, Any],
) -> tuple[str, str, str, str] | None:
    """The binding key ingest gave a participant (`crud.ingest.peer_key_for`)."""
    name = person.get("name")
    if not isinstance(name, str):
        return None
    if person.get("id") is not None:
        return (source, instance, "external_id", str(person["id"]))
    container_id = (container or {}).get("id")
    scope = (
        f"container:{len(container_id)}:{container_id}"
        if isinstance(container_id, str)
        else "standalone"
    )
    return (source, instance, "provisional_name", f"{scope}:{name.casefold()}")


async def load_inputs(db: AsyncSession, workspace_name: str) -> Inputs:
    """Read the accounts and records of a workspace's imports, for the rules.

    SELECT-only: run it in a read-only session.
    """
    tenant_id = models.scoped_tenant_id()
    binding, peer = models.IngestPeerBinding, models.Peer
    record, message = models.IngestRecord, models.Message

    # Accounts: the bindings ingest trusts, as `ingest_status` counts them.
    bindings = (
        await db.execute(
            select(
                binding.id,
                binding.source,
                binding.source_instance,
                binding.key_kind,
                binding.key_value,
                binding.peer_name,
                binding.display_name,
            )
            .join(
                peer,
                (peer.tenant_id == binding.tenant_id)
                & (peer.workspace_name == binding.workspace_name)
                & (peer.name == binding.peer_name),
            )
            .where(
                binding.tenant_id == tenant_id,
                binding.workspace_name == workspace_name,
                peer.tenant_id == tenant_id,
                peer.workspace_name == workspace_name,
                (peer.internal_metadata["ingest"]["binding_id"].astext == binding.id)
                | binding.confirmed_at.is_not(None),
            )
            .order_by(binding.id)
        )
    ).all()
    by_key = {
        (b.source, b.source_instance, b.key_kind, b.key_value): b for b in bindings
    }

    # What the newest record said about each author, and how many each wrote.
    newest_author = (
        _records_with_messages(workspace_name)
        .add_columns(record.author_binding_id, record.source_context["author"])
        .where(record.author_binding_id.is_not(None))
        .distinct(record.author_binding_id)
        .order_by(record.author_binding_id, message.created_at.desc())
    )
    profiles: dict[str, dict[str, Any]] = {
        binding_id: context
        for binding_id, context in (await db.execute(newest_author)).tuples()
        if isinstance(context, dict)
    }
    # A Result has `.keys()`, so dict() would read it as a mapping: unpack rows.
    counts: dict[str, int] = {
        binding_id: n
        for binding_id, n in (
            await db.execute(
                _records_with_messages(workspace_name)
                .add_columns(record.author_binding_id, func.count(record.id))
                .where(record.author_binding_id.is_not(None))
                .group_by(record.author_binding_id)
            )
        ).tuples()
    }
    # Participants: a newer record's word wins, as for authors.
    participant_rows = await db.execute(
        _records_with_messages(workspace_name)
        .add_columns(
            record.source,
            record.source_instance,
            record.source_context["container"],
            record.source_context["participants"],
            message.created_at,
        )
        .where(func.jsonb_array_length(record.source_context["participants"]) > 0)
    )
    seen_at: dict[str, datetime.datetime] = {}
    for source, instance, container, people, at in participant_rows.tuples():
        listed: list[Any] = people if isinstance(people, list) else []  # pyright: ignore[reportUnknownVariableType]
        for person in listed:
            if not isinstance(person, dict):
                continue
            context: dict[str, Any] = person  # pyright: ignore[reportUnknownVariableType]
            key = _participant_key(source, instance, container, context)
            found = by_key.get(key) if key else None
            # What the account wrote itself wins over how others named it.
            if found is None or (found.id in profiles and found.id not in seen_at):
                continue
            if found.id not in seen_at or at > seen_at[found.id]:
                seen_at[found.id] = at
                profiles[found.id] = context

    # Explicit `peer_id` authors: no binding, so their peer is the account.
    explicit = (
        await db.execute(
            _records_with_messages(workspace_name)
            .add_columns(message.peer_name, func.count(record.id))
            .where(record.author_binding_id.is_(None))
            .group_by(message.peer_name)
        )
    ).tuples()

    accounts: list[Account] = []
    snapshots: dict[str, dict[str, Any]] = {}
    for b in bindings:
        profile = profiles.get(b.id, {})
        names = [n for n in (b.display_name, profile.get("name")) if n]
        aliases = profile.get("aliases")
        if isinstance(aliases, list):
            names += [a for a in aliases if isinstance(a, str)]  # pyright: ignore[reportUnknownVariableType]
        handle = profile.get("handle")
        digest = profile.get("email_digest")
        account = Account(
            key=b.id,
            source=b.source,
            instance=b.source_instance,
            kind=_KINDS.get(b.key_kind, "external"),
            names=tuple(dict.fromkeys(names)),
            handle=handle if isinstance(handle, str) else None,
            email_digest=digest if isinstance(digest, str) else None,
            bot=profile.get("bot") is True,
            peer=b.peer_name,
        )
        accounts.append(account)
        snapshots[b.id] = {
            "subject": b.id,
            "kind": account.kind,
            "source": b.source,
            "source_instance": b.source_instance,
            "id": b.key_value if b.key_kind == "external_id" else None,
            "display_name": b.display_name,
            "handle": account.handle,
            "records": counts.get(b.id, 0),
        }
    for peer_name, n in explicit:
        key = EXPLICIT + peer_name
        accounts.append(
            Account(
                key=key,
                source="explicit",
                instance="",
                kind="explicit",
                names=(peer_name,),
                handle=peer_name,
                peer=peer_name,
            )
        )
        snapshots[key] = {
            "subject": key,
            "kind": "explicit",
            "source": "explicit",
            "source_instance": "",
            "id": None,
            "display_name": peer_name,
            "handle": peer_name,
            "records": n,
        }

    # Posts: every record, with its text only when the rules could use it: a
    # link, or a source's name (a self-reference names the other source).
    sources = sorted({b.source for b in bindings})
    pattern = r"https?://" + "".join(f"|\\m{s}\\M" for s in sources)
    text = case((message.content.op("~*")(pattern), message.content), else_="")
    post_rows = await db.execute(
        _records_with_messages(workspace_name).add_columns(
            record.author_binding_id,
            message.peer_name,
            message.created_at,
            message.h_metadata["url"].astext,
            record.participant_binding_ids,
            text,
        )
    )
    posts = [
        Post(
            author=author or EXPLICIT + peer_name,
            at=at,
            text=content or "",
            url=url,
            participants=tuple(
                str(p)
                for p in (participants if isinstance(participants, list) else [])  # pyright: ignore[reportUnknownVariableType]
                if isinstance(p, str)
            ),
        )
        for author, peer_name, at, url, participants, content in post_rows.tuples()
    ]
    return Inputs(accounts=accounts, posts=posts, snapshots=snapshots)


def subject_key(accounts: Sequence[str]) -> str:
    """Two accounts, in sorted order: the same pair is always the same suggestion."""
    return "|".join(sorted(accounts))


async def save_suggestions(
    db: AsyncSession,
    workspace_name: str,
    workspace_id: str,
    found: Iterable[Suggestion],
    snapshots: Mapping[str, dict[str, Any]],
    refreshed_at: datetime.datetime,
    demoted: Mapping[tuple[str, str], str] | None = None,
) -> int:
    """Store a refresh's suggestions; return how many rows it added or updated.

    New pairs are added as `open`. An `open` suggestion the refresh made again
    gets its new evidence, unless a later refresh already wrote it. Accepted
    and dismissed suggestions, and open ones this refresh didn't make, are
    left as they are. `demoted` names earlier open `high` suggestions this
    refresh found would now join two people: they drop to `low`, with why.

    Takes the shared workspace fence first, so workspace deletion can't
    interleave, then writes nothing unless the workspace is still the one the
    refresh read (`workspace_id`): a workspace deleted, or deleted and made
    again under its name, after the read gets no suggestions from it. Rows are
    written in key order.
    """
    rows = [
        {
            "workspace_name": workspace_name,
            "kind": "same_person",
            "subject_key": subject_key(s.accounts),
            "accounts": [snapshots[k] for k in s.accounts],
            "confidence": s.confidence,
            "score": s.score,
            "evidence": [
                {"signal": e.signal, "level": e.level, "detail": e.detail}
                for e in s.evidence
            ],
            "stage": "rules",
            "rules_version": RULES_VERSION,
            "organizer_tokens": 0,
            "status": "open",
            "refreshed_at": refreshed_at,
        }
        for s in found
    ]
    await take_workspace_fence(db, workspace_name, mode="shared")
    current = await db.scalar(
        select(models.Workspace.id).where(
            models.Workspace.tenant_id == models.scoped_tenant_id(),
            models.Workspace.name == workspace_name,
            models.Workspace.auto_enabled_at.is_not(None),
        )
    )
    if current != workspace_id:
        return 0
    rows.sort(key=lambda r: r["subject_key"])
    table = models.IngestSuggestion
    written = 0
    for pair, reason in sorted((demoted or {}).items()):
        conflict = func.jsonb_build_array(
            func.jsonb_build_object(
                "signal", "conflict", "level", "strong", "detail", reason
            )
        )
        result = await db.execute(
            update(table)
            .where(
                table.tenant_id == models.scoped_tenant_id(),
                table.workspace_name == workspace_name,
                table.kind == "same_person",
                table.subject_key == subject_key(pair),
                table.status == "open",
                table.confidence == "high",
            )
            .values(
                confidence="low",
                evidence=table.evidence.op("||")(conflict),
                refreshed_at=refreshed_at,
            )
            .returning(table.id)
        )
        written += len(result.all())
    for start in range(0, len(rows), 500):
        statement = pg_insert(table).values(rows[start : start + 500])
        statement = statement.on_conflict_do_update(
            constraint="uq_ingest_suggestions_subject",
            set_={
                name: statement.excluded[name]
                for name in (
                    "accounts",
                    "confidence",
                    "score",
                    "evidence",
                    "stage",
                    "rules_version",
                    "organizer_tokens",
                    "refreshed_at",
                )
            },
            where=(table.status == "open")
            & (table.refreshed_at < statement.excluded.refreshed_at),
        ).returning(table.id)
        written += len((await db.execute(statement)).all())
    return written


async def open_high_pairs(
    db: AsyncSession, workspace_name: str
) -> list[tuple[str, str]]:
    """The account pairs of the workspace's open `high` suggestions."""
    table = models.IngestSuggestion
    keys = await db.scalars(
        select(table.subject_key).where(
            table.tenant_id == models.scoped_tenant_id(),
            table.workspace_name == workspace_name,
            table.kind == "same_person",
            table.status == "open",
            table.confidence == "high",
        )
    )
    pairs: list[tuple[str, str]] = []
    for key in keys:
        first, _, second = key.partition("|")
        if first and second:
            pairs.append((first, second))
    return pairs


# ---------------------------------------------------------------------------
# Reading, accepting and dismissing
# ---------------------------------------------------------------------------


def suggestions_query(
    workspace_name: str, *, status: str | None, confidence: str | None
) -> Select[tuple[models.IngestSuggestion]]:
    """A workspace's suggestions, best first."""
    table = models.IngestSuggestion
    query = select(table).where(
        table.tenant_id == models.scoped_tenant_id(),
        table.workspace_name == workspace_name,
    )
    if status is not None:
        query = query.where(table.status == status)
    if confidence is not None:
        query = query.where(table.confidence == confidence)
    return query.order_by(table.score.desc(), table.id)


async def current_peers(
    db: AsyncSession, workspace_name: str, subjects: Iterable[str]
) -> dict[str, str]:
    """The peer each account maps to now; a missing binding is left out."""
    wanted = set(subjects)
    peers = {s: s.removeprefix(EXPLICIT) for s in wanted if s.startswith(EXPLICIT)}
    ids = sorted(s for s in wanted if not s.startswith(EXPLICIT))
    if ids:
        binding = models.IngestPeerBinding
        rows = await db.execute(
            select(binding.id, binding.peer_name).where(
                binding.tenant_id == models.scoped_tenant_id(),
                binding.workspace_name == workspace_name,
                binding.id.in_(ids),
            )
        )
        peers.update({binding_id: name for binding_id, name in rows.tuples()})
    return peers


def to_schema(
    row: models.IngestSuggestion, peers: Mapping[str, str]
) -> schemas.IngestSuggestion:
    accounts = [
        schemas.IngestSuggestionAccount(
            source=a["source"],
            source_instance=a["source_instance"],
            id=a.get("id"),
            display_name=a.get("display_name"),
            handle=a.get("handle"),
            peer_id=peers.get(a["subject"]),
            records=a.get("records", 0),
        )
        for a in row.accounts
    ]
    now = [peers.get(a["subject"]) for a in row.accounts]
    return schemas.IngestSuggestion(
        id=row.id,
        kind="same_person",
        confidence=row.confidence,  # pyright: ignore[reportArgumentType]
        score=row.score,
        status=row.status,  # pyright: ignore[reportArgumentType]
        linked=None not in now and len(set(now)) == 1,
        accounts=accounts,
        evidence=[schemas.IngestSuggestionEvidence(**e) for e in row.evidence],
        stage=row.stage,
        refreshed_at=row.refreshed_at,
    )


async def _locked(
    db: AsyncSession, workspace_name: str, suggestion_id: str
) -> models.IngestSuggestion:
    table = models.IngestSuggestion
    row = await db.scalar(
        select(table)
        .where(
            table.tenant_id == models.scoped_tenant_id(),
            table.workspace_name == workspace_name,
            table.id == suggestion_id,
        )
        .with_for_update()
    )
    if row is None:
        raise ResourceNotFoundException(f"Suggestion {suggestion_id} not found")
    return row


async def _kept(
    db: AsyncSession,
    workspace_name: str,
    row: models.IngestSuggestion,
    peers: Mapping[str, str],
) -> list[str]:
    """The two peers to link, the one to keep first.

    Link merges away only a person an import created for a source account,
    so a peer anything else made is kept: an explicit peer, one made by hand
    that an account was linked into, or a person known only by name in one
    place (a provisional account). Otherwise the account with more records
    when the suggestion was made, so its name and configuration stay; on a
    tie, the suggestion's first account.
    """
    names = sorted({peers[a["subject"]] for a in row.accounts})
    created_by_import = set(
        await db.scalars(
            select(models.Peer.name).where(
                models.Peer.tenant_id == models.scoped_tenant_id(),
                models.Peer.workspace_name == workspace_name,
                models.Peer.name.in_(names),
                models.Peer.internal_metadata["ingest"]["binding_id"].is_not(None),
            )
        )
    )

    def rank(account: Mapping[str, Any]) -> tuple[bool, int]:
        peer = peers[account["subject"]]
        must_keep = peer not in created_by_import or account["kind"] in (
            "explicit",
            "provisional",
            "source",
        )
        return (must_keep, int(account.get("records", 0)))

    # A stable sort: on a tie the first account stays first.
    ordered = sorted(row.accounts, key=rank, reverse=True)
    return list(dict.fromkeys(peers[a["subject"]] for a in ordered))


async def accept_suggestion(
    db: AsyncSession, workspace_name: str, suggestion_id: str, keep: str | None
) -> tuple[models.IngestSuggestion, schemas.IngestLinkResult | None]:
    """Link the suggestion's two accounts into one person, and mark it accepted.

    Takes the exclusive workspace fence before any row lock, as link does. A
    pair already on one person is marked accepted with nothing to link.

    Raises:
        ResourceNotFoundException: no such suggestion, or an account is gone.
        ConflictException: it was dismissed, `keep` isn't one of its people, or
            the link is refused (the people have imported messages: reset first).
    """
    await take_workspace_fence(db, workspace_name, mode="exclusive")
    row = await _locked(db, workspace_name, suggestion_id)
    if row.status == "dismissed":
        raise ConflictException(
            f"Suggestion {suggestion_id} was dismissed; link the people with"
            + " POST /ingest/link instead."
        )
    subjects = [a["subject"] for a in row.accounts]
    peers = await current_peers(db, workspace_name, subjects)
    if missing := [s for s in subjects if s not in peers]:
        raise ResourceNotFoundException(
            f"An account of suggestion {suggestion_id} no longer exists: {missing}"
        )
    order = await _kept(db, workspace_name, row, peers)
    result: schemas.IngestLinkResult | None = None
    if len(order) > 1:
        if keep is not None:
            if keep not in order:
                raise ConflictException(
                    f"Keep one of the suggestion's people: {', '.join(order)}"
                )
            order = [keep, *(p for p in order if p != keep)]
        result = await link_peers(db, workspace_name, order)
    if row.status != "accepted":
        await db.execute(
            update(models.IngestSuggestion)
            .where(
                models.IngestSuggestion.tenant_id == row.tenant_id,
                models.IngestSuggestion.id == row.id,
            )
            .values(status="accepted", decided_at=func.statement_timestamp())
        )
    await db.refresh(row)
    return row, result


async def dismiss_suggestion(
    db: AsyncSession, workspace_name: str, suggestion_id: str
) -> models.IngestSuggestion:
    """Mark an open suggestion dismissed; a refresh never brings it back.

    Raises:
        ResourceNotFoundException: no such suggestion.
        ConflictException: it was already accepted.
    """
    await take_workspace_fence(db, workspace_name, mode="shared")
    row = await _locked(db, workspace_name, suggestion_id)
    if row.status == "accepted":
        raise ConflictException(
            f"Suggestion {suggestion_id} was accepted; take an account back out"
            + " with POST /ingest/unlink."
        )
    if row.status == "open":
        await db.execute(
            update(models.IngestSuggestion)
            .where(
                models.IngestSuggestion.tenant_id == row.tenant_id,
                models.IngestSuggestion.id == row.id,
            )
            .values(status="dismissed", decided_at=func.statement_timestamp())
        )
        await db.refresh(row)
    return row


async def suggestion_counts(db: AsyncSession, workspace_name: str) -> dict[str, int]:
    """Open suggestions by confidence."""
    table = models.IngestSuggestion
    counts: defaultdict[str, int] = defaultdict(int)
    rows = await db.execute(
        select(table.confidence, func.count(table.id))
        .where(
            table.tenant_id == models.scoped_tenant_id(),
            table.workspace_name == workspace_name,
            table.status == "open",
        )
        .group_by(table.confidence)
    )
    for confidence, n in rows.tuples():
        counts[confidence] = n
    return dict(counts)


def refreshing_clause(work_unit_key: str) -> Any:
    """Whether a refresh of a workspace's suggestions is waiting or running.

    `work_unit_key` is the workspace's organize work unit, which names it.
    """
    return or_(
        select(models.QueueItem.id)
        .where(
            models.QueueItem.work_unit_key == work_unit_key,
            models.QueueItem.processed == False,  # noqa: E712
        )
        .exists(),
        select(models.ActiveQueueSession.id)
        .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
        .exists(),
    )

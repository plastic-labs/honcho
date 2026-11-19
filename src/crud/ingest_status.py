"""What an import produced: its records, the people and sessions they map to.

Read by `GET /v3/workspaces/{workspace_id}/ingest/status`, `/ingest/people` and
`/ingest/sessions`, so a user can review an import before deriving it.

People and sessions come from the records themselves: a person is the author of
an ingested record or an account an ingest binding maps to them, and a session
holds ingested records. Bindings add the source accounts and names, and only
bindings ingest trusts count (it created the target for that binding, or a
person confirmed it), the same rule ingest follows when it reuses one.

"Modeled" mirrors what the deriver does with the settings it reads when records
are released: a person's `observe_me`, and a session's reasoning setting as
`utils.config_helpers.get_configuration` resolves it (the session's, else the
workspace's, else the server default). A session-level `observe_me` for one
member, or a record's own configuration, still overrides them for that member
or record.

Every statement filters each table on `tenant_id` itself: several tables are
named only in joins and aggregates, where the ORM tenant hook doesn't reach.
"""

from collections.abc import Iterable
from typing import Any

from sqlalchemy import Select, case, func, literal, null, select
from sqlalchemy.dialects.postgresql import JSONB, aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.ingest_suggestions import refreshing_clause, suggestion_counts
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import create_organize_payload
from src.utils.work_unit import construct_work_unit_key


async def require_auto_workspace(
    db: AsyncSession, workspace_name: str
) -> models.Workspace:
    """The workspace; refuse one that doesn't exist (404) or isn't auto (409)."""
    workspace = await db.scalar(
        select(models.Workspace).where(
            models.Workspace.tenant_id == models.scoped_tenant_id(),
            models.Workspace.name == workspace_name,
        )
    )
    if workspace is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")
    if workspace.auto_enabled_at is None:
        raise ConflictException(
            f"Workspace {workspace_name} isn't an auto workspace. Ingest writes"
            + " only to auto workspaces: create one with auto: true."
        )
    return workspace


def _in_active_session(record: Any) -> Any:
    """A record whose session is active: a reset marks its sessions inactive at once."""
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


def _owned(target: Any, binding: Any) -> Any:
    """A binding ingest trusts: it created the target, or a person confirmed it."""
    return (
        target.internal_metadata["ingest"]["binding_id"].astext == binding.id
    ) | binding.confirmed_at.is_not(None)


def _first(source: Any, instance: Any, name: Any, *order: Any) -> Any:
    """A group's first (source, instance, name), kept together as one object."""
    return func.jsonb_agg(
        aggregate_order_by(
            func.jsonb_build_object(
                "source", source, "source_instance", instance, "name", name
            ),
            *order,
        )
    ).op("->")(0)


def _people(workspace_name: str) -> Any:
    tenant_id = models.scoped_tenant_id()
    binding = models.IngestPeerBinding
    record = models.IngestRecord
    message = models.Message
    peer = models.Peer
    accounts = (
        select(
            binding.peer_name,
            func.jsonb_agg(
                aggregate_order_by(
                    func.jsonb_build_object(
                        "source",
                        binding.source,
                        "source_instance",
                        binding.source_instance,
                        "id",
                        case(
                            (binding.key_kind == "external_id", binding.key_value),
                            else_=None,
                        ),
                        "display_name",
                        binding.display_name,
                    ),
                    # The account the import created the person for first: it
                    # names the person, and linked accounts follow.
                    case(
                        (
                            peer.internal_metadata["ingest"]["binding_id"].astext
                            == binding.id,
                            0,
                        ),
                        else_=1,
                    ),
                    binding.created_at,
                    binding.id,
                )
            ).label("accounts"),
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
            _owned(peer, binding),
        )
        .group_by(binding.peer_name)
        .subquery("accounts")
    )
    # Who wrote each ingested record is its message's author, which an
    # explicit `peer_id` sets without any binding.
    written = (
        select(message.peer_name, func.count(record.id).label("messages"))
        .join(
            message,
            (message.tenant_id == record.tenant_id)
            & (message.workspace_name == record.workspace_name)
            & (message.public_id == record.message_public_id),
        )
        .where(
            record.tenant_id == tenant_id,
            record.workspace_name == workspace_name,
            _in_active_session(record),
        )
        .group_by(message.peer_name)
        .subquery("written")
    )
    peer_name = func.coalesce(accounts.c.peer_name, written.c.peer_name)
    return (
        select(
            peer_name.label("peer_id"),
            accounts.c.accounts.op("->")(0)
            .op("->>")("display_name")
            .label("display_name"),
            func.coalesce(accounts.c.accounts, literal([], JSONB)).label("accounts"),
            func.coalesce(written.c.messages, 0).label("messages"),
            func.coalesce(peer.configuration["observe_me"].as_boolean(), True).label(
                "modeled"
            ),
        )
        .select_from(
            accounts.join(
                written, written.c.peer_name == accounts.c.peer_name, full=True
            )
        )
        .join(
            peer,
            (peer.tenant_id == tenant_id)
            & (peer.workspace_name == workspace_name)
            & (peer.name == peer_name),
        )
        .subquery("people")
    )


def _sessions(workspace_name: str) -> Any:
    tenant_id = models.scoped_tenant_id()
    binding = models.IngestSessionBinding
    record = models.IngestRecord
    session = models.Session
    held = (
        select(
            record.session_name,
            func.count(record.id).label("messages"),
            _first(
                record.source,
                record.source_instance,
                null(),
                record.created_at,
                record.id,
            ).label("first_record"),
        )
        .where(record.tenant_id == tenant_id, record.workspace_name == workspace_name)
        .group_by(record.session_name)
        .subquery("held")
    )
    containers = (
        select(
            binding.session_name,
            _first(
                binding.source,
                binding.source_instance,
                binding.display_name,
                binding.created_at,
                binding.id,
            ).label("first_container"),
        )
        .join(
            session,
            (session.tenant_id == binding.tenant_id)
            & (session.workspace_name == binding.workspace_name)
            & (session.name == binding.session_name),
        )
        .where(
            binding.tenant_id == tenant_id,
            binding.workspace_name == workspace_name,
            _owned(session, binding),
        )
        .group_by(binding.session_name)
        .subquery("containers")
    )
    origin = func.coalesce(containers.c.first_container, held.c.first_record)
    return (
        select(
            held.c.session_name.label("session_id"),
            containers.c.first_container.op("->>")("name").label("display_name"),
            origin.op("->>")("source").label("source"),
            origin.op("->>")("source_instance").label("source_instance"),
            held.c.messages,
            session.configuration.label("configuration"),
        )
        .select_from(held)
        .join(
            session,
            (session.tenant_id == tenant_id)
            & (session.workspace_name == workspace_name)
            & (session.name == held.c.session_name)
            & session.is_active.is_(True),
        )
        .outerjoin(containers, containers.c.session_name == held.c.session_name)
        .subquery("sessions")
    )


def people_query(workspace_name: str) -> Select[Any]:
    """The workspace's imported people, most records first."""
    people = _people(workspace_name)
    return select(people).order_by(people.c.messages.desc(), people.c.peer_id)


def sessions_query(workspace_name: str) -> Select[Any]:
    """The workspace's imported sessions, most records first.

    Rows carry the session's configuration; `session_summaries` turns them into
    `IngestSessionSummary` with `modeled` resolved as the deriver resolves it.
    """
    sessions = _sessions(workspace_name)
    return select(sessions).order_by(sessions.c.messages.desc(), sessions.c.session_id)


def session_summaries(
    rows: Iterable[Any], workspace: models.Workspace
) -> list[schemas.IngestSessionSummary]:
    """Rows of `sessions_query` as summaries, `modeled` from the deriver's rules."""
    summaries: list[schemas.IngestSessionSummary] = []
    for row in rows:
        resolved = get_configuration(
            None, models.Session(configuration=row.configuration or {}), workspace
        )
        summaries.append(
            schemas.IngestSessionSummary(
                session_id=row.session_id,
                display_name=row.display_name,
                source=row.source,
                source_instance=row.source_instance,
                messages=row.messages,
                modeled=resolved.reasoning.enabled,
            )
        )
    return summaries


async def get_ingest_status(
    db: AsyncSession, workspace: models.Workspace
) -> schemas.IngestStatus:
    """Counts for the workspace's import: records, tokens, people and sessions."""
    tenant_id = models.scoped_tenant_id()
    workspace_name = workspace.name
    record = models.IngestRecord
    counted = await db.execute(
        select(record.derivation, func.count())
        .where(
            record.tenant_id == tenant_id,
            record.workspace_name == workspace_name,
            _in_active_session(record),
        )
        .group_by(record.derivation)
    )
    by_state: dict[str, int] = {state: count for state, count in counted.tuples()}
    message = models.Message
    deferred_tokens = await db.scalar(
        select(func.coalesce(func.sum(message.token_count), 0))
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
            record.derivation == "deferred",
            _in_active_session(record),
        )
    )
    people = _people(workspace_name)
    people_total, people_off = (
        await db.execute(
            select(
                func.count(),
                func.count().filter(people.c.modeled.is_(False)),
            ).select_from(people)
        )
    ).one()
    sessions = session_summaries(
        (await db.execute(sessions_query(workspace_name))).all(), workspace
    )
    being_deleted = await db.scalar(
        select(func.count(func.distinct(record.session_name))).where(
            record.tenant_id == tenant_id,
            record.workspace_name == workspace_name,
            ~_in_active_session(record),
        )
    )
    return schemas.IngestStatus(
        records=schemas.IngestRecordCounts(
            deferred=by_state.get("deferred", 0),
            pending=by_state.get("pending", 0),
            queued=by_state.get("queued", 0),
        ),
        deferred_tokens=int(deferred_tokens or 0),
        people=people_total,
        people_not_modeled=people_off,
        sessions=len(sessions),
        sessions_not_modeled=sum(1 for s in sessions if not s.modeled),
        sessions_being_deleted=int(being_deleted or 0),
        suggestions=await _suggestions(db, workspace_name),
    )


async def _suggestions(
    db: AsyncSession, workspace_name: str
) -> schemas.IngestSuggestionCounts:
    """Open suggestions by confidence, and whether a refresh is waiting or running."""
    counts = await suggestion_counts(db, workspace_name)
    key = construct_work_unit_key(workspace_name, create_organize_payload())
    refreshing = bool(await db.scalar(select(refreshing_clause(key))))
    return schemas.IngestSuggestionCounts(
        high=counts.get("high", 0), low=counts.get("low", 0), refreshing=refreshing
    )


__all__ = [
    "get_ingest_status",
    "people_query",
    "require_auto_workspace",
    "session_summaries",
    "sessions_query",
]

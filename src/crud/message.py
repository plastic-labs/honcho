from collections.abc import Sequence
from datetime import datetime
from logging import getLogger
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import ColumnElement, Select, and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from src import models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.telemetry.events import EmbeddingCallPurpose
from src.utils.filter import apply_filter
from src.utils.formatting import ILIKE_ESCAPE_CHAR, escape_ilike_pattern
from src.utils.types import embedding_call_purpose
from src.vector_store import get_external_vector_store

from .peer import reject_scope_peers
from .session import get_or_create_session

logger = getLogger(__name__)


def _deduplicate_messages(
    messages: Sequence[models.Message], limit: int
) -> list[models.Message]:
    """Deduplicate messages by public_id, preserving input order."""
    seen: set[str] = set()
    result: list[models.Message] = []
    for msg in messages:
        if msg.public_id not in seen:
            seen.add(msg.public_id)
            result.append(msg)
            if len(result) >= limit:
                break
    return result


def _expunge_snippets(
    db: AsyncSession, snippets: list[tuple[list[models.Message], list[models.Message]]]
) -> None:
    """Detach snippet messages from the session, guarding against duplicates."""
    seen: set[int] = set()
    for matches, context in snippets:
        for msg in [*matches, *context]:
            obj_id = id(msg)
            if obj_id in seen:
                continue
            db.expunge(msg)
            seen.add(obj_id)


async def get_peer_session_names(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    *,
    active_only: bool = False,
) -> list[str]:
    """Get all session names where a peer has a membership record.

    By default any membership record (regardless of joined_at/left_at) grants
    visibility to all messages in that session — this is the loose definition
    recall scoping uses.

    Pass ``active_only=True`` for the strict definition (``left_at IS NULL``),
    matching :func:`src.crud.session.is_peer_in_session`. The auth layer must
    use the strict one so that a single peer-scoped key gets the same answer
    whether it names a session directly or via a filter allowlist.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        active_only: Restrict to sessions the peer has not left

    Returns:
        Distinct session names the peer has a matching membership record in.
    """
    stmt = (
        select(models.session_peers_table.c.session_name)
        .where(models.session_peers_table.c.workspace_name == workspace_name)
        .where(models.session_peers_table.c.peer_name == peer_name)
        .distinct()
    )
    if active_only:
        stmt = stmt.where(models.session_peers_table.c.left_at.is_(None))
    result = await db.execute(stmt)
    return [row[0] for row in result.all()]


async def resolve_session_scope(
    db: AsyncSession | None,
    workspace_name: str,
    session_name: str | None,
    session_allowlist: list[str] | None,
    observer: str | None,
    *,
    operation_name: str = "resolve_session_scope",
) -> tuple[list[str] | None, bool]:
    """Resolve the effective session scope for a message query.

    Returns ``(allowed_session_names, deny)``:

    - ``allowed_session_names is None`` — apply no allowlist filter. Either the
      query is unrestricted, or ``session_name`` already pins it to one session.
    - a populated list — restrict the query to exactly these sessions.
    - ``deny=True`` — the caller must return an empty result *without* querying.

    The distinction between ``None`` and an empty list is load-bearing: the
    external vector stores drop an empty ``IN`` clause rather than matching
    nothing, so collapsing the two would fail open. This function therefore
    never returns an empty list — it returns ``deny=True`` instead.

    Touches the database only when an observer lookup is actually required, so
    callers on the external-vector-store path don't check out a connection
    before their network call.

    Args:
        db: Database session to reuse. Pass None to let this function open its
            own short-lived read-only session if (and only if) it needs one.
        workspace_name: Name of the workspace
        session_name: A single pinned session, if the caller named one
        session_allowlist: Optional session allowlist. ``None`` is unrestricted;
            an empty list fails closed.
        observer: When set, scope is limited to this peer's sessions and then
            intersected with ``session_allowlist``
        operation_name: Label for the self-managed DB session, when one is opened

    Returns:
        Tuple of (allowlist to filter on or None, whether to deny outright).
    """
    if session_name:
        # A specific session was requested. Fail closed when the allowlist
        # forbids it — routes guard this too, but other CRUD callers (the
        # dialectic tools) don't, so enforce it at the boundary.
        if session_allowlist is not None and session_name not in session_allowlist:
            return None, True
        return None, False

    if observer is None:
        if session_allowlist is None:
            return None, False
        allowed = list(session_allowlist)
        return (allowed, False) if allowed else (None, True)

    if db is not None:
        allowed = await get_peer_session_names(db, workspace_name, observer)
    else:
        async with tracked_db(f"{operation_name}.peer_scope", read_only=True) as own_db:
            allowed = await get_peer_session_names(own_db, workspace_name, observer)

    if session_allowlist is not None:
        scope = set(session_allowlist)
        allowed = [s for s in allowed if s in scope]

    return (allowed, False) if allowed else (None, True)


def observer_scope_clause(
    workspace_name: str,
    observer: str,
    session_column: InstrumentedAttribute[str],
) -> ColumnElement[bool]:
    """Correlated EXISTS restricting ``session_column`` to the observer's sessions.

    The in-database equivalent of filtering on :func:`get_peer_session_names`.
    Prefer it whenever the scope feeds a single SQL statement: a peer's
    membership count is unbounded, and materializing the names turns each one
    into its own bind parameter. The PostgreSQL wire protocol caps parameters
    at 65535 per statement, so a peer in enough sessions produces a query the
    driver cannot serialize at all — and the resulting error carries every
    parameter in its text.

    Matches the loose membership definition ``get_peer_session_names`` uses by
    default: any membership record grants visibility, whether or not the peer
    has since left the session.
    """
    session_peers = models.session_peers_table
    return (
        select(1)
        .where(session_peers.c.workspace_name == workspace_name)
        .where(session_peers.c.peer_name == observer)
        .where(session_peers.c.session_name == session_column)
        .exists()
    )


def resolve_session_scope_clauses(
    workspace_name: str,
    session_name: str | None,
    session_allowlist: list[str] | None,
    observer: str | None,
    session_column: InstrumentedAttribute[str],
) -> tuple[list[ColumnElement[bool]], bool]:
    """SQL-side counterpart to :func:`resolve_session_scope`.

    Returns ``(clauses, deny)``, where ``clauses`` are ANDed onto the caller's
    statement and ``deny=True`` means return an empty result without querying.
    Unlike :func:`resolve_session_scope` this touches no database and grows no
    bind parameters with the observer's session count — the observer half
    becomes a correlated EXISTS instead of an ``IN`` over fetched names.

    Scoping matches :func:`resolve_session_scope` case for case, with one
    deliberate difference: where that function returns ``deny=True`` because an
    observer's membership (or its intersection with the allowlist) is empty,
    this returns clauses that simply match no rows. Callers reach the same empty
    result, at the cost of running one indexed query that returns nothing.

    ``session_allowlist`` stays an ``IN`` clause: it is caller-supplied and
    therefore bounded, so it carries none of the unbounded-growth risk.

    Args:
        workspace_name: Name of the workspace
        session_name: A single pinned session, if the caller named one. The
            caller applies its own equality filter; this function only checks
            the allowlist permits it.
        session_allowlist: Optional session allowlist. ``None`` is
            unrestricted; an empty list fails closed.
        observer: When set, scope is limited to this peer's sessions
        session_column: The session-name column to scope, e.g.
            ``models.Message.session_name``
    """
    if session_name:
        # Fail closed when the allowlist forbids the pinned session, matching
        # `resolve_session_scope` — routes guard this too, but the dialectic
        # tools reach CRUD directly, so enforce it at the boundary.
        if session_allowlist is not None and session_name not in session_allowlist:
            return [], True
        return [], False

    clauses: list[ColumnElement[bool]] = []

    if observer is not None:
        clauses.append(observer_scope_clause(workspace_name, observer, session_column))

    if session_allowlist is not None:
        if not session_allowlist:
            return [], True
        clauses.append(session_column.in_(session_allowlist))

    return clauses, False


def _apply_token_limit(
    base_conditions: list[ColumnElement[Any]], token_limit: int
) -> Select[tuple[models.Message]]:
    """
    Helper function to apply token limit logic to a message query.

    Creates a subquery that calculates running sum of tokens for most recent messages
    and returns a select statement that joins with this subquery to limit results
    based on token count.

    Args:
        base_conditions: List of conditions to apply to the base query
        token_limit: Maximum number of tokens to include in the messages

    Returns:
        Select statement with token limit applied
    """
    # Create a subquery that calculates running sum of tokens for most recent messages
    token_subquery = (
        select(
            models.Message.id,
            func.sum(models.Message.token_count)
            .over(order_by=models.Message.id.desc())
            .label("running_token_sum"),
        )
        .where(*base_conditions)
        .subquery()
    )

    # Select Message objects where running sum doesn't exceed token_limit
    return (
        select(models.Message)
        .join(token_subquery, models.Message.id == token_subquery.c.id)
        .where(token_subquery.c.running_token_sum <= token_limit)
    )


async def _build_merged_snippets(
    db: AsyncSession,
    workspace_name: str,
    matched_messages: list[models.Message],
    context_window: int,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Group matched messages by session, merge overlapping context ranges, and fetch context.

    Takes a list of matched messages and builds conversation snippets by:
    1. Grouping matches by session name
    2. Sorting matches within each session by sequence number
    3. Merging overlapping context windows to avoid duplicate context
    4. Fetching the full context for each merged range from the database

    Args:
        db: Database session
        workspace_name: Name of the workspace
        matched_messages: List of messages that matched a search query
        context_window: Number of messages before/after each match to include

    Returns:
        List of tuples: (matched_messages_in_range, context_messages)
        Each tuple represents a snippet where context_messages includes all messages
        in the merged range (including the matched messages), ordered chronologically.
    """
    if not matched_messages:
        return []

    session_matches: dict[str, list[models.Message]] = {}
    for msg in matched_messages:
        session_matches.setdefault(msg.session_name, []).append(msg)

    # Build merged ranges per session, then issue a single batched query
    session_ranges: dict[str, list[tuple[int, int, list[models.Message]]]] = {}
    for sess_name, matches in session_matches.items():
        matches.sort(key=lambda m: m.seq_in_session)

        merged_ranges: list[tuple[int, int, list[models.Message]]] = []
        for match in matches:
            start = match.seq_in_session - context_window
            end = match.seq_in_session + context_window

            if merged_ranges and start <= merged_ranges[-1][1] + 1:
                prev_start, prev_end, prev_matches = merged_ranges[-1]
                merged_ranges[-1] = (
                    prev_start,
                    max(prev_end, end),
                    [*prev_matches, match],
                )
            else:
                merged_ranges.append((start, end, [match]))

        session_ranges[sess_name] = merged_ranges

    # One OR-of-ANDs predicate covers every (session, range) pair
    session_predicates = [
        and_(
            models.Message.session_name == sess_name,
            or_(
                *(
                    models.Message.seq_in_session.between(start_seq, end_seq)
                    for start_seq, end_seq, _ in merged_ranges
                )
            ),
        )
        for sess_name, merged_ranges in session_ranges.items()
    ]

    context_stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(or_(*session_predicates))
        .order_by(
            models.Message.session_name.asc(),
            models.Message.seq_in_session.asc(),
        )
    )

    context_result = await db.execute(context_stmt)
    by_session: dict[str, list[models.Message]] = {}
    for msg in context_result.scalars().all():
        by_session.setdefault(msg.session_name, []).append(msg)

    snippets: list[
        tuple[list[models.Message], list[models.Message]]
    ] = []  # list of tuples, each containing query matches and context messages
    for sess_name, merged_ranges in session_ranges.items():
        all_context_messages = by_session.get(sess_name, [])
        for start_seq, end_seq, range_matches in merged_ranges:
            context_messages = [
                msg
                for msg in all_context_messages
                if start_seq <= msg.seq_in_session <= end_seq
            ]
            snippets.append((range_matches, context_messages))

    return snippets


async def create_messages(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    workspace_name: str,
    session_name: str,
) -> list[models.Message]:
    """
    Bulk create messages for a session while maintaining order.

    Args:
        db: Database session
        messages: List of messages to create
        workspace_name: Name of the workspace
        session_name: Name of the session to create messages in

    Returns:
        List of created message objects

    Raises:
        ValidationException: If a message is authored by a scope peer
    """
    # Scope peers are silent observers — they can never author messages. Keyed
    # off name+flag so a legacy peer merely occupying the reserved namespace
    # keeps ingesting. Must stay *before* get_or_create_session below: that call
    # would create the scope peer and add it with a default SessionPeerConfig(),
    # clobbering its observe_others=True/observe_me=False membership config.
    await reject_scope_peers(
        db,
        workspace_name,
        (message.peer_name for message in messages),
        action="Scope peers cannot author messages.",
    )

    # Get or create session with peers in messages list
    peers = {message.peer_name: schemas.SessionPeerConfig() for message in messages}
    await get_or_create_session(
        db,
        session=schemas.SessionCreate(name=session_name, peers=peers),
        workspace_name=workspace_name,
    )

    await db.execute(text("SET LOCAL lock_timeout = '5s'"))
    await db.execute(
        text(
            "SELECT pg_advisory_xact_lock(hashtext(:workspace_name), hashtext(:session_name))"
        ),
        {"workspace_name": workspace_name, "session_name": session_name},
    )

    # Get the last sequence number on a session - uses (workspace_name, session_name, seq_in_session) index
    last_seq = (
        await db.scalar(
            select(models.Message.seq_in_session)
            .where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
            .order_by(models.Message.seq_in_session.desc())
            .limit(1)
        )
        or 0
    )

    # Create list of message objects (this will trigger the before_insert event)
    message_objects: list[models.Message] = []
    for offset, message in enumerate(messages, start=1):
        message_seq_in_session = last_seq + offset
        message_obj = models.Message(
            session_name=session_name,
            peer_name=message.peer_name,
            content=message.content,
            h_metadata=message.metadata or {},
            workspace_name=workspace_name,
            public_id=generate_nanoid(),
            token_count=len(message.encoded_message),
            created_at=message.created_at,  # Use provided created_at if available
            seq_in_session=message_seq_in_session,
        )
        message_objects.append(message_obj)

    db.add_all(message_objects)

    # If embedding is enabled, locally chunk the content and insert
    # one pending MessageEmbedding row per chunk in chunk order. The actual
    # embedding work is deferred to the reconciler
    if settings.EMBED_MESSAGES:
        id_resource_dict = {
            message_obj.public_id: message_obj.content
            for message_obj in message_objects
            if message_obj.content and message_obj.content.strip()
        }
        if id_resource_dict:
            chunks_by_id = embedding_client.prepare_chunks(id_resource_dict)
            peer_by_id = {m.public_id: m.peer_name for m in message_objects}
            pending_rows: list[models.MessageEmbedding] = []
            for message_obj in message_objects:
                chunks = chunks_by_id.get(message_obj.public_id, [])
                for chunk_text in chunks:
                    pending_rows.append(
                        models.MessageEmbedding(
                            content=chunk_text,
                            message_id=message_obj.public_id,
                            workspace_name=workspace_name,
                            session_name=session_name,
                            peer_name=peer_by_id[message_obj.public_id],
                            sync_state="pending",
                            embedding=None,
                        )
                    )
            if pending_rows:
                db.add_all(pending_rows)

    await db.commit()

    return message_objects


async def get_messages(
    workspace_name: str,
    session_name: str,
    reverse: bool | None = False,
    filters: dict[str, Any] | None = None,
    token_limit: int | None = None,
    message_count_limit: int | None = None,
) -> Select[tuple[models.Message]]:
    """
    Get messages from a session. If token_limit is provided, the n most recent messages
    with token count adding up to the limit will be returned. If message_count_limit is provided,
    the n most recent messages will be returned. If both are provided, message_count_limit will be
    used.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session
        reverse: Whether to reverse the order of messages
        filters: Filter to apply to the messages
        token_limit: Maximum number of tokens to include in the messages
        message_count_limit: Maximum number of messages to include

    Returns:
        Select statement for the messages
    """
    # Base query with workspace and session filters
    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]

    # Apply message count limit first (takes precedence over token limit)
    if message_count_limit is not None:
        stmt = select(models.Message).where(*base_conditions)
        stmt = apply_filter(stmt, models.Message, filters)
        # For message count limit, we want the most recent N messages
        # So we order by id desc to get most recent, then apply limit
        stmt = stmt.order_by(models.Message.id.desc()).limit(message_count_limit)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())
    elif token_limit is not None:
        # Apply token limit logic using helper function
        stmt = _apply_token_limit(base_conditions, token_limit)
        stmt = apply_filter(stmt, models.Message, filters)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())
    else:
        # Default case - no limits applied
        stmt = select(models.Message).where(*base_conditions)
        stmt = apply_filter(stmt, models.Message, filters)
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())

    return stmt


async def get_messages_id_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_id: int = 0,
    end_id: int | None = None,
    token_limit: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by primary key ID range.
    If end_id is not provided, all messages after and including start_id will be returned.
    If start_id is not provided, start will be beginning of session.

    Note: list is *inclusive* of the end_id message and start_id message.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        start_id: Primary key ID of the first message to return
        end_id: Primary key ID of the last message (exclusive)

    Returns:
        List of messages
    """
    if start_id < 0 or (end_id is not None and (start_id >= end_id or end_id <= 0)):
        return []

    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]
    if end_id:
        base_conditions.append(
            and_(models.Message.id >= start_id, models.Message.id < end_id)
        )
    else:
        base_conditions.append(models.Message.id >= start_id)

    if token_limit:
        # Apply token limit logic using helper function
        stmt = _apply_token_limit(base_conditions, token_limit)
        stmt = stmt.order_by(models.Message.id)
    else:
        stmt = select(models.Message).where(*base_conditions)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_messages_by_seq_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_seq: int = 1,
    end_seq: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by seq_in_session range.

    This is useful for getting the last N messages in a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        start_seq: Sequence number of the first message to return (inclusive)
        end_seq: Sequence number of the last message to return (inclusive)

    Returns:
        List of messages ordered by seq_in_session
    """
    if start_seq < 1 or (end_seq is not None and start_seq > end_seq):
        return []

    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]

    if end_seq is not None:
        base_conditions.append(
            and_(
                models.Message.seq_in_session >= start_seq,
                models.Message.seq_in_session <= end_seq,
            )
        )
    else:
        base_conditions.append(models.Message.seq_in_session >= start_seq)

    stmt = (
        select(models.Message)
        .where(*base_conditions)
        .order_by(models.Message.seq_in_session.asc())
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_message_seq_in_session(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: int,
) -> int:
    """
    Get the sequence number of a message within a session.

    Args:
        db: Database session
        session_name: Name of the session
        message_id: Primary key ID of the message

    Returns:
        The sequence number of the message (1-indexed)
    """
    stmt = (
        select(models.Message.seq_in_session)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.id == message_id)
    )
    seq: int | None = await db.scalar(stmt)
    return int(seq) if seq is not None else 0


async def get_message(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> models.Message | None:
    stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.public_id == message_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_message(
    db: AsyncSession,
    message: schemas.MessageUpdate,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> bool:
    honcho_message = await get_message(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        message_id=message_id,
    )
    if honcho_message is None:
        raise ValueError("Message not found or does not belong to user")
    if (
        message.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_message.h_metadata = message.metadata
    await db.commit()
    # await db.refresh(honcho_message)
    return honcho_message


async def _search_messages_external(
    workspace_name: str,
    query_embedding: list[float],
    limit: int,
    *,
    session_name: str | None = None,
    allowed_session_names: list[str] | None = None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
) -> list[str]:
    """Query the external vector store and return ordered message IDs.

    Multiple vector records can map to the same message (chunked embeddings),
    so we oversample from the vector store and deduplicate by message_id.
    """
    external_vector_store = get_external_vector_store()
    if external_vector_store is None:
        return []

    namespace = external_vector_store.get_vector_namespace("message", workspace_name)

    vector_filters: dict[str, Any] = {}
    if session_name:
        vector_filters["session_name"] = session_name
    elif allowed_session_names is not None:
        vector_filters["session_name"] = {"in": allowed_session_names}

    # Oversample: chunks can map to the same message, and date filters are
    # applied post-fetch (vector stores don't support temporal filtering),
    # so fetch extra to compensate for both deduplication and filtering.
    has_date_filters = after_date is not None or before_date is not None
    oversample = 6 if has_date_filters else 3
    vector_results = await external_vector_store.query(
        namespace,
        query_embedding,
        top_k=limit * oversample,
        filters=vector_filters if vector_filters else None,
        include_attributes=["message_id"],
    )

    if not vector_results:
        return []

    # Deduplicate by message_id preserving similarity order
    seen: dict[str, None] = {}
    for vr in vector_results:
        mid = vr.metadata.get("message_id")
        if mid and mid not in seen:
            seen[mid] = None
    message_ids = list(seen.keys())

    if not message_ids:
        return []

    return message_ids


async def _fetch_messages_by_ids(
    db: AsyncSession,
    workspace_name: str,
    message_ids: list[str],
    *,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
) -> list[models.Message]:
    """Fetch messages by ID, preserving the supplied ordering."""
    fetch_stmt = (
        select(models.Message)
        .where(models.Message.public_id.in_(message_ids))
        .where(models.Message.workspace_name == workspace_name)
    )
    if after_date:
        fetch_stmt = fetch_stmt.where(models.Message.created_at >= after_date)
    if before_date:
        fetch_stmt = fetch_stmt.where(models.Message.created_at <= before_date)

    result = await db.execute(fetch_stmt)
    messages_by_id = {msg.public_id: msg for msg in result.scalars().all()}

    return [messages_by_id[mid] for mid in message_ids if mid in messages_by_id]


async def _search_messages_pgvector(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    *,
    query_embedding: list[float],
    allowed_session_names: list[str] | None = None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 10,
    context_window: int = 2,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """Run semantic message search against pgvector-backed embeddings."""
    # pgvector path: cosine distance in SQL
    # Oversample because a message with multiple embedding chunks can
    # produce duplicate rows; we deduplicate in Python to preserve HNSW
    # index usage (a DISTINCT ON subquery would prevent the index scan).
    match_stmt = (
        select(models.Message)
        .join(
            models.MessageEmbedding,
            models.Message.public_id == models.MessageEmbedding.message_id,
        )
        # Exclude pending rows that haven't been embedded yet: their NULL
        # distance sorts last and would pad the window with unranked messages.
        .where(models.MessageEmbedding.embedding.isnot(None))
        .where(models.MessageEmbedding.workspace_name == workspace_name)
        .order_by(models.MessageEmbedding.embedding.cosine_distance(query_embedding))
        .limit(limit * 2)
    )

    if session_name:
        match_stmt = match_stmt.where(
            models.MessageEmbedding.session_name == session_name
        )
    elif allowed_session_names is not None:
        match_stmt = match_stmt.where(
            models.MessageEmbedding.session_name.in_(allowed_session_names)
        )

    if after_date:
        match_stmt = match_stmt.where(models.Message.created_at >= after_date)
    if before_date:
        match_stmt = match_stmt.where(models.Message.created_at <= before_date)

    result = await db.execute(match_stmt)
    matched_messages = _deduplicate_messages(result.scalars().all(), limit)

    return await _build_merged_snippets(
        db, workspace_name, matched_messages, context_window
    )


async def _semantic_search_messages(
    workspace_name: str,
    session_name: str | None,
    *,
    query_embedding: list[float],
    limit: int = 10,
    context_window: int = 2,
    operation_name: str,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """Run semantic message search with optional temporal filters.

    When observer is provided and session_name is None, results are
    scoped to sessions the observer has any membership record in. When
    session_allowlist is provided, that membership scope is further
    intersected with the allowlist (fail-closed: empty result on empty
    intersection).
    """
    # db=None: the helper opens its own short-lived session only if it needs
    # an observer lookup, so the external-store path below stays the first
    # thing that happens when no observer scoping applies.
    allowed_session_names, deny = await resolve_session_scope(
        None,
        workspace_name,
        session_name,
        session_allowlist,
        observer,
        operation_name=operation_name,
    )
    if deny:
        return []

    if settings.VECTOR_STORE.TYPE != "pgvector" and settings.VECTOR_STORE.MIGRATED:
        message_ids = await _search_messages_external(
            workspace_name,
            query_embedding,
            limit,
            session_name=session_name,
            allowed_session_names=allowed_session_names,
            after_date=after_date,
            before_date=before_date,
        )
        if not message_ids:
            return []

        async with tracked_db(operation_name, read_only=True) as db:
            matched_messages = (
                await _fetch_messages_by_ids(
                    db,
                    workspace_name,
                    message_ids,
                    after_date=after_date,
                    before_date=before_date,
                )
            )[:limit]
            snippets = await _build_merged_snippets(
                db, workspace_name, matched_messages, context_window
            )
            _expunge_snippets(db, snippets)
            return snippets

    async with tracked_db(operation_name, read_only=True) as db:
        snippets = await _search_messages_pgvector(
            db,
            workspace_name,
            session_name,
            query_embedding=query_embedding,
            allowed_session_names=allowed_session_names,
            after_date=after_date,
            before_date=before_date,
            limit=limit,
            context_window=context_window,
        )
        _expunge_snippets(db, snippets)
        return snippets


async def search_messages(
    workspace_name: str,
    session_name: str | None,
    query: str,
    limit: int = 10,
    context_window: int = 2,
    embedding: list[float] | None = None,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity and return conversation snippets.

    Each result includes matched messages plus surrounding context. Overlapping
    snippets within the same session are merged to avoid repetition.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session (optional)
            Deprecated for *scoping*: prefer session_allowlist, which
            intersects with observer membership. This parameter also pins
            the query to one session and bypasses observer scoping, so it
            is not a drop-in equivalent and is not removed.
        query: Search query text
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        embedding: Optional pre-computed embedding
        observer: When provided and session_name is None, scope results
            to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an
            empty list fails closed (empty result); a populated list is
            intersected with the observer's session scope when observer is set

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
        Context messages are ordered chronologically and include the matched messages.
    """
    if embedding is not None:
        query_embedding = embedding
    else:
        # Caller didn't precompute; tag this fallback path as search_messages.
        # Callers that have a more specific intent should set their own
        # context manager before calling and pass the precomputed embedding.
        with embedding_call_purpose(
            EmbeddingCallPurpose.SEARCH_MESSAGES.value,
            workspace_name=workspace_name,
        ):
            query_embedding = await embedding_client.embed(query)
    return await _semantic_search_messages(
        workspace_name,
        session_name,
        query_embedding=query_embedding,
        limit=limit,
        context_window=context_window,
        operation_name="message.search_messages",
        observer=observer,
        session_allowlist=session_allowlist,
    )


async def _grep_messages_internal(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    text: str,
    limit: int = 10,
    context_window: int = 2,
    allowed_session_names: list[str] | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """Internal implementation of exact-text message search."""
    # Build the base query with ILIKE for case-insensitive text search
    escaped_text = escape_ilike_pattern(text)
    match_stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(
            models.Message.content.ilike(f"%{escaped_text}%", escape=ILIKE_ESCAPE_CHAR)
        )
        .order_by(models.Message.created_at.desc())
        .limit(limit)
    )

    if session_name:
        match_stmt = match_stmt.where(models.Message.session_name == session_name)
    elif allowed_session_names is not None:
        match_stmt = match_stmt.where(
            models.Message.session_name.in_(allowed_session_names)
        )

    result = await db.execute(match_stmt)
    matched_messages = list(result.scalars().all())

    return await _build_merged_snippets(
        db, workspace_name, matched_messages, context_window
    )


async def grep_messages(
    workspace_name: str,
    session_name: str | None,
    text: str,
    limit: int = 10,
    context_window: int = 2,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages containing specific text (case-insensitive substring match).

    Unlike semantic search, this finds EXACT text matches. Useful for finding
    specific names, dates, phrases, or keywords.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session (optional - searches all sessions if None)
            Deprecated for *scoping*: prefer session_allowlist, which
            intersects with observer membership. This parameter also pins
            the query to one session and bypasses observer scoping, so it
            is not a drop-in equivalent and is not removed.
        text: Text to search for (case-insensitive)
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        observer: When provided and session_name is None, scope results
            to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an
            empty list fails closed (empty result); a populated list is
            intersected with the observer's session scope when observer is set

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    async with tracked_db("message.grep_messages", read_only=True) as db:
        allowed_session_names, deny = await resolve_session_scope(
            db, workspace_name, session_name, session_allowlist, observer
        )
        if deny:
            return []

        snippets = await _grep_messages_internal(
            db,
            workspace_name,
            session_name,
            text,
            limit,
            context_window,
            allowed_session_names=allowed_session_names,
        )
        _expunge_snippets(db, snippets)
        return snippets


async def get_messages_by_date_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 20,
    order: str = "desc",
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[models.Message]:
    """
    Get messages within a date range.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session (optional - searches all sessions if None)
            Deprecated for *scoping*: prefer session_allowlist, which
            intersects with observer membership. This parameter also pins
            the query to one session and bypasses observer scoping, so it
            is not a drop-in equivalent and is not removed.
        after_date: Return messages after this datetime
        before_date: Return messages before this datetime
        limit: Maximum messages to return
        order: Sort order - 'asc' for oldest first, 'desc' for newest first
        observer: When provided and session_name is None, scope results
            to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an
            empty list fails closed (empty result); a populated list is
            intersected with the observer's session scope when observer is set

    Returns:
        List of messages within the date range
    """
    allowed_session_names, deny = await resolve_session_scope(
        db, workspace_name, session_name, session_allowlist, observer
    )
    if deny:
        return []

    stmt = select(models.Message).where(models.Message.workspace_name == workspace_name)

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)
    elif allowed_session_names is not None:
        stmt = stmt.where(models.Message.session_name.in_(allowed_session_names))
    if after_date:
        stmt = stmt.where(models.Message.created_at >= after_date)
    if before_date:
        stmt = stmt.where(models.Message.created_at <= before_date)

    if order == "asc":
        stmt = stmt.order_by(models.Message.created_at.asc())
    else:
        stmt = stmt.order_by(models.Message.created_at.desc())

    stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def search_messages_temporal(
    workspace_name: str,
    session_name: str | None,
    query: str,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 10,
    context_window: int = 2,
    embedding: list[float] | None = None,
    observer: str | None = None,
    session_allowlist: list[str] | None = None,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity with optional date filtering.

    Combines the power of semantic search with time constraints. Use after_date
    to find recent mentions, or before_date to find what was said before a certain point.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session (optional)
            Deprecated for *scoping*: prefer session_allowlist, which
            intersects with observer membership. This parameter also pins
            the query to one session and bypasses observer scoping, so it
            is not a drop-in equivalent and is not removed.
        query: Search query text
        after_date: Only return messages after this datetime
        before_date: Only return messages before this datetime
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match to include
        embedding: Optional pre-computed embedding for the query
        observer: When provided and session_name is None, scope results
            to sessions this peer belongs to
        session_allowlist: Optional session allowlist. None is unrestricted; an
            empty list fails closed (empty result); a populated list is
            intersected with the observer's session scope when observer is set

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    if embedding is not None:
        query_embedding = embedding
    else:
        # Caller didn't precompute; tag this fallback path as search_messages.
        # Callers that have a more specific intent should set their own
        # context manager before calling and pass the precomputed embedding.
        with embedding_call_purpose(
            EmbeddingCallPurpose.SEARCH_MESSAGES.value,
            workspace_name=workspace_name,
        ):
            query_embedding = await embedding_client.embed(query)
    return await _semantic_search_messages(
        workspace_name,
        session_name,
        query_embedding=query_embedding,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        context_window=context_window,
        operation_name="message.search_messages_temporal",
        observer=observer,
        session_allowlist=session_allowlist,
    )

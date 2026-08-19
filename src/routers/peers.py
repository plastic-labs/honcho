"""FastAPI routes for peer resources and peer-scoped operations."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi.responses import StreamingResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.config import settings
from src.crud.message import get_peer_session_names
from src.crud.session import is_peer_in_session
from src.dependencies import db, read_db, tracked_db
from src.dialectic.chat import agentic_chat, agentic_chat_stream
from src.embedding_client import embedding_client
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
    ValidationException,
)
from src.security import JWTParams, require_auth
from src.telemetry import prometheus_metrics
from src.telemetry.events import EmbeddingCallPurpose, GetContextEvent, emit
from src.utils.filter import MAX_SESSION_ALLOWLIST_ENTRIES, extract_session_allowlist
from src.utils.schema_conversion import json_response_schema_to_pydantic
from src.utils.scopes import (
    is_scope_peer,
    is_scope_peer_name,
    validate_no_scope_peer_names,
)
from src.utils.search import search
from src.utils.types import embedding_call_purpose

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/peers",
    tags=["peers"],
)


def _validate_scope_option(
    *,
    filters: dict[str, Any] | None,
    session_id: str | None,
    jwt_params: JWTParams,
) -> None:
    """Enforce the v1 `scope` exclusions and auth rule (chat/representation).

    `scope` is mutually exclusive with `filters` and `session_id` (422), and a
    scope's member sessions may exceed a peer's own membership, so scoped
    reads require a workspace- or admin-level key.

    401 rather than 403: every other scope surface refuses a narrow key with 401
    — the `/scopes` router via `require_auth`, and the `scopes` field on session
    create — so a peer key would otherwise get two different codes for the same
    feature depending on which side of it was touched.
    """
    if filters is not None:
        raise ValidationException("`scope` and `filters` are mutually exclusive")
    if session_id:
        raise ValidationException("`scope` and `session_id` are mutually exclusive")
    if jwt_params.p is not None:
        raise AuthenticationException(
            "`scope` requires a workspace- or admin-level key"
        )


async def _resolve_scope_option(
    workspace_id: str,
    scope: str | list[str],
    *,
    db_action: str,
) -> tuple[str | None, list[str] | None]:
    """Map a validated `scope` option to (observer_override, session_allowlist).

    A single scope swaps the observer to the scope peer: conclusion recall is
    then confined to the (scope, observed) collection and message recall to
    the scope's session membership by existing observer semantics. A list of
    scopes keeps the path peer as observer and returns the union of the
    scopes' member sessions as an explicit allowlist (fail-closed when empty).
    """
    async with tracked_db(db_action, read_only=True) as scope_db:
        if isinstance(scope, str):
            [scope_peer] = await crud.resolve_scope_peers(
                scope_db, workspace_id, [scope]
            )
            return scope_peer, None

        scope_peers = await crud.resolve_scope_peers(scope_db, workspace_id, scope)
        union: list[str] = []
        seen: set[str] = set()
        for scope_peer in scope_peers:
            for session_name in await get_peer_session_names(
                scope_db, workspace_id, scope_peer
            ):
                if session_name not in seen:
                    seen.add(session_name)
                    union.append(session_name)

    if len(union) > MAX_SESSION_ALLOWLIST_ENTRIES:
        raise ValidationException(
            "The scopes' combined membership exceeds the maximum of "
            + f"{MAX_SESSION_ALLOWLIST_ENTRIES} sessions per request"
        )
    return None, union


@router.post(
    "/list",
    response_model=Page[schemas.Peer],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_peers(
    workspace_id: str = Path(...),
    options: schemas.PeerGet | None = Body(
        None, description="Filtering options for the peers list"
    ),
    reverse: bool = Query(False, description="Whether to reverse the order of results"),
    db: AsyncSession = read_db,
):
    """Get all Peers for a Workspace, paginated with optional filters.

    Scope peers are excluded by default; set `kind` to "scope" for scope peers
    only, or "all" for everything.
    """
    filter_param = None
    if options and hasattr(options, "filters"):
        filter_param = options.filters
        if filter_param == {}:
            filter_param = None

    return await apaginate(
        db,
        await crud.get_peers(
            workspace_name=workspace_id,
            filters=filter_param,
            reverse=reverse,
            kind=options.kind if options else None,
        ),
    )


@router.post(
    "",
    response_model=schemas.Peer,
)
async def get_or_create_peer(
    response: Response,
    workspace_id: str = Path(...),
    peer: schemas.PeerCreate = Body(..., description="Peer creation parameters"),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get a Peer by ID or create a new Peer with the given ID.

    If peer_id is provided as a query parameter, it uses that (must match JWT workspace_id).
    Otherwise, it uses the peer_id from the JWT.
    """
    # validate workspace query param
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    if peer.name:
        if not jwt_params.ad and jwt_params.p is not None and jwt_params.p != peer.name:
            raise AuthenticationException("Unauthorized access to resource")
    else:
        # Use peer_id from JWT
        if not jwt_params.p:
            raise AuthenticationException("Peer ID not found in query parameter or JWT")
        peer.name = jwt_params.p

    # The scope namespace is reserved: scope peers are only created through
    # the scopes facade (POST /workspaces/{workspace_id}/scopes).
    validate_no_scope_peer_names(
        [peer.name],
        action="Use the scopes routes to create scopes.",
    )

    result = await crud.get_or_create_peers(
        db, workspace_name=workspace_id, peers=[peer]
    )
    await db.commit()
    await result.post_commit()
    response.status_code = 201 if result.created else 200
    return result.resource[0]


@router.put(
    "/{peer_id}",
    response_model=schemas.Peer,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def update_peer(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    peer: schemas.PeerUpdate = Body(..., description="Updated peer parameters"),
    db: AsyncSession = db,
):
    """Update a Peer's metadata and/or configuration.

    Returns 422 if the peer is a scope — use the scopes routes to manage scopes.
    """
    # Three-way on the reserved namespace, all enforced inside ``crud.update_peer``
    # on the resolved row so there is no check-then-use window: a real scope is
    # refused (this route replaces `configuration` wholesale, so it must never touch
    # a facade-managed peer); an existing *unflagged* peer that merely occupies the
    # namespace is a normal peer and updates fine; and a reserved-prefix name that
    # does not exist is refused by create-path validation rather than being minted.
    #
    # Kept out of the docstring deliberately: FastAPI publishes that into the
    # OpenAPI description, and callers need the contract, not the mechanism.
    updated_peer = await crud.update_peer(
        db, workspace_name=workspace_id, peer_name=peer_id, peer=peer
    )
    return updated_peer


@router.post(
    "/{peer_id}/sessions",
    response_model=Page[schemas.Session],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_sessions_for_peer(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    options: schemas.SessionGet | None = Body(
        None, description="Filtering options for the sessions list"
    ),
    reverse: bool = Query(False, description="Whether to reverse the order of results"),
    db: AsyncSession = read_db,
):
    """Get all Sessions for a Peer, paginated with optional filters."""
    filter_param = None

    if options and hasattr(options, "filters"):
        filter_param = options.filters
        if filter_param == {}:
            filter_param = None

    return await apaginate(
        db,
        await crud.get_sessions_for_peer(
            workspace_name=workspace_id,
            peer_name=peer_id,
            filters=filter_param,
            reverse=reverse,
        ),
    )


@router.post(
    "/{peer_id}/chat",
    responses={
        200: {
            "content": {
                "application/json": {
                    "schema": schemas.DialecticResponse.model_json_schema()
                },
                "text/event-stream": {},
            },
        },
    },
)
async def chat(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    options: schemas.DialecticOptions = Body(...),
    jwt_params: JWTParams = Depends(
        require_auth(workspace_name="workspace_id", peer_name="peer_id")
    ),
):
    """
    Query a Peer's representation using natural language. Performs agentic search and reasoning to comprehensively
    answer the query based on all latent knowledge gathered about the peer from their messages and conclusions.
    """
    # Scope peers are never observed, so no representation of them exists to
    # query. Covers the path-level observer too: a scope `peer_id` no longer
    # errors out downstream now that crud.get_peer takes a plain name, and
    # querying from a scope's perspective is a read-side surface that does not
    # exist yet.
    scope_candidates = [
        n for n in (peer_id, options.target) if n is not None and is_scope_peer_name(n)
    ]
    if scope_candidates:
        async with tracked_db("peers.chat.scope_check", read_only=True) as s_db:
            # Strict variant, matching the representation route: `target` is an
            # observed position and nothing here creates the peer, so a reserved
            # name that does not exist yet must be refused rather than answered
            # and then turned into a scope.
            await crud.reject_scope_observed(
                s_db,
                workspace_id,
                scope_candidates,
                action=(
                    "No representation is formed of a scope, so a scope cannot "
                    "be a chat observer or target."
                ),
            )

    # Scoped reads: a single scope swaps the observer to the scope
    # peer; a list of scopes becomes a session allowlist over their union.
    observer = peer_id
    scope_session_union: list[str] | None = None
    if options.scope is not None:
        _validate_scope_option(
            filters=options.filters,
            session_id=options.session_id,
            jwt_params=jwt_params,
        )
        observer_override, scope_session_union = await _resolve_scope_option(
            workspace_id, options.scope, db_action="peers.chat.resolve_scope"
        )
        if observer_override is not None:
            observer = observer_override

    # The session id arrives in the body, so require_auth can't gate on it. A
    # peer-scoped key may only scope a chat to a session its peer belongs to;
    # without this check it could read any session's messages (the dialectic
    # injects session history) by naming it here. Workspace/admin tokens
    # (jwt_params.p is None) are unaffected.
    if jwt_params.p is not None and options.session_id:
        async with tracked_db("peers.chat.is_peer_in_session", read_only=True) as s_db:
            if not await is_peer_in_session(
                s_db, workspace_id, options.session_id, jwt_params.p
            ):
                raise AuthenticationException("JWT not permissioned for this resource")

    # Parse the session allowlist from filters (422 on unsupported keys/shapes,
    # and on a session_id the allowlist doesn't cover).
    session_allowlist = extract_session_allowlist(
        options.filters, must_include=options.session_id
    )
    # A peer-scoped key may only name sessions its peer belongs to — the
    # allowlist reaches message recall the same way session_id does above.
    # `active_only` matches the is_peer_in_session check above, so both gates
    # answer the same question for a peer that has left a session.
    if jwt_params.p is not None and session_allowlist is not None:
        async with tracked_db("peers.chat.session_scope_auth", read_only=True) as s_db:
            member_sessions = set(
                await get_peer_session_names(
                    s_db, workspace_id, jwt_params.p, active_only=True
                )
            )
        if not set(session_allowlist) <= member_sessions:
            raise AuthenticationException("JWT not permissioned for this resource")

    # A list of scopes resolves to a session allowlist over their union, which
    # replaces any filters-derived allowlist (the two are mutually exclusive, so
    # only one can be set).
    if scope_session_union is not None:
        session_allowlist = scope_session_union

    # Convert the caller's JSON Schema so malformed schemas fail immediately with 422
    response_model: type[BaseModel] | None = None
    if options.response_format is not None:
        try:
            response_model = json_response_schema_to_pydantic(options.response_format)
        except ValueError as e:
            raise ValidationException(f"Invalid response_format: {e}") from None

    # Get or create the peer to ensure it exists
    async with tracked_db("peers.chat.get_or_create_peer") as peer_db:
        peers_result = await crud.get_or_create_peers(
            peer_db,
            workspace_name=workspace_id,
            peers=[schemas.PeerSpec(name=peer_id)],
        )
        # Re-check on the resolved row: the name-level check above ran before the
        # peer was resolved, so a scope created in between would be picked up here
        # as existing and used as the chat observer. Deliberately NOT named
        # `observer` — that holds the effective observer, which a single `scope`
        # has already swapped to the scope peer, and rebinding it here would
        # silently undo the swap.
        path_peer = peers_result.resource[0]
        if is_scope_peer(path_peer.name, path_peer.internal_metadata):
            raise ValidationException(
                "No representation is formed of a scope, so a scope cannot be a "
                + "chat observer or target."
            )
        await peer_db.commit()
    await peers_result.post_commit()

    if options.stream:
        # Stream the response using Server-Sent Events

        async def format_sse_stream(
            chunks: AsyncIterator[str],
        ) -> AsyncIterator[str]:
            """Format chunks as SSE events."""
            async for chunk in chunks:
                yield f"data: {json.dumps({'delta': {'content': chunk}, 'done': False})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_dialectic_call(
                workspace_name=workspace_id,
                reasoning_level=options.reasoning_level,
            )

        return StreamingResponse(
            format_sse_stream(
                agentic_chat_stream(
                    workspace_name=workspace_id,
                    session_name=options.session_id,
                    query=options.query,
                    observer=observer,
                    observed=options.target if options.target is not None else peer_id,
                    reasoning_level=options.reasoning_level,
                    session_allowlist=session_allowlist,
                    response_model=response_model,
                )
            ),
            media_type="text/event-stream",
        )

    response = await agentic_chat(
        workspace_name=workspace_id,
        session_name=options.session_id,
        query=options.query,
        # a single `scope` swaps the observer to the scope peer
        observer=observer,
        # if target is given, that's the observed peer. otherwise, observer==observed
        # and it's answered from the omniscient Honcho perspective
        observed=options.target if options.target is not None else peer_id,
        reasoning_level=options.reasoning_level,
        session_allowlist=session_allowlist,
        response_model=response_model,
    )

    # Prometheus metrics
    if settings.METRICS.ENABLED:
        prometheus_metrics.record_dialectic_call(
            workspace_name=workspace_id,
            reasoning_level=options.reasoning_level,
        )

    return schemas.DialecticResponse(content=response if response else None)


@router.post(
    "/{peer_id}/representation",
    response_model=schemas.RepresentationResponse,
)
async def get_representation(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    options: schemas.PeerRepresentationGet = Body(
        ..., description="Options for getting the peer representation"
    ),
    jwt_params: JWTParams = Depends(
        require_auth(workspace_name="workspace_id", peer_name="peer_id")
    ),
):
    """Get a curated subset of a Peer's Representation. A Representation is always a subset of the total
    knowledge about the Peer. The subset can be scoped and filtered in various ways.


    If a session_id is provided in the body, we get the Representation of the Peer scoped to that Session.
    If a target is provided, we get the Representation of the target from the perspective of the Peer.
    If no target is provided, we get the omniscient Honcho Representation of the Peer.
    """
    # Fast-fail before any embedding work. Same guard as the authoritative one
    # below, so a reserved name is refused here rather than after paying for an
    # embedding; the check is repeated at the read because this session closes and
    # a scope could be created in between.
    scope_candidates = [
        n for n in (peer_id, options.target) if n is not None and is_scope_peer_name(n)
    ]
    if scope_candidates:
        async with tracked_db(
            "peers.representation.scope_check", read_only=True
        ) as s_db:
            await crud.reject_scope_observed(
                s_db,
                workspace_id,
                scope_candidates,
                action=(
                    "No representation is formed of a scope, so a scope cannot "
                    "be a representation observer or target."
                ),
            )

    # Parse the session allowlist from filters (422 on unsupported keys/shapes,
    # and on a session_id the allowlist doesn't cover).
    session_allowlist = extract_session_allowlist(
        options.filters, must_include=options.session_id
    )

    # Scoped reads: a single scope swaps the observer to the scope
    # peer; a list of scopes becomes a session allowlist over their union.
    observer = peer_id
    scope_session_union: list[str] | None = None
    if options.scope is not None:
        _validate_scope_option(
            filters=options.filters,
            session_id=options.session_id,
            jwt_params=jwt_params,
        )
        observer_override, scope_session_union = await _resolve_scope_option(
            workspace_id, options.scope, db_action="peers.representation.resolve_scope"
        )
        if observer_override is not None:
            observer = observer_override
    if scope_session_union is not None:
        session_allowlist = scope_session_union

    try:
        embedding: list[float] | None = None
        if options.search_query:
            try:
                with embedding_call_purpose(
                    EmbeddingCallPurpose.SEARCH_MEMORY.value,
                    workspace_name=workspace_id,
                    parent_category="api",
                ):
                    embedding = await embedding_client.embed(options.search_query)
            except Exception:
                # Swallowed on purpose (see include_semantic_query below), but not
                # silently: without this a provider outage degrades every search
                # request to derived+recent retrieval with no signal anywhere.
                logger.warning(
                    "Representation search embedding failed for workspace %s,"
                    + " degrading to non-semantic retrieval",
                    workspace_id,
                    exc_info=True,
                )

        observed = options.target if options.target is not None else peer_id
        # Re-check and read in one short session, opened only now — after the
        # embedding call above, so no connection is held across external work.
        # The early check ran in a session that has since closed and, being
        # name-based, also passed any reserved name that did not yet exist; a scope
        # created in between would otherwise be used here. Sharing the session with
        # the read means a scope committed after this check cannot have any
        # conclusions in the collection the read then examines.
        async with tracked_db(
            "peers.representation.read", read_only=True
        ) as read_session:
            await crud.reject_scope_observed(
                read_session,
                workspace_id,
                {peer_id, observed},
                action=(
                    "No representation is formed of a scope, so a scope cannot be"
                    " a representation observer or target."
                ),
            )
            # If no target specified, this is the global (omniscient) representation
            representation = await crud.get_working_representation(
                workspace_id,
                db=read_session,
                # a single `scope` swaps the observer to the scope peer
                observer=observer,
                observed=observed,
                session_allowlist=[options.session_id]
                if options.session_id is not None
                else session_allowlist,
                # Only ask for the semantic branch when we actually have an
                # embedding. The precompute above is suppressed, and both
                # `RepresentationManager.get_working_representation` and
                # `crud.query_documents` fall back to embedding internally when a
                # query arrives without one — which would run an external call
                # inside this session, and the innermost fallback is unsuppressed
                # (a provider outage would surface as a 500). Degrading to
                # derived+recent retrieval keeps the session DB-only.
                include_semantic_query=options.search_query
                if embedding is not None
                else None,
                embedding=embedding,
                semantic_search_top_k=options.search_top_k,
                semantic_search_max_distance=options.search_max_distance,
                include_most_derived=options.include_most_frequent
                if options.include_most_frequent is not None
                else False,
                max_observations=options.max_conclusions
                if options.max_conclusions is not None
                else settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
                parent_category="api",
            )
        return schemas.RepresentationResponse(
            representation=representation.format_as_markdown()
        )
    except ValueError as e:
        logger.warning(f"Failed to get representation for peer {peer_id}: {str(e)}")
        raise ResourceNotFoundException("Peer or session not found") from e


@router.get(
    "/{peer_id}/card",
    response_model=schemas.PeerCardResponse,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_peer_card(
    workspace_id: str = Path(...),
    peer_id: str = Path(..., description="ID of the observer peer"),
    target: str | None = Query(
        None,
        description="Optional target peer to retrieve a card for, from the observer's perspective. If not provided, returns the observer's own card",
    ),
    db: AsyncSession = read_db,
):
    """Get a peer card for a specific peer relationship.

    Returns the peer card that the observer peer has for the target peer if it exists.
    If no target is specified, returns the observer's own peer card.
    """
    # If no target specified, get the observer's own card
    observed = target if target is not None else peer_id

    peer_card = await crud.get_peer_card(
        db, workspace_id, observer=peer_id, observed=observed
    )
    return schemas.PeerCardResponse(peer_card=peer_card)


@router.put(
    "/{peer_id}/card",
    response_model=schemas.PeerCardResponse,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def set_peer_card(
    workspace_id: str = Path(...),
    peer_id: str = Path(..., description="ID of the observer peer"),
    peer_card_data: schemas.PeerCardSet = Body(
        ..., description="Peer card data to set"
    ),
    target: str | None = Query(
        None,
        description="Optional target peer to set a card for, from the observer's perspective. If not provided, sets the observer's own card",
    ),
    db: AsyncSession = db,
):
    """Set a peer card for a specific peer relationship.

    Sets the peer card that the observer peer has for the target peer.
    If no target is specified, sets the observer's own peer card.
    """
    # If no target specified, set the observer's own card
    observed = target if target is not None else peer_id

    # The scope guard lives in crud.set_peer_card, in the same transaction as the
    # JSONB write, so the Dreamer and agent-tool paths are covered too. Nothing
    # expensive happens between here and there, so a duplicate early check would
    # only cost an extra query.
    await crud.set_peer_card(
        db,
        workspace_id,
        peer_card=peer_card_data.peer_card,
        observer=peer_id,
        observed=observed,
    )

    return schemas.PeerCardResponse(peer_card=peer_card_data.peer_card)


@router.get(
    "/{peer_id}/context",
    response_model=schemas.PeerContext,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_peer_context(
    workspace_id: str = Path(...),
    peer_id: str = Path(..., description="ID of the observer peer"),
    target: str | None = Query(
        None,
        description="Optional target peer to get context for, from the observer's perspective. If not provided, returns the observer's own context (self-observation)",
    ),
    search_query: str | None = Query(
        None,
        description="Optional query to curate the representation around semantic search results",
    ),
    search_top_k: int | None = Query(
        None,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Number of semantic-search-retrieved conclusions to include",
    ),
    search_max_distance: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Only used if `search_query` is provided. Maximum distance for semantically relevant conclusions",
    ),
    include_most_frequent: bool = Query(
        default=True,
        description="Whether to include the most frequent conclusions in the representation",
    ),
    max_conclusions: int | None = Query(
        None,
        ge=1,
        le=100,
        description="Maximum number of conclusions to include in the representation",
    ),
):
    """
    Get context for a peer, including their representation and peer card.

    This endpoint returns a curated subset of the representation and peer card for a peer.
    If a target is specified, returns the context for the target from the
    observer peer's perspective. If no target is specified, returns the
    peer's own context (self-observation).

    This is useful for getting all the context needed about a peer without
    making multiple API calls.
    """
    # Scope peers may not appear on the generic peer-context surface: no
    # representation is formed of a scope, and scoped reads go through the
    # `scope` option on chat/representation/session-context instead. Flag-based
    # rather than prefix-based, so a legacy peer merely occupying the reserved
    # name keeps working; strict on a reserved name that does not exist yet,
    # since nothing here creates it. Costs no query when no reserved name is
    # present, and runs before any embedding work.
    scope_candidates = [
        n for n in (peer_id, target) if n is not None and is_scope_peer_name(n)
    ]
    if scope_candidates:
        async with tracked_db("peers.context.scope_check", read_only=True) as s_db:
            await crud.reject_scope_observed(
                s_db,
                workspace_id,
                scope_candidates,
                action="Use the `scope` option on the read routes instead.",
            )

    # If no target specified, get the peer's own context (self-observation)
    observed = target if target is not None else peer_id
    context_started = perf_counter()

    try:
        embedding: list[float] | None = None
        if search_query:
            with (
                suppress(Exception),
                embedding_call_purpose(
                    EmbeddingCallPurpose.SEARCH_MEMORY.value,
                    workspace_name=workspace_id,
                    parent_category="api",
                ),
            ):
                embedding = await embedding_client.embed(search_query)

        # Get the working representation
        representation = await crud.get_working_representation(
            workspace_id,
            observer=peer_id,
            observed=observed,
            session_allowlist=None,  # Peer context is global, not session-scoped
            include_semantic_query=search_query,
            embedding=embedding,
            semantic_search_top_k=search_top_k,
            semantic_search_max_distance=search_max_distance,
            include_most_derived=include_most_frequent,
            max_observations=max_conclusions
            if max_conclusions is not None
            else settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
            parent_category="api",
        )

        async with tracked_db(
            "peers.get_peer_context.peer_card", read_only=True
        ) as card_db:
            peer_card = await crud.get_peer_card(
                card_db, workspace_id, observer=peer_id, observed=observed
            )

        response = schemas.PeerContext(
            peer_id=peer_id,
            target_id=observed,
            representation=representation.format_as_markdown(),
            peer_card=peer_card,
        )
        emit(
            GetContextEvent(
                workspace_name=workspace_id,
                context_scope="peer",
                peer_name=peer_id,
                target_name=observed,
                has_representation=bool(response.representation),
                has_peer_card=peer_card is not None,
                search_query_provided=search_query is not None,
                search_top_k=search_top_k,
                search_max_distance=search_max_distance,
                include_most_frequent=include_most_frequent,
                max_conclusions=max_conclusions,
                total_duration_ms=(perf_counter() - context_started) * 1000,
            )
        )
        return response
    except ValueError as e:
        logger.warning(f"Failed to get context for peer {peer_id}: {str(e)}")
        raise ResourceNotFoundException("Peer not found") from e


@router.post(
    "/{peer_id}/search",
    response_model=list[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def search_peer(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    body: schemas.MessageSearchOptions = Body(
        ...,
        description="Message search parameters. Use `limit` to control the number of results returned.",
    ),
):
    """Search a Peer's messages, optionally filtered by various criteria."""
    # take user-provided filter and add workspace_id and peer_id to it
    filters = body.filters or {}
    filters["workspace_id"] = workspace_id
    filters["peer_id"] = peer_id
    return await search(body.query, filters=filters, limit=body.limit)

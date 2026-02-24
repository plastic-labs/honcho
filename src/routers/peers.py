import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi.responses import StreamingResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.config import settings
from src.dependencies import db, tracked_db
from src.dialectic.chat import agentic_chat, agentic_chat_stream
from src.exceptions import AuthenticationException, ResourceNotFoundException
from src.security import JWTParams, require_auth
from src.telemetry import prometheus_metrics
from src.utils.search import search

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/peers",
    tags=["peers"],
)


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
    db: AsyncSession = db,
):
    """Get all Peers for a Workspace, paginated with optional filters."""
    filter_param = None
    if options and hasattr(options, "filters"):
        filter_param = options.filters
        if filter_param == {}:
            filter_param = None

    return await apaginate(
        db,
        await crud.get_peers(workspace_name=workspace_id, filters=filter_param),
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
    """Update a Peer's metadata and/or configuration."""
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
    db: AsyncSession = db,
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
        ),
    )


@router.post(
    "/{peer_id}/chat",
    summary="Query a Peer's representation using natural language",
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
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def chat(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    options: schemas.DialecticOptions = Body(...),
):
    """
    Query a Peer's representation using natural language. Performs agentic search and reasoning to comprehensively
    answer the query based on all latent knowledge gathered about the peer from their messages and conclusions.
    """
    # Get or create the peer to ensure it exists
    async with tracked_db("peers.chat.get_or_create_peer") as peer_db:
        peers_result = await crud.get_or_create_peers(
            peer_db,
            workspace_name=workspace_id,
            peers=[schemas.PeerCreate(name=peer_id)],
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
                    observer=peer_id,
                    observed=options.target if options.target is not None else peer_id,
                    reasoning_level=options.reasoning_level,
                )
            ),
            media_type="text/event-stream",
        )

    response = await agentic_chat(
        workspace_name=workspace_id,
        session_name=options.session_id,
        query=options.query,
        observer=peer_id,
        # if target is given, that's the observed peer. otherwise, observer==observed
        # and it's answered from the omniscient Honcho perspective
        observed=options.target if options.target is not None else peer_id,
        reasoning_level=options.reasoning_level,
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
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_representation(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    options: schemas.PeerRepresentationGet = Body(
        ..., description="Options for getting the peer representation"
    ),
):
    """Get a curated subset of a Peer's Representation. A Representation is always a subset of the total
    knowledge about the Peer. The subset can be scoped and filtered in various ways.


    If a session_id is provided in the body, we get the Representation of the Peer scoped to that Session.
    If a target is provided, we get the Representation of the target from the perspective of the Peer.
    If no target is provided, we get the omniscient Honcho Representation of the Peer.
    """
    try:
        # If no target specified, get global representation (omniscient Honcho perspective)
        representation = await crud.get_working_representation(
            workspace_id,
            observer=peer_id,
            observed=options.target if options.target is not None else peer_id,
            session_name=options.session_id,
            include_semantic_query=options.search_query,
            semantic_search_top_k=options.search_top_k,
            semantic_search_max_distance=options.search_max_distance,
            include_most_derived=options.include_most_frequent
            if options.include_most_frequent is not None
            else False,
            max_observations=options.max_conclusions
            if options.max_conclusions is not None
            else settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
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
    db: AsyncSession = db,
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
    db: AsyncSession = db,
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
    # If no target specified, get the peer's own context (self-observation)
    observed = target if target is not None else peer_id

    try:
        # Get the working representation
        representation = await crud.get_working_representation(
            workspace_id,
            observer=peer_id,
            observed=observed,
            session_name=None,  # Peer context is global, not session-scoped
            include_semantic_query=search_query,
            semantic_search_top_k=search_top_k,
            semantic_search_max_distance=search_max_distance,
            include_most_derived=include_most_frequent,
            max_observations=max_conclusions
            if max_conclusions is not None
            else settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
        )

        # Get the peer card
        peer_card = await crud.get_peer_card(
            db, workspace_id, observer=peer_id, observed=observed
        )

        return schemas.PeerContext(
            peer_id=peer_id,
            target_id=observed,
            representation=representation.format_as_markdown(),
            peer_card=peer_card,
        )
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
    db: AsyncSession = db,
):
    """Search a Peer's messages, optionally filtered by various criteria."""
    # take user-provided filter and add workspace_id and peer_id to it
    filters = body.filters or {}
    filters["workspace_id"] = workspace_id
    filters["peer_id"] = peer_id
    return await search(db, body.query, filters=filters, limit=body.limit)

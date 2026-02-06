import logging

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import config, crud, schemas
from src.cache.client import cache
from src.crud.session import session_cache_key, session_peer_config_cache_key
from src.dependencies import db, tracked_db
from src.deriver.enqueue import enqueue_deletion
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
    ValidationException,
)
from src.security import JWTParams, require_auth
from src.utils import summarizer
from src.utils.representation import Representation
from src.utils.search import search
from src.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions",
    tags=["sessions"],
)


async def _get_working_representation_task(
    workspace_id: str,
    last_message: str | None,
    *,
    observer: str,
    observed: str,
    session_name: str | None,
    search_top_k: int | None,
    search_max_distance: float | None,
    include_most_derived: bool,
    max_observations: int | None,
) -> Representation:
    """
    Atomic task to get working representation using tracked_db.

    Args:
        workspace_id: The workspace identifier
        last_message: Optional last message for semantic query
        observer: Name of the observer peer
        observed: Name of the observed peer
        session_name: Optional session to filter by
        search_top_k: Number of semantic-search-retrieved observations to include in the representation
        search_max_distance: Maximum distance to search for semantically relevant observations
        include_most_derived: Whether to include the most derived observations in the representation
        max_observations: Maximum number of observations to include in the representation

    Returns:
        The working representation
    """
    return await crud.get_working_representation(
        workspace_name=workspace_id,
        observer=observer,
        observed=observed,
        session_name=session_name,
        include_semantic_query=last_message,
        semantic_search_top_k=search_top_k,
        semantic_search_max_distance=search_max_distance,
        include_most_derived=include_most_derived,
        max_observations=max_observations
        if max_observations is not None
        else config.settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    )


async def _get_peer_card_task(
    workspace_id: str,
    *,
    observer: str,
    observed: str,
) -> list[str] | None:
    """
    Atomic task to get peer card using tracked_db.

    Args:
        workspace_id: The workspace identifier
        observer: Name of the observer peer
        observed: Name of the observed peer

    Returns:
        The peer card or None if not found
    """
    async with tracked_db("get_peer_card") as db:
        return await crud.get_peer_card(
            db,
            workspace_name=workspace_id,
            observer=observer,
            observed=observed,
        )


async def _get_session_context_task(
    workspace_id: str,
    session_id: str,
    token_limit: int,
    include_summary: bool,
) -> tuple[schemas.Summary | None, list[schemas.Message]]:
    """
    Atomic task to get session context using tracked_db.

    Args:
        workspace_id: The workspace identifier
        session_id: The session identifier
        token_limit: Maximum tokens for the context
        include_summary: Whether to include summary if available

    Returns:
        Tuple of (summary, messages)
    """
    async with tracked_db("get_session_context") as db:
        summary, messages = await summarizer.get_session_context(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            token_limit=token_limit,
            include_summary=include_summary,
        )
        # Convert SQLAlchemy models to Pydantic schemas while session is active
        message_schemas = [schemas.Message.model_validate(msg) for msg in messages]
        return summary, message_schemas


@router.post(
    "/list",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_sessions(
    workspace_id: str = Path(...),
    options: schemas.SessionGet | None = Body(
        None, description="Filtering and pagination options for the sessions list"
    ),
    db: AsyncSession = db,
):
    """Get all Sessions for a Workspace, paginated with optional filters."""
    filter_param = None

    if options and hasattr(options, "filters") and options.filters:
        filter_param = options.filters
        if filter_param == {}:  # Explicitly check for empty dict
            filter_param = None

    return await apaginate(
        db, await crud.get_sessions(workspace_name=workspace_id, filters=filter_param)
    )


@router.post(
    "",
    response_model=schemas.Session,
)
async def get_or_create_session(
    response: Response,
    workspace_id: str = Path(...),
    session: schemas.SessionCreate = Body(
        ..., description="Session creation parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get a Session by ID or create a new Session with the given ID.

    If Session ID is provided as a parameter, it verifies the Session is in the Workspace.
    Otherwise, it uses the session_id from the JWT for verification.
    """
    # Verify JWT has access to the requested resource
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    # Use session from JWT if not provided in query
    if session.name:
        if (
            not jwt_params.ad
            and jwt_params.s is not None
            and jwt_params.s != session.name
        ):
            raise AuthenticationException("Unauthorized access to resource")
    else:
        if not jwt_params.s:
            raise AuthenticationException(
                "Session ID not found in query parameter or JWT"
            )
        session.name = jwt_params.s

    # Handle session creation with proper error handling
    try:
        result = await crud.get_or_create_session(
            db, workspace_name=workspace_id, session=session
        )
        response.status_code = 201 if result.created else 200
        return result.resource
    except ValueError as e:
        logger.warning(f"Failed to get or create session {session.name}: {str(e)}")
        raise ValidationException(str(e)) from e


@router.put(
    "/{session_id}",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def update_session(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    session: schemas.SessionUpdate = Body(
        ..., description="Updated session parameters"
    ),
    db: AsyncSession = db,
):
    """Update a Session's metadata and/or configuration."""
    try:
        updated_session = await crud.update_session(
            db, workspace_name=workspace_id, session_name=session_id, session=session
        )
        return updated_session
    except ValueError as e:
        logger.warning(f"Failed to update session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.delete(
    "/{session_id}",
    status_code=202,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def delete_session(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """
    Delete a Session and all associated messages.

    The Session is marked as inactive immediately and returns 202 Accepted. The actual
    deletion of all related data happens asynchronously via the queue with retry support.

    This action cannot be undone.
    """
    try:
        # Mark session as inactive immediately (fast operation)
        session = await crud.get_session(db, session_id, workspace_id)
        session.is_active = False

        # Enqueue deletion task for processing with retry support
        # Pass db session so it's all in one transaction
        await enqueue_deletion(
            workspace_name=workspace_id,
            deletion_type="session",
            resource_id=session_id,
            db_session=db,
        )

        await db.commit()
        try:
            await cache.delete(session_cache_key(workspace_id, session_id))
            await cache.delete(session_peer_config_cache_key(workspace_id, session_id))
        except Exception:  # nosec B110 — best-effort; TTL will clean up
            pass

        logger.debug("Session %s marked as inactive, deletion enqueued", session_id)
        return {"message": "Session deleted successfully"}
    except ValueError as e:
        logger.warning(f"Failed to delete session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post(
    "/{session_id}/clone",
    response_model=schemas.Session,
    status_code=201,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def clone_session(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    message_id: str | None = Query(
        None, description="Message ID to cut off the clone at"
    ),
    db: AsyncSession = db,
):
    """Clone a Session, optionally up to a specific message ID."""
    try:
        # TODO: Update crud.clone_session to work with new paradigm
        cloned_session = await crud.clone_session(
            db,
            workspace_name=workspace_id,
            original_session_name=session_id,
            cutoff_message_id=message_id,
        )
        logger.debug("Session %s cloned successfully", session_id)
        return cloned_session
    except ValueError as e:
        logger.warning(f"Failed to clone session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def add_peers_to_session(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    peers: dict[str, schemas.SessionPeerConfig] = Body(
        ...,
        description="List of peer IDs (with session-level configuration) to add to the session",
    ),
    db: AsyncSession = db,
):
    """Add Peers to a Session. If a Peer does not yet exist, it will be created automatically."""
    try:
        result = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(
                name=session_id,
                peers=peers,
            ),
            workspace_name=workspace_id,
        )
        return result.resource
    except ValueError as e:
        logger.warning(f"Failed to add peers to session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.put(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def set_session_peers(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    peers: dict[str, schemas.SessionPeerConfig] = Body(
        ...,
        description="List of peer IDs (with session-level configuration) to set for the session",
    ),
    db: AsyncSession = db,
):
    """
    Set the Peers in a Session. If a Peer does not yet exist, it will be created automatically.

    This will fully replace the current set of Peers in the Session.
    """
    try:
        await crud.set_peers_for_session(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_names=peers,
        )
        # Get the session to return
        result = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(name=session_id),
            workspace_name=workspace_id,
        )
        logger.debug("Set peers for session %s successfully", session_id)
        return result.resource
    except ValueError as e:
        logger.warning(f"Failed to set peers for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Failed to set peers for session") from e


@router.delete(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def remove_peers_from_session(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    peers: list[str] = Body(
        ..., description="List of peer IDs to remove from the session"
    ),
    db: AsyncSession = db,
):
    """Remove Peers by ID from a Session."""
    try:
        await crud.remove_peers_from_session(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_names=set(peers),
        )
        # Get the session to return
        result = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(name=session_id),
            workspace_name=workspace_id,
        )
        logger.debug("Removed peers from session %s successfully", session_id)
        return result.resource
    except ValueError as e:
        logger.warning(f"Failed to remove peers from session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/peers/{peer_id}/config",
    response_model=schemas.SessionPeerConfig,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_peer_config(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    peer_id: str = Path(...),
    db: AsyncSession = db,
):
    """Get the configuration for a Peer in a Session."""
    return await crud.get_peer_config(
        db,
        workspace_name=workspace_id,
        session_name=session_id,
        peer_id=peer_id,
    )


@router.put(
    "/{session_id}/peers/{peer_id}/config",
    status_code=204,
    response_model=None,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def set_peer_config(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    peer_id: str = Path(...),
    config: schemas.SessionPeerConfig = Body(..., description="New peer configuration"),
    db: AsyncSession = db,
):
    """Set the configuration for a Peer in a Session."""
    try:
        await crud.set_peer_config(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_name=peer_id,
            config=config,
        )
        logger.debug(
            "Set peer config for %s in session %s successfully", peer_id, session_id
        )
    except ValueError as e:
        logger.warning(
            f"Failed to set peer config for {peer_id} in session {session_id}: {str(e)}"
        )
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/peers",
    response_model=Page[schemas.Peer],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_session_peers(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """Get all Peers in a Session. Results are paginated."""
    try:
        peers_query = await crud.get_peers_from_session(
            workspace_name=workspace_id, session_name=session_id
        )
        return await apaginate(db, peers_query)
    except ValueError as e:
        logger.warning(f"Failed to get peers from session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/context",
    response_model=schemas.SessionContext,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_session_context(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    tokens: int | None = Query(
        None,
        le=config.settings.GET_CONTEXT_MAX_TOKENS,
        description=f"Number of tokens to use for the context. Includes summary if set to true. Includes representation and peer card if they are included in the response. If not provided, the context will be exhaustive (within {config.settings.GET_CONTEXT_MAX_TOKENS} tokens)",
    ),
    *,
    search_query: str | None = Query(
        None,
        description="A query string used to fetch semantically relevant conclusions",
    ),
    include_summary: bool = Query(
        default=True,
        description="Whether or not to include a summary *if* one is available for the session",
        alias="summary",
    ),
    peer_target: str | None = Query(
        None,
        description="The target of the perspective. If given without `peer_perspective`, will get the Honcho-level representation and peer card for this peer. If given with `peer_perspective`, will get the representation and card for this peer *from the perspective of that peer*.",
    ),
    peer_perspective: str | None = Query(
        None,
        description="A peer to get context for. If given, response will attempt to include representation and card from the perspective of that peer. Must be provided with `peer_target`.",
    ),
    limit_to_session: bool = Query(
        default=False,
        description="Only used if `search_query` is provided. Whether to limit the representation to the session (as opposed to everything known about the target peer)",
    ),
    search_top_k: int | None = Query(
        None,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. The number of semantic-search-retrieved conclusions to include in the representation",
    ),
    search_max_distance: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Only used if `search_query` is provided. The maximum distance to search for semantically relevant conclusions",
    ),
    include_most_frequent: bool = Query(
        default=False,
        description="Only used if `search_query` is provided. Whether to include the most frequent conclusions in the representation",
    ),
    max_conclusions: int | None = Query(
        None,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. The maximum number of conclusions to include in the representation",
    ),
):
    """
    Produce a context object from the Session. The caller provides an optional token limit which the entire context must fit into.
    If not provided, the context will be exhaustive (within configured max tokens). To do this, we allocate 40% of the token limit
    to the summary, and 60% to recent messages -- as many as can fit. Note that the summary will usually take up less space than
    this. If the caller does not want a summary, we allocate all the tokens to recent messages.
    """
    token_limit = (
        tokens if tokens is not None else config.settings.GET_CONTEXT_MAX_TOKENS
    )

    if peer_perspective and not peer_target:
        raise ValidationException(
            "peer_target must be provided if peer_perspective is provided"
        )

    if not peer_target:
        # No representation or card needed
        summary, messages = await _get_session_context_task(
            workspace_id, session_id, token_limit, include_summary
        )
        return schemas.SessionContext(
            name=session_id,
            messages=messages,
            summary=summary,
        )

    observer = peer_perspective or peer_target
    observed = peer_target

    # Run representation and card tasks sequentially to avoid event loop issues
    # with tracked_db creating separate database sessions
    representation = await _get_working_representation_task(
        workspace_id,
        search_query,
        observer=observer,
        observed=observed,
        session_name=session_id if limit_to_session else None,
        search_top_k=search_top_k,
        search_max_distance=search_max_distance,
        include_most_derived=include_most_frequent,
        max_observations=max_conclusions,
    )
    card = await _get_peer_card_task(workspace_id, observer=observer, observed=observed)

    # adjust token limit downward to account for approximate token count of representation and card
    # TODO determine if this impacts performance too much
    adjusted_token_limit = (
        token_limit - estimate_tokens(str(representation)) - estimate_tokens(card)
    )

    # Get the session context with the adjusted limit
    summary, messages = await _get_session_context_task(
        workspace_id, session_id, adjusted_token_limit, include_summary
    )

    return schemas.SessionContext(
        name=session_id,
        messages=messages,
        summary=summary,
        peer_representation=representation.format_as_markdown(),
        peer_card=card,
    )


@router.get(
    "/{session_id}/summaries",
    response_model=schemas.SessionSummaries,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_session_summaries(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
) -> schemas.SessionSummaries:
    """
    Get available summaries for a Session.

    Returns both short and long summaries if available, including metadata like
    the message ID they cover up to, creation timestamp, and token count.
    """
    # Reuse the logic from get_context to get both summaries
    short_summary, long_summary = await summarizer.get_both_summaries(
        db,
        workspace_name=workspace_id,
        session_name=session_id,
    )

    # Convert the internal Summary TypedDict to our Pydantic schema
    short_summary_schema = None
    if short_summary:
        short_summary_schema = summarizer.to_schema_summary(short_summary)

    long_summary_schema = None
    if long_summary:
        long_summary_schema = summarizer.to_schema_summary(long_summary)

    return schemas.SessionSummaries(
        name=session_id,
        short_summary=short_summary_schema,
        long_summary=long_summary_schema,
    )


@router.post(
    "/{session_id}/search",
    response_model=list[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def search_session(
    workspace_id: str = Path(...),
    session_id: str = Path(...),
    body: schemas.MessageSearchOptions = Body(
        ..., description="Message search parameters"
    ),
    db: AsyncSession = db,
):
    """
    Search a Session with optional filters. Use `limit` to control the number of results returned.
    """
    # take user-provided filter and add workspace_id and session_id to it
    filters = body.filters or {}
    filters["workspace_id"] = workspace_id
    filters["session_id"] = session_id
    return await search(
        db,
        body.query,
        filters=filters,
        limit=body.limit,
    )

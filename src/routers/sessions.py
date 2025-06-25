import logging

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.config import settings
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    DisabledException,
    ResourceNotFoundException,
    ValidationException,
)
from src.security import JWTParams, require_auth
from src.utils import history

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions",
    tags=["sessions"],
)


@router.post(
    "",
    response_model=schemas.Session,
)
async def get_or_create_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session: schemas.SessionCreate = Body(
        ..., description="Session creation parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get a specific session in a workspace.

    If session_id is provided as a query parameter, it verifies the session is in the workspace.
    Otherwise, it uses the session_id from the JWT token for verification.
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
        return await crud.get_or_create_session(
            db, workspace_name=workspace_id, session=session
        )
    except ValueError as e:
        logger.warning(f"Failed to get or create session {session.name}: {str(e)}")
        raise ValidationException(str(e)) from e


@router.post(
    "/list",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_sessions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    options: schemas.SessionGet | None = Body(
        None, description="Filtering and pagination options for the sessions list"
    ),
    db: AsyncSession = db,
):
    """Get All Sessions in a Workspace"""
    filter_param = None
    is_active_param = False  # Default from schema

    if options:
        if hasattr(options, "filter") and options.filter:
            filter_param = options.filter
            if filter_param == {}:  # Explicitly check for empty dict
                filter_param = None
        if hasattr(options, "is_active"):  # Check if is_active is present
            is_active_param = options.is_active

    return await paginate(
        db,
        await crud.get_sessions(
            workspace_name=workspace_id,
            is_active=is_active_param,
            filter=filter_param,
        ),
    )


@router.put(
    "/{session_id}",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def update_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to update"),
    session: schemas.SessionUpdate = Body(
        ..., description="Updated session parameters"
    ),
    db: AsyncSession = db,
):
    """Update the metadata of a Session"""
    try:
        updated_session = await crud.update_session(
            db, workspace_name=workspace_id, session_name=session_id, session=session
        )
        logger.info(f"Session {session_id} updated successfully")
        return updated_session
    except ValueError as e:
        logger.warning(f"Failed to update session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.delete(
    "/{session_id}",
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def delete_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to delete"),
    db: AsyncSession = db,
):
    """Delete a session by marking it as inactive"""
    try:
        await crud.delete_session(
            db, workspace_name=workspace_id, session_name=session_id
        )
        logger.info(f"Session {session_id} deleted successfully")
        return {"message": "Session deleted successfully"}
    except ValueError as e:
        logger.warning(f"Failed to delete session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/clone",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def clone_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to clone"),
    db: AsyncSession = db,
    message_id: str | None = Query(
        None, description="Message ID to cut off the clone at"
    ),
):
    """Clone a session, optionally up to a specific message"""
    try:
        # TODO: Update crud.clone_session to work with new paradigm
        cloned_session = await crud.clone_session(
            db,
            workspace_name=workspace_id,
            original_session_name=session_id,
            cutoff_message_id=message_id,
        )
        logger.info(f"Session {session_id} cloned successfully")
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
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: dict[str, schemas.SessionPeerConfig] = Body(
        ..., description="List of peer IDs to add to the session"
    ),
    db: AsyncSession = db,
):
    """Add peers to a session"""
    try:
        session = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(
                name=session_id,
                peers=peers,
            ),
            workspace_name=workspace_id,
        )
        logger.info(f"Added peers to session {session_id} successfully")
        return session
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
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: dict[str, schemas.SessionPeerConfig] = Body(
        ..., description="List of peer IDs to set for the session"
    ),
    db: AsyncSession = db,
):
    """Set the peers in a session"""
    try:
        await crud.set_peers_for_session(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_names=peers,
        )
        # Get the session to return
        session = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(name=session_id),
            workspace_name=workspace_id,
        )
        logger.info(f"Set peers for session {session_id} successfully")
        return session
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
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: list[str] = Body(
        ..., description="List of peer IDs to remove from the session"
    ),
    db: AsyncSession = db,
):
    """Remove peers from a session"""
    try:
        await crud.remove_peers_from_session(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_names=set(peers),
        )
        # Get the session to return
        session = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(name=session_id),
            workspace_name=workspace_id,
        )
        logger.info(f"Removed peers from session {session_id} successfully")
        return session
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
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peer_id: str = Path(..., description="ID of the peer"),
    db: AsyncSession = db,
):
    """Get the configuration for a peer in a session"""
    return await crud.get_peer_config(
        db,
        workspace_name=workspace_id,
        session_name=session_id,
        peer_id=peer_id,
    )


@router.post(
    "/{session_id}/peers/{peer_id}/config",
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def set_peer_config(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peer_id: str = Path(..., description="ID of the peer"),
    config: schemas.SessionPeerConfig = Body(..., description="Peer configuration"),
    db: AsyncSession = db,
):
    """Set the configuration for a peer in a session"""
    try:
        await crud.set_peer_config(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_id=peer_id,
            config=config,
        )
        logger.info(
            f"Set peer config for {peer_id} in session {session_id} successfully"
        )
        return Response(status_code=200)
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
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    db: AsyncSession = db,
):
    """Get peers from a session"""
    try:
        peers_query = await crud.get_peers_from_session(
            workspace_name=workspace_id, session_name=session_id
        )
        return await paginate(db, peers_query)
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
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    tokens: int | None = Query(
        None,
        description="Number of tokens to use for the context. Includes summary if set to true",
    ),
    summary: bool = Query(
        False,
        description="Whether to summarize the session history prior to the cutoff message",
    ),  # default to false
    db: AsyncSession = db,
):
    """
    Produce a context object from the session. The caller provides a token limit which the entire context must fit into.
    To do this, we allocate 40% of the token limit to the summary, and 60% to recent messages -- as many as can fit.
    If the caller does not want a summary, we allocate all the tokens to recent messages.
    The default token limit if not provided is 2048. (TODO: make this configurable)
    """
    token_limit = tokens or 2048
    summary_tokens = int(token_limit * 0.4) if summary else 0
    messages_tokens = token_limit - summary_tokens

    # Get the messages to return verbatim
    messages_stmt = await crud.get_messages(
        workspace_name=workspace_id,
        session_name=session_id,
        token_limit=messages_tokens,
    )
    result = await db.execute(messages_stmt)
    messages = list(result.scalars().all())

    # Get the most recently created summary for the session
    last_summary = await history.get_summary(
        db,
        workspace_name=workspace_id,
        session_name=session_id,
    )

    # Get messages between the last summary and the first message we'll return verbatim, if any
    messages_before = await crud.get_messages_id_range(
        db,
        workspace_name=workspace_id,
        session_name=session_id,
        peer_name=None,
        start_id=last_summary["message_id"] if last_summary else 0,
        end_id=messages[0].id if messages else None,
    )

    # Make a summary if the user wants one
    if summary_tokens > 0:
        # Make a *new* summary if there are unsummarized messages between the last summary and the ones
        # we'll return verbatim, or if the last summary is too many tokens -- otherwise, just use the last summary
        if (
            not last_summary
            or len(messages_before) > 0
            or last_summary["token_count"] > summary_tokens
        ):
            new_summary = await history.create_summary(
                messages=messages_before,
                max_tokens=summary_tokens,
            )
            summary_content = new_summary["content"]
        else:
            summary_content = last_summary["content"]
            summary_tokens = last_summary["token_count"]
    else:
        summary_content = ""

    return schemas.SessionContext(
        name=session_id,
        messages=messages,  # pyright: ignore
        summary=summary_content,
    )


@router.post(
    "/{session_id}/search",
    response_model=Page[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def search_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    search: schemas.MessageSearchOptions = Body(
        ..., description="Message search parameters "
    ),
    db: AsyncSession = db,
):
    """Search a Session"""
    query, use_semantic_search = search.query, search.use_semantic_search
    embed_messages_enabled = settings.LLM.EMBED_MESSAGES
    if use_semantic_search and not embed_messages_enabled:
        raise DisabledException(
            "Semantic search requires EMBED_MESSAGES flag to be enabled"
        )

    stmt = await crud.search(
        query,
        workspace_name=workspace_id,
        session_name=session_id,
        use_semantic_search=use_semantic_search,
    )

    return await paginate(db, stmt)

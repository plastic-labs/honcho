import logging

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
    ValidationException,
)
from src.security import JWTParams, require_auth
from src.utils import summarizer

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

    if options and hasattr(options, "filter") and options.filter:
        filter_param = options.filter
        if filter_param == {}:  # Explicitly check for empty dict
            filter_param = None

    return await apaginate(
        db, await crud.get_sessions(workspace_name=workspace_id, filters=filter_param)
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
            peer_name=peer_id,
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

    logger.info(
        f"Context request for session {session_id}: token_limit={token_limit}, "
        + f"summary_tokens={summary_tokens}, messages_tokens={messages_tokens}, summary_requested={summary}"
    )

    # Get the recent messages to return verbatim
    messages_stmt = await crud.get_messages(
        workspace_name=workspace_id,
        session_name=session_id,
        token_limit=messages_tokens,
    )
    result = await db.execute(messages_stmt)
    messages = list(result.scalars().all())

    logger.info(
        f"Retrieved {len(messages)} recent messages for verbatim return (IDs: {[m.id for m in messages]})"
    )

    summary_content = ""

    if summary_tokens > 0 and messages:
        # Check if we should create a new cumulative summary
        (
            should_create,
            messages_to_summarize,
            latest_summary,
        ) = await summarizer.should_create_summary(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_name=None,  # Session-level summary
            message_id=messages[
                0
            ].id,  # Cutoff at the first message we'll return verbatim
            summary_type=summarizer.SummaryType.SHORT,
        )

        # Check for gaps: if latest summary exists but doesn't cover up to the recent messages,
        # we have a gap that must be filled regardless of the threshold
        has_gap = False
        if latest_summary and messages:
            gap_start = latest_summary["message_id"] + 1
            gap_end = messages[0].id
            has_gap = gap_start < gap_end
            if has_gap:
                logger.info(
                    f"Gap detected: summary ends at message {latest_summary['message_id']}, recent messages start at {messages[0].id} (missing messages {gap_start}-{gap_end - 1})"
                )

        logger.info(
            f"Summary decision: should_create={should_create}, "
            + f"unsummarized_messages={len(messages_to_summarize)}, "
            + f"has_existing_summary={latest_summary is not None}, "
            + f"has_gap={has_gap}"
        )

        if latest_summary:
            logger.info(
                f"Existing summary covers {latest_summary['message_count']} messages "
                + f"up to message {latest_summary['message_id']}, "
                + f"token_count={latest_summary['token_count']}"
            )

        # We must create a new summary if either:
        # 1. The threshold is met (should_create=True), OR
        # 2. There's a gap between existing summary and recent messages
        must_create_summary = should_create or has_gap

        if must_create_summary:
            # Create a new cumulative summary covering ALL messages from the beginning
            # up to the start of the recent messages
            all_messages_to_summarize = await crud.get_messages_id_range(
                db,
                workspace_name=workspace_id,
                session_name=session_id,
                peer_name=None,
                start_id=0,
                end_id=messages[0].id,
            )

            if has_gap:
                logger.info(
                    f"Creating NEW cumulative summary to fill gap: covering {len(all_messages_to_summarize)} messages "
                    + f"from start to message {messages[0].id}"
                )
            else:
                logger.info(
                    f"Creating NEW cumulative summary (threshold met): covering {len(all_messages_to_summarize)} messages "
                    + f"from start to message {messages[0].id}"
                )

            if all_messages_to_summarize:
                # Create cumulative summary
                new_summary = await summarizer.create_summary(
                    messages=all_messages_to_summarize,
                    previous_summary_text=None,  # Start fresh for cumulative summary
                    summary_type=summarizer.SummaryType.SHORT,
                    max_tokens=summary_tokens,
                )

                # Save the new cumulative summary
                await summarizer.save_summary(
                    db,
                    summary=new_summary,
                    workspace_name=workspace_id,
                    session_name=session_id,
                )
                summary_content = new_summary["content"]
                logger.info(
                    f"Saved new cumulative summary with {new_summary['token_count']} tokens"
                )
            else:
                summary_content = ""
                logger.info("No messages to summarize, using empty summary")

        elif latest_summary:
            # Use existing summary if it fits within token limit and there's no gap
            if latest_summary["token_count"] <= summary_tokens:
                summary_content = latest_summary["content"]
                logger.info(
                    f"Reusing existing summary ({latest_summary['token_count']} tokens fits in {summary_tokens} limit)"
                )
            else:
                # Existing summary is too big - truncate it
                # This is a simple truncation - could be improved with smarter trimming
                summary_content = latest_summary["content"][
                    : summary_tokens * 4
                ]  # Rough estimate: 4 chars per token
                logger.info(
                    f"Truncated existing summary to fit {summary_tokens} token limit"
                )
        else:
            # No existing summary and not enough messages to create one
            summary_content = ""
            logger.info(
                "No existing summary and insufficient messages to create new summary"
            )
    else:
        if summary_tokens == 0:
            logger.info("Summary not requested, returning messages only")
        else:
            logger.info("No messages available for summarization")

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
    query, semantic = search.query, search.semantic

    stmt = await crud.search(
        query,
        workspace_name=workspace_id,
        session_name=session_id,
        semantic=semantic,
    )

    return await apaginate(db, stmt)

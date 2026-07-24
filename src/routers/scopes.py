"""FastAPI routes for scope resources.

A scope is a named grouping of sessions that provides a visibility boundary
within a peer. Internally a scope is a peer named ``scope__<name>`` that
observes its member sessions and never speaks; these routes are the facade
that keeps the observer/observed mechanics hidden.

All scopes routes require a workspace-level (or admin) key: scopes are an
app-level admin surface, so peer- and session-scoped keys are rejected.

Note: backfill of pre-existing documents and reconciliation on removal land
in a follow-up (DEV-1999) — for now, scope membership only affects messages
ingested *after* the membership change.
"""

import logging

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db, read_db
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/scopes",
    tags=["scopes"],
)


@router.post(
    "",
    response_model=schemas.Scope,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_or_create_scope(
    response: Response,
    workspace_id: str = Path(...),
    scope: schemas.ScopeCreate = Body(..., description="Scope creation parameters"),
    db: AsyncSession = db,
):
    """
    Get a Scope by ID or create a new Scope with the given ID.

    Returns 201 when the scope is created and 200 when it already exists.
    A pre-existing peer occupying the scope's reserved internal name is never
    adopted; that conflict returns 409.
    """
    result = await crud.get_or_create_scopes(db, workspace_id, [scope])
    await db.commit()
    await result.post_commit()
    response.status_code = 201 if result.created else 200
    return result.resource[0]


@router.post(
    "/list",
    response_model=Page[schemas.Scope],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_scopes(
    workspace_id: str = Path(...),
    reverse: bool = Query(False, description="Whether to reverse the order of results"),
    db: AsyncSession = read_db,
):
    """Get all Scopes for a Workspace. Results are paginated."""
    return await apaginate(
        db,
        await crud.get_scopes(workspace_name=workspace_id, reverse=reverse),
    )


@router.get(
    "/{scope_id}",
    response_model=schemas.Scope,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_scope(
    workspace_id: str = Path(...),
    scope_id: str = Path(...),
    db: AsyncSession = read_db,
):
    """Get a single Scope by ID."""
    return await crud.get_scope(db, workspace_id, scope_id)


@router.post(
    "/{scope_id}/sessions",
    response_model=schemas.ScopeSessions,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def add_sessions_to_scope(
    workspace_id: str = Path(...),
    scope_id: str = Path(...),
    body: schemas.ScopeSessionsAdd = Body(
        ..., description="IDs of the sessions to add to the scope"
    ),
    db: AsyncSession = db,
):
    """
    Add Sessions to a Scope.

    All named sessions must already exist (404 otherwise). Returns the scope's
    full member session list after the addition.

    Note: membership only affects messages ingested after this call — backfill
    of pre-existing documents lands in a follow-up (DEV-1999).
    """
    session_ids = await crud.add_sessions_to_scope(
        db,
        workspace_name=workspace_id,
        scope_name=scope_id,
        session_names=body.session_ids,
    )
    return schemas.ScopeSessions(session_ids=session_ids)


@router.delete(
    "/{scope_id}/sessions/{session_id}",
    status_code=204,
    response_model=None,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def remove_session_from_scope(
    workspace_id: str = Path(...),
    scope_id: str = Path(...),
    session_id: str = Path(...),
    db: AsyncSession = db,
):
    """
    Remove a Session from a Scope.

    Note: documents already derived while the session was a member are left in
    place — reconciliation on removal lands in a follow-up (DEV-1999).
    """
    await crud.remove_session_from_scope(
        db,
        workspace_name=workspace_id,
        scope_name=scope_id,
        session_name=session_id,
    )


@router.get(
    "/{scope_id}/sessions",
    response_model=schemas.ScopeSessions,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_scope_sessions(
    workspace_id: str = Path(...),
    scope_id: str = Path(...),
    db: AsyncSession = read_db,
):
    """Get the IDs of the Sessions that are members of a Scope."""
    session_ids = await crud.get_scope_session_names(db, workspace_id, scope_id)
    return schemas.ScopeSessions(session_ids=session_ids)

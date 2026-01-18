"""Falsification traces router for read-only access to falsification records.

This module provides GET-only endpoints for querying falsification traces.
Traces are immutable records of the Falsifier agent's attempts to find
contradictory evidence for predictions. They cannot be created or modified via the API.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.dependencies import db
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/traces",
    tags=["reasoning"],
    responses={
        404: {"description": "Workspace or trace not found"},
        401: {"description": "Unauthorized - invalid or missing authentication"},
    },
)


@router.get(
    "",
    response_model=Page[schemas.TraceResponse],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def list_traces(
    workspace_id: str = Path(..., description="Workspace ID"),
    prediction_id: str | None = Query(None, description="Filter by prediction ID"),
    final_status: str | None = Query(
        None, description="Filter by final status (falsified/unfalsified)"
    ),
    db: AsyncSession = db,
):
    """
    List all falsification traces for a workspace with optional filters.

    Falsification traces are immutable records of the Falsifier agent's work.
    Each trace documents:
    - Search queries executed to find contradictions
    - Contradicting premises found (if any)
    - Reasoning chain explaining the determination
    - Search efficiency metrics

    Query Parameters:
    - prediction_id: Filter by the prediction being tested
    - final_status: Filter by the final determination (falsified/unfalsified)
    """
    stmt = select(models.FalsificationTrace).where(
        models.FalsificationTrace.workspace_name == workspace_id
    )

    if prediction_id:
        stmt = stmt.where(models.FalsificationTrace.prediction_id == prediction_id)
    if final_status:
        stmt = stmt.where(models.FalsificationTrace.final_status == final_status)

    stmt = stmt.order_by(models.FalsificationTrace.created_at.desc())

    return await apaginate(db, stmt)


@router.get(
    "/{trace_id}",
    response_model=schemas.TraceResponse,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_trace(
    workspace_id: str = Path(..., description="Workspace ID"),
    trace_id: str = Path(..., description="Trace ID"),
    db: AsyncSession = db,
):
    """
    Get a specific falsification trace by ID.

    Returns the complete trace including all search queries, contradicting
    premises, reasoning chain, and efficiency metrics. This provides full
    transparency into how the Falsifier agent tested the prediction.
    """
    trace = await crud.trace.get_trace(
        db, workspace_name=workspace_id, trace_id=trace_id
    )

    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return trace


@router.get(
    "/prediction/{prediction_id}",
    response_model=Page[schemas.TraceResponse],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_traces_for_prediction(
    workspace_id: str = Path(..., description="Workspace ID"),
    prediction_id: str = Path(..., description="Prediction ID"),
    db: AsyncSession = db,
):
    """
    Get all falsification traces for a specific prediction.

    This is a convenience endpoint that returns all traces associated with
    a prediction, ordered by creation time. Multiple traces can exist if
    the prediction was retested.

    This endpoint is equivalent to:
    GET /traces?prediction_id={prediction_id}
    """
    # Verify prediction exists
    prediction = await crud.prediction.get_prediction(
        db, workspace_name=workspace_id, prediction_id=prediction_id
    )
    if not prediction:
        raise HTTPException(
            status_code=404, detail=f"Prediction {prediction_id} not found"
        )

    stmt = select(models.FalsificationTrace).where(
        models.FalsificationTrace.workspace_name == workspace_id,
        models.FalsificationTrace.prediction_id == prediction_id,
    )

    stmt = stmt.order_by(models.FalsificationTrace.created_at.desc())

    return await apaginate(db, stmt)

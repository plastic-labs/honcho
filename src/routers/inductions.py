"""Inductions router for read-only access to pattern extraction artifacts.

This module provides GET-only endpoints for querying inductions (patterns)
extracted during reasoning dreams. Inductions are created exclusively by the
Inductor agent and cannot be created or modified via the API.
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
    prefix="/workspaces/{workspace_id}/inductions",
    tags=["reasoning"],
    responses={
        404: {"description": "Workspace or induction not found"},
        401: {"description": "Unauthorized - invalid or missing authentication"},
    },
)


@router.get(
    "",
    response_model=Page[schemas.InductionResponse],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def list_inductions(
    workspace_id: str = Path(..., description="Workspace ID"),
    observer: str | None = Query(None, description="Filter by observer peer"),
    observed: str | None = Query(None, description="Filter by observed peer"),
    pattern_type: str | None = Query(
        None,
        description="Filter by pattern type (preference/behavior/personality/tendency/temporal/conditional/structural/correlational)",
    ),
    confidence: str | None = Query(None, description="Filter by confidence level (high/medium/low)"),
    db: AsyncSession = db,
):
    """
    List all inductions for a workspace with optional filters.

    Inductions are general patterns extracted by the Inductor agent from
    clusters of unfalsified predictions. They represent stable patterns in
    behavior or preferences discovered through the reasoning process.

    Query Parameters:
    - observer: Filter by observer peer name
    - observed: Filter by observed peer name
    - pattern_type: Filter by the type of pattern identified
    - confidence: Filter by confidence level (high/medium/low)
    """
    stmt = select(models.Induction).where(
        models.Induction.workspace_name == workspace_id
    )

    if observer:
        stmt = stmt.where(models.Induction.observer == observer)
    if observed:
        stmt = stmt.where(models.Induction.observed == observed)
    if pattern_type:
        stmt = stmt.where(models.Induction.pattern_type == pattern_type)
    if confidence:
        stmt = stmt.where(models.Induction.confidence == confidence)

    stmt = stmt.order_by(models.Induction.created_at.desc())

    return await apaginate(db, stmt)


@router.get(
    "/{induction_id}",
    response_model=schemas.InductionResponse,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_induction(
    workspace_id: str = Path(..., description="Workspace ID"),
    induction_id: str = Path(..., description="Induction ID"),
    db: AsyncSession = db,
):
    """
    Get a specific induction by ID.

    Returns detailed information about a single induction including its
    pattern description, type, confidence level, stability score, and
    source predictions/premises.
    """
    induction = await crud.induction.get_induction(
        db, workspace_id, induction_id
    )

    if not induction:
        raise HTTPException(
            status_code=404, detail=f"Induction {induction_id} not found"
        )

    return induction


@router.get(
    "/{induction_id}/sources",
    response_model=schemas.InductionSources,
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_induction_sources(
    workspace_id: str = Path(..., description="Workspace ID"),
    induction_id: str = Path(..., description="Induction ID"),
    db: AsyncSession = db,
):
    """
    Get the source predictions and premises that formed an induction.

    Returns:
    - The induction itself
    - Source predictions (unfalsified predictions that formed the pattern)
    - Source premises (original observations that led to the predictions)

    This provides full transparency into how the pattern was discovered.
    """
    induction = await crud.induction.get_induction(
        db, workspace_id, induction_id
    )
    if not induction:
        raise HTTPException(
            status_code=404, detail=f"Induction {induction_id} not found"
        )

    # Get source predictions
    source_predictions = []
    if induction.source_prediction_ids:
        pred_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_id,
            models.Prediction.id.in_(induction.source_prediction_ids),
        )
        pred_result = await db.execute(pred_stmt)
        source_predictions = list(pred_result.scalars().all())

    # Get source premises (observations)
    source_premises = []
    if induction.source_premise_ids:
        prem_stmt = select(models.Document).where(
            models.Document.workspace_name == workspace_id,
            models.Document.id.in_(induction.source_premise_ids),
        )
        prem_result = await db.execute(prem_stmt)
        source_premises = list(prem_result.scalars().all())

    return {
        "induction": induction,
        "source_predictions": source_predictions,
        "source_premises": source_premises,
    }

"""CRUD operations for Prediction model."""

from logging import getLogger
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.workspace import get_or_create_workspace
from src.embedding_client import embedding_client
from src.exceptions import ResourceNotFoundException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def create_prediction(
    db: AsyncSession,
    prediction: schemas.PredictionCreate,
    workspace_name: str,
) -> models.Prediction:
    """
    Create a new prediction with vector embedding.

    Args:
        db: Database session
        prediction: Prediction creation schema
        workspace_name: Name of the workspace

    Returns:
        Created prediction object

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    # Ensure workspace exists
    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))

    # Generate embedding for the prediction content
    embedding = await embedding_client.embed(prediction.content)

    # Create prediction object
    prediction_obj = models.Prediction(
        content=prediction.content,
        hypothesis_id=prediction.hypothesis_id,
        status=prediction.status or "untested",
        source_hypothesis_ids=prediction.source_hypothesis_ids,
        is_blind=prediction.is_blind if prediction.is_blind is not None else True,
        workspace_name=workspace_name,
        collection_id=prediction.collection_id,
        embedding=embedding,
    )

    db.add(prediction_obj)
    await db.commit()
    await db.refresh(prediction_obj)

    logger.debug(
        "Prediction %s created successfully in workspace %s",
        prediction_obj.id,
        workspace_name,
    )
    return prediction_obj


async def get_prediction(
    db: AsyncSession,
    workspace_name: str,
    prediction_id: str,
) -> models.Prediction:
    """
    Get a prediction by ID.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        prediction_id: ID of the prediction

    Returns:
        The prediction if found

    Raises:
        ResourceNotFoundException: If the prediction does not exist
    """
    stmt = (
        select(models.Prediction)
        .where(models.Prediction.workspace_name == workspace_name)
        .where(models.Prediction.id == prediction_id)
    )
    result = await db.execute(stmt)
    prediction = result.scalar_one_or_none()

    if prediction is None:
        raise ResourceNotFoundException(
            f"Prediction {prediction_id} not found in workspace {workspace_name}"
        )

    return prediction


async def update_prediction(
    db: AsyncSession,
    workspace_name: str,
    prediction_id: str,
    prediction_update: schemas.PredictionUpdate,
) -> models.Prediction:
    """
    Update a prediction.

    Note: Predictions are mostly immutable. Only status can be updated.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        prediction_id: ID of the prediction to update
        prediction_update: Prediction update schema

    Returns:
        The updated prediction

    Raises:
        ResourceNotFoundException: If the prediction does not exist
    """
    # Get existing prediction
    prediction = await get_prediction(db, workspace_name, prediction_id)

    # Track if any changes were made
    needs_update = False

    # Update status if provided
    if prediction_update.status is not None:
        prediction.status = prediction_update.status
        needs_update = True

    # Early exit if unchanged
    if not needs_update:
        logger.debug(
            "Prediction %s unchanged in workspace %s, skipping update",
            prediction_id,
            workspace_name,
        )
        return prediction

    await db.commit()
    await db.refresh(prediction)

    logger.debug(
        "Prediction %s updated successfully in workspace %s",
        prediction_id,
        workspace_name,
    )
    return prediction


async def delete_prediction(
    db: AsyncSession,
    workspace_name: str,
    prediction_id: str,
) -> bool:
    """
    Delete a prediction.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        prediction_id: ID of the prediction to delete

    Returns:
        True if deleted successfully

    Raises:
        ResourceNotFoundException: If the prediction does not exist
    """
    # Get existing prediction (will raise if not found)
    prediction = await get_prediction(db, workspace_name, prediction_id)

    await db.delete(prediction)
    await db.commit()

    logger.debug(
        "Prediction %s deleted successfully from workspace %s",
        prediction_id,
        workspace_name,
    )
    return True


async def list_predictions(
    workspace_name: str,
    hypothesis_id: str | None = None,
    collection_id: str | None = None,
    status: str | None = None,
    is_blind: bool | None = None,
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.Prediction]]:
    """
    List predictions with optional filtering.

    Args:
        workspace_name: Name of the workspace
        hypothesis_id: Filter by hypothesis ID
        collection_id: Filter by collection ID
        status: Filter by status (unfalsified | falsified | untested)
        is_blind: Filter by blind prediction status
        filters: Additional metadata filters

    Returns:
        SQLAlchemy Select statement for the predictions
    """
    stmt = select(models.Prediction).where(
        models.Prediction.workspace_name == workspace_name
    )

    # Apply optional filters
    if hypothesis_id is not None:
        stmt = stmt.where(models.Prediction.hypothesis_id == hypothesis_id)

    if collection_id is not None:
        stmt = stmt.where(models.Prediction.collection_id == collection_id)

    if status is not None:
        stmt = stmt.where(models.Prediction.status == status)

    if is_blind is not None:
        stmt = stmt.where(models.Prediction.is_blind == is_blind)

    # Apply metadata filters
    stmt = apply_filter(stmt, models.Prediction, filters)

    # Order by created_at descending (most recent first)
    stmt = stmt.order_by(models.Prediction.created_at.desc())

    return stmt


async def search_predictions(
    db: AsyncSession,
    workspace_name: str,
    query: str,
    hypothesis_id: str | None = None,
    limit: int = 10,
) -> list[models.Prediction]:
    """
    Search for predictions using semantic similarity.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        query: Search query text
        hypothesis_id: Optional filter by hypothesis ID
        limit: Maximum number of predictions to return

    Returns:
        List of predictions ordered by semantic similarity
    """
    # Generate embedding for the search query
    query_embedding = await embedding_client.embed(query)

    # Build query with vector similarity
    stmt = (
        select(models.Prediction)
        .where(models.Prediction.workspace_name == workspace_name)
        .order_by(models.Prediction.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )

    # Apply optional hypothesis filter
    if hypothesis_id is not None:
        stmt = stmt.where(models.Prediction.hypothesis_id == hypothesis_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())

from __future__ import annotations

import datetime
import logging
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, exceptions, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import check_and_schedule_dream
from src.embedding_client import embedding_client
from src.utils.formatting import format_datetime_utc
from src.utils.logging import accumulate_metric
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)

logger = logging.getLogger(__name__)

# Fetch extra documents to ensure we have enough after filtering
FILTER_OVERSAMPLING_FACTOR = 3


class RepresentationManager:
    """Unified manager for representation and document queries."""

    def __init__(
        self,
        workspace_name: str,
        *,
        observer: str,
        observed: str,
    ) -> None:
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed

    async def save_representation(
        self,
        representation: Representation,
        message_id_range: tuple[int, int],
        session_name: str,
        message_created_at: datetime.datetime,
    ) -> int:
        """
        Save Representation objects to the collection as a set of documents.

        Args:
            representation: Representation object
            message_id_range: Message ID range to link with observations
            session_name: Session name to link with existing summary context
            message_created_at: Timestamp when the message was created

        Returns:
            The number of *new documents saved*
        """

        new_documents = 0

        if not representation.deductive and not representation.explicit:
            logger.debug("No observations to save")
            return new_documents

        all_observations = representation.deductive + representation.explicit

        # Batch embed all observations
        batch_embed_start = time.perf_counter()

        observation_texts = [
            obs.conclusion if isinstance(obs, DeductiveObservation) else obs.content
            for obs in all_observations
        ]
        try:
            embeddings = await embedding_client.simple_batch_embed(observation_texts)
        except ValueError as e:
            raise exceptions.ValidationException(
                f"Observation content exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}."
            ) from e

        batch_embed_duration = (time.perf_counter() - batch_embed_start) * 1000
        accumulate_metric(
            f"deriver_{message_id_range[1]}_{self.observer}",
            "embed_new_observations",
            batch_embed_duration,
            "ms",
        )

        # Batch create document objects
        create_document_start = time.perf_counter()
        async with tracked_db("representation_manager.save_representation") as db:
            new_documents = await self._save_representation_internal(
                db,
                all_observations,
                embeddings,
                message_id_range,
                session_name,
                message_created_at,
            )

        create_document_duration = (time.perf_counter() - create_document_start) * 1000
        accumulate_metric(
            f"deriver_{message_id_range[1]}_{self.observer}",
            "save_new_observations",
            create_document_duration,
            "ms",
        )

        return new_documents

    async def _save_representation_internal(
        self,
        db: AsyncSession,
        all_observations: list[ExplicitObservation | DeductiveObservation],
        embeddings: list[list[float]],
        message_id_range: tuple[int, int],
        session_name: str,
        message_created_at: datetime.datetime,
    ) -> int:
        # get_or_create_collection already handles IntegrityError with rollback and a retry
        collection = await crud.get_or_create_collection(
            db,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
        )

        # Prepare all documents for bulk creation
        documents_to_create: list[schemas.DocumentCreate] = []
        for obs, embedding in zip(all_observations, embeddings, strict=True):
            # NOTE: will add additional levels of reasoning in the future
            if isinstance(obs, DeductiveObservation):
                obs_level = "deductive"
                obs_content = obs.conclusion
                obs_premises = obs.premises
            else:
                obs_level = "explicit"
                obs_content = obs.content
                obs_premises = None

            metadata: schemas.DocumentMetadata = schemas.DocumentMetadata(
                message_ids=[message_id_range],
                premises=obs_premises,
                message_created_at=format_datetime_utc(message_created_at),
            )

            documents_to_create.append(
                schemas.DocumentCreate(
                    content=obs_content,
                    session_name=session_name,
                    level=obs_level,
                    metadata=metadata,
                    embedding=embedding,
                )
            )

        # Use bulk creation with optional duplicate detection
        new_documents = await crud.create_documents(
            db,
            documents_to_create,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            deduplicate=settings.DERIVER.DEDUPLICATE,
        )

        try:
            await check_and_schedule_dream(db, collection)
        except Exception as e:
            logger.warning(f"Failed to check dream scheduling: {e}")

        return new_documents

    async def get_working_representation(
        self,
        *,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    ) -> Representation:
        """
        Get working representation with flexible query options.

        Args:
            session_name: Optional session to filter by
            include_semantic_query: Query for semantic search
            semantic_search_top_k: Number of semantic results
            semantic_search_max_distance: Maximum distance for semantic search
            include_most_derived: Include most derived observations
            max_observations: Maximum total observations to return

        Returns:
            Representation combining various query strategies
        """
        async with tracked_db(
            "representation_manager.get_working_representation"
        ) as db:
            return await self._get_working_representation_internal(
                db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                include_most_derived=include_most_derived,
                max_observations=max_observations,
            )

    # Private helper methods

    async def _get_working_representation_internal(
        self,
        db: AsyncSession,
        *,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        include_most_derived: bool = False,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    ) -> Representation:
        """Internal implementation of get_working_representation."""
        total = max_observations

        # Calculate how many observations to get from each source
        semantic_observations = (
            min(
                max(
                    0,
                    semantic_search_top_k
                    if semantic_search_top_k is not None
                    else total // 3,
                ),
                total,
            )
            if include_semantic_query
            else 0
        )

        if include_semantic_query and include_most_derived:
            # three-way blend: both semantic and derived requested
            top_observations = min(max(0, total // 3), total - semantic_observations)
        elif include_most_derived:
            # two-way blend: only derived requested
            top_observations = min(max(0, total // 2), total - semantic_observations)
        else:
            # no derived observations requested
            top_observations = 0

        # remaining observations are recent
        recent_observations = total - semantic_observations - top_observations

        representation = Representation()

        # Get semantic observations if requested
        if include_semantic_query:
            semantic_docs = await self._query_documents_semantic(
                db,
                query=include_semantic_query,
                top_k=semantic_observations,
                max_distance=semantic_search_max_distance,
            )
            representation.merge_representation(
                Representation.from_documents(semantic_docs)
            )

        # Get most derived observations if requested
        if include_most_derived:
            derived_docs = await self._query_documents_most_derived(
                db, top_k=top_observations
            )
            representation.merge_representation(
                Representation.from_documents(derived_docs)
            )

        # Get recent observations
        recent_docs = await self._query_documents_recent(
            db, top_k=recent_observations, session_name=session_name
        )

        representation.merge_representation(Representation.from_documents(recent_docs))

        return representation

    async def _query_documents_semantic(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float | None = None,
        level: str | None = None,
    ) -> list[models.Document]:
        """Query documents by semantic similarity."""
        try:
            if level:
                return await self._query_documents_for_level(
                    db,
                    query,
                    level,
                    top_k,
                    max_distance,
                )
            else:
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=query,
                    max_distance=max_distance,
                    top_k=top_k,
                )
                db.expunge_all()
                return list(documents)

        except Exception as e:
            logger.error(f"Error getting relevant observations: {e}")
            return []

    async def _query_documents_recent(
        self, db: AsyncSession, top_k: int, session_name: str | None = None
    ) -> list[models.Document]:
        """Query most recent documents."""
        stmt = (
            select(models.Document)
            .limit(top_k)
            .where(
                models.Document.workspace_name == self.workspace_name,
                models.Document.observer == self.observer,
                models.Document.observed == self.observed,
                *(
                    [models.Document.session_name == session_name]
                    if session_name is not None
                    else []
                ),
            )
            .order_by(models.Document.created_at.desc())
        )

        result = await db.execute(stmt)
        documents = result.scalars().all()
        db.expunge_all()
        return list(documents)

    async def _query_documents_most_derived(
        self, db: AsyncSession, top_k: int
    ) -> list[models.Document]:
        """Query most derived documents."""
        stmt = (
            select(models.Document)
            .limit(top_k)
            .where(
                models.Document.workspace_name == self.workspace_name,
                models.Document.observer == self.observer,
                models.Document.observed == self.observed,
            )
            .order_by(models.Document.times_derived.desc())
        )

        result = await db.execute(stmt)
        documents = result.scalars().all()
        db.expunge_all()
        return list(documents)

    async def _get_observations_internal(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float,
        level: str | None,
    ) -> list[models.Document]:
        """Internal method that does the actual observation retrieval."""
        return await self._query_documents_semantic(
            db, query, top_k, max_distance, level
        )

    async def _query_documents_for_level(
        self,
        db: AsyncSession,
        query: str,
        level: str,
        count: int,
        max_distance: float | None = None,
    ) -> list[models.Document]:
        """Query documents for a specific level."""
        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            query=query,
            max_distance=max_distance,
            top_k=count * FILTER_OVERSAMPLING_FACTOR,
            filters=self._build_filter_conditions(level),
        )

        # Sort by creation time and return top count
        docs_sorted: list[models.Document] = sorted(
            list(documents), key=lambda x: x.created_at, reverse=True
        )
        return docs_sorted[:count]

    def _build_filter_conditions(
        self,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Build complete filter conditions for document queries."""
        conditions: list[dict[str, Any]] = []

        if level:
            conditions.append({"level": level})

        if not conditions:
            return {}

        return conditions[0] if len(conditions) == 1 else {"AND": conditions}


# Module-level functions for backward compatibility and convenience


async def get_working_representation(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    session_name: str | None = None,
    include_semantic_query: str | None = None,
    semantic_search_top_k: int | None = None,
    semantic_search_max_distance: float | None = None,
    include_most_derived: bool = False,
    max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
) -> Representation:
    """
    Get raw working representation data from the relevant document collection.

    This is a convenience function that creates a RepresentationManager and calls
    get_working_representation on it.
    """
    manager = RepresentationManager(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )
    return await manager.get_working_representation(
        session_name=session_name,
        include_semantic_query=include_semantic_query,
        semantic_search_top_k=semantic_search_top_k,
        semantic_search_max_distance=semantic_search_max_distance,
        include_most_derived=include_most_derived,
        max_observations=max_observations,
    )

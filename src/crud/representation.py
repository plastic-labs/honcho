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
from src.utils.logging import accumulate_metric, conditional_observe
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

    @conditional_observe
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

        # Use bulk creation with NO duplicate detection
        new_documents = await crud.create_documents(
            db,
            documents_to_create,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
        )

        try:
            await check_and_schedule_dream(db, collection)
        except Exception as e:
            logger.warning(f"Failed to check dream scheduling: {e}")

        return new_documents

    async def get_relevant_observations(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_distance: float = 0.3,
        level: str | None = None,
        conversation_context: str = "",
    ) -> Representation:
        """
        Unified method to get relevant observations with flexible options.

        Args:
            query: The search query
            top_k: Number of results to return
            max_distance: Maximum distance for semantic similarity
            level: Optional reasoning level to filter by
            conversation_context: Additional conversation context

        Returns:
            Representation
        """
        async with tracked_db("representation_manager.get_relevant_observations") as db:
            documents = await self._get_observations_internal(
                db,
                query,
                top_k,
                max_distance,
                level,
                conversation_context,
            )

        # convert documents to representation
        return Representation.from_documents(documents)

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
        conversation_context: str = "",
    ) -> list[models.Document]:
        """Query documents by semantic similarity."""
        try:
            if level:
                return await self._query_documents_for_level(
                    db,
                    query,
                    level,
                    conversation_context,
                    top_k,
                    max_distance,
                )
            else:
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=self._build_truncated_query(query, conversation_context),
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
        conversation_context: str,
    ) -> list[models.Document]:
        """Internal method that does the actual observation retrieval."""
        return await self._query_documents_semantic(
            db, query, top_k, max_distance, level, conversation_context
        )

    async def _query_documents_for_level(
        self,
        db: AsyncSession,
        query: str,
        level: str,
        conversation_context: str,
        count: int,
        max_distance: float | None = None,
    ) -> list[models.Document]:
        """Query documents for a specific level."""
        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            query=self._build_truncated_query(query, conversation_context),
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

    def _build_truncated_query(
        self,
        query: str,
        conversation_context: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Build a query that fits within token limits with clear priorities.

        Args:
            query: The search query
            conversation_context: Optional conversation context to include
            max_tokens: Maximum tokens allowed (defaults to setting with buffer)

        Returns:
            Truncated query string that fits within token limits
        """
        max_tokens = max_tokens or (settings.MAX_EMBEDDING_TOKENS - 100)
        encoding = embedding_client.encoding

        # Pre-calculate all token counts once
        query_prefix = "Current message: "
        context_prefix = "\nContext: "

        prefix_tokens = len(encoding.encode(query_prefix))
        context_prefix_tokens = len(encoding.encode(context_prefix))
        query_tokens = encoding.encode(query)

        # Simple case: query alone fits
        if prefix_tokens + len(query_tokens) <= max_tokens:
            if not conversation_context:
                return f"{query_prefix}{query}"

            # Try to add context
            context_tokens = encoding.encode(conversation_context)
            total_without_context = (
                prefix_tokens + len(query_tokens) + context_prefix_tokens
            )

            if total_without_context + len(context_tokens) <= max_tokens:
                return f"{query_prefix}{query}{context_prefix}{conversation_context}"

            # Truncate context to fit
            available_context_tokens = max_tokens - total_without_context
            if available_context_tokens > 0:
                truncated_context = encoding.decode(
                    context_tokens[-available_context_tokens:]
                )
                return f"{query_prefix}{query}{context_prefix}{truncated_context}"
            else:
                # No room left for context; keep full query intact
                return f"{query_prefix}{query}"

        # Query itself is too long - truncate it
        available_query_tokens = max_tokens - prefix_tokens
        if available_query_tokens > 0:
            # Keep the end (recency) of the query
            truncated_query = encoding.decode(query_tokens[-available_query_tokens:])
            return f"{query_prefix}{truncated_query}"

        # Pathological case - just return what we can
        logger.warning("Token limit too restrictive: %s", max_tokens)
        return encoding.decode(query_tokens[:max_tokens])


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

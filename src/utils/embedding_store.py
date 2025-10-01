from __future__ import annotations

import datetime
import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import check_and_schedule_dream
from src.embedding_client import embedding_client
from src.exceptions import ValidationException
from src.utils.formatting import format_datetime_utc
from src.utils.langfuse_client import get_langfuse_client
from src.utils.logging import accumulate_metric, conditional_observe
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)

logger = logging.getLogger(__name__)

lf = get_langfuse_client()

# Fetch extra documents to ensure we have enough after filtering
FILTER_OVERSAMPLING_FACTOR = 3


class EmbeddingStore:
    """Embedding store specialized for observation-based reasoning with structured metadata."""

    def __init__(
        self,
        workspace_name: str,
        peer_name: str,
        collection_name: str,
        *,
        db: AsyncSession | None = None,
    ) -> None:
        self.workspace_name: str = workspace_name
        self.peer_name: str = peer_name
        self.collection_name: str = collection_name
        self.db: AsyncSession | None = db

    @conditional_observe
    async def save_representation(
        self,
        representation: Representation,
        message_id: int,
        session_name: str,
        message_created_at: datetime.datetime,
    ) -> int:
        """
        Save Representation objects to the collection as a set of documents.

        Args:
            representation: Representation object
            message_id: Message ID to link with observations
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
            raise ValidationException(
                f"Observation content exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}."
            ) from e

        batch_embed_duration = (time.perf_counter() - batch_embed_start) * 1000
        accumulate_metric(
            f"deriver_{message_id}_{self.peer_name}",
            "embed_new_observations",
            batch_embed_duration,
            "ms",
        )

        # Batch create document objects
        create_document_start = time.perf_counter()
        if self.db:
            new_documents = await self._save_representation_internal(
                self.db,
                all_observations,
                embeddings,
                message_id,
                session_name,
                message_created_at,
            )
        else:
            async with tracked_db("embedding_store.save_representation") as db:
                new_documents = await self._save_representation_internal(
                    db,
                    all_observations,
                    embeddings,
                    message_id,
                    session_name,
                    message_created_at,
                )

        create_document_duration = (time.perf_counter() - create_document_start) * 1000
        accumulate_metric(
            f"deriver_{message_id}_{self.peer_name}",
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
        message_id: int,
        session_name: str,
        message_created_at: datetime.datetime,
    ) -> int:
        # Prepare all documents for bulk creation
        documents_to_create: list[schemas.DocumentCreate] = []
        for obs in all_observations:
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
                message_id=message_id,
                session_name=session_name,
                level=obs_level,
                premises=obs_premises,
                message_created_at=format_datetime_utc(message_created_at),
            )

            documents_to_create.append(
                schemas.DocumentCreate(content=obs_content, metadata=metadata)
            )

        # Use bulk creation with NO duplicate detection
        _created_documents, new_documents = await crud.create_documents_bulk(
            db,
            documents_to_create,
            self.workspace_name,
            self.collection_name,
            self.peer_name,
            embeddings,
        )

        try:
            await check_and_schedule_dream(
                db, self.workspace_name, self.collection_name, self.peer_name
            )
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
            for_reasoning: If True, returns ObservationContext for ed reasoning

        Returns:
            Representation
        """
        async with tracked_db("embedding_store.get_relevant_observations") as db:
            documents = await self._get_observations_internal(
                db,
                query,
                top_k,
                max_distance,
                level,
                conversation_context,
            )

            # convert documents to representation
            # use level to filter documents
            return crud.representation_from_documents(documents)

    def _build_filter_conditions(
        self,
        level: str | None = None,
    ) -> dict[str, Any]:
        """Build complete filter conditions for document queries."""
        conditions: list[dict[str, Any]] = []

        if level:
            conditions.append({"internal_metadata": {"level": level}})

        if not conditions:
            return {}

        return conditions[0] if len(conditions) == 1 else {"AND": conditions}

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
        try:
            if level:
                return await self._query_documents_for_level(
                    db,
                    query,
                    level,
                    conversation_context,
                    max_distance,
                    top_k,
                )
            else:
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    peer_name=self.peer_name,
                    collection_name=self.collection_name,
                    query=self._build_truncated_query(query, ""),
                    max_distance=max_distance,
                    top_k=top_k,
                )
                db.expunge_all()
                return list(documents)

        except Exception as e:
            logger.error(f"Error getting relevant observations: {e}")
            return []

    async def _query_documents_for_level(
        self,
        db: AsyncSession,
        query: str,
        level: str,
        conversation_context: str,
        max_distance: float,
        count: int,
    ) -> list[models.Document]:
        """Query documents for a specific level."""
        # Construct the combined query with truncation to prevent token limit errors

        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            peer_name=self.peer_name,
            collection_name=self.collection_name,
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

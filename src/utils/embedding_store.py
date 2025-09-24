from __future__ import annotations

import datetime
import logging
from typing import Any, Literal, overload

from langfuse import get_client
from openai.types import CreateEmbeddingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.utils.formatting import format_datetime_utc
from src.utils.logging import conditional_observe
from src.utils.shared_models import (
    Observation,
    ObservationContext,
    ObservationMetadata,
    ReasoningLevel,
    UnifiedObservation,
)

logger = logging.getLogger(__name__)

lf = get_client()


class EmbeddingStore:
    """Embedding store specialized for observation-based reasoning with structured metadata."""

    def __init__(
        self, workspace_name: str, peer_name: str, collection_name: str
    ) -> None:
        self.workspace_name: str = workspace_name
        self.peer_name: str = peer_name
        self.collection_name: str = collection_name

    @conditional_observe
    async def save_unified_observations(
        self,
        observations: list[UnifiedObservation],
        message_id: int,
        session_name: str,
        message_created_at: datetime.datetime,
        fallback_level: str = "explicit",
        similarity_threshold: float = 0.85,
    ) -> None:
        """Save UnifiedObservation objects to the collection.

        This method handles UnifiedObservation objects by:
        1. Generating embeddings only from conclusions
        2. Storing premises in metadata for reference

        Args:
            observations: List of UnifiedObservation objects or strings
            message_id: Message ID to link with observations
            session_name: Session name to link with existing summary context
            message_created_at: Timestamp when the message was created
            fallback_level: Reasoning level for the observations if not provided
            similarity_threshold: Threshold for considering observations similar
        """
        # Extract conclusions for deduplication and embedding
        conclusions: list[str] = [obs.conclusion for obs in observations]

        # Remove duplicates before saving
        unique_conclusions: list[str] = await self.remove_duplicates(
            conclusions, similarity_threshold=similarity_threshold
        )
        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_observation(
                input={"observations": [obs.model_dump() for obs in observations]},
                output={"unique_conclusions": unique_conclusions},
            )

        if not unique_conclusions:
            logger.debug("No unique observations to save after deduplication")
            return

        # Create mapping from conclusion back to original observation
        conclusion_to_observation: dict[str, UnifiedObservation] = {
            obs.conclusion: obs for obs in observations
        }

        # Filter unified observations to only unique ones
        unique_observations: list[UnifiedObservation] = [
            conclusion_to_observation[conclusion] for conclusion in unique_conclusions
        ]

        # Batch embed all unique conclusions (not premises)
        embeddings: list[list[float]] = []
        batch_size: int = 2048  # OpenAI batch limit

        for i in range(0, len(unique_conclusions), batch_size):
            batch = unique_conclusions[i : i + batch_size]
            response: CreateEmbeddingResponse = (
                await embedding_client.client.embeddings.create(
                    input=batch, model="text-embedding-3-small"
                )
            )
            embeddings.extend([data.embedding for data in response.data])

        # Batch create document objects
        document_objects: list[models.Document] = []
        for obs, embedding in zip(unique_observations, embeddings, strict=True):
            # Use the observation's own level or fall back to parameter level
            obs_level = obs.level or fallback_level

            # Build metadata including premises
            metadata: dict[str, Any] = {
                "level": obs_level,
                "message_id": message_id,
                "session_name": session_name,
                "premises": obs.premises,  # Store premises in metadata
                "created_at": format_datetime_utc(message_created_at),
            }

            doc = models.Document(
                workspace_name=self.workspace_name,
                peer_name=self.peer_name,
                collection_name=self.collection_name,
                content=obs.conclusion,  # Store only conclusion as content
                internal_metadata=metadata,
                embedding=embedding,  # Embedding generated from conclusion only
                created_at=message_created_at,
            )
            document_objects.append(doc)
        async with tracked_db("ed_embedding_store.save_unified_observations") as db:
            # Batch insert all documents
            db.add_all(document_objects)
            await db.commit()
            logger.debug("Batch created %s unified observations", len(document_objects))

    @overload
    async def get_relevant_observations(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_distance: float = 0.3,
        level: str | None = None,
        conversation_context: str = "",
        for_reasoning: Literal[True],
    ) -> ObservationContext: ...

    @overload
    async def get_relevant_observations(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_distance: float = 0.3,
        level: str | None = None,
        conversation_context: str = "",
        for_reasoning: Literal[False],
    ) -> list[models.Document]: ...

    async def get_relevant_observations(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_distance: float = 0.3,
        level: str | None = None,
        conversation_context: str = "",
        for_reasoning: bool = False,
    ) -> list[models.Document] | ObservationContext:
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
            List of documents or ObservationContext (if for_reasoning=True)
        """
        async with tracked_db("embedding_store.get_relevant_observations") as db:
            return await self._get_observations_internal(
                db,
                query,
                top_k,
                max_distance,
                level,
                conversation_context,
                for_reasoning,
            )

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
        for_reasoning: bool,
    ) -> Any:
        """Internal method that does the actual observation retrieval."""
        try:
            if for_reasoning:
                return await self._get_observations_for_reasoning(
                    db,
                    query,
                    max_distance,
                    conversation_context,
                )
            else:
                # Regular document list return
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
            if for_reasoning:
                return ObservationContext()
            return []

    async def _get_observations_for_reasoning(
        self,
        db: AsyncSession,
        query: str,
        max_distance: float,
        conversation_context: str,
    ) -> ObservationContext:
        """Get observations formatted for reasoning with ObservationContext."""
        context = ObservationContext()

        for level_name in ["explicit", "deductive"]:
            count: int = getattr(self, f"{level_name}_observations_count", 5)
            level_enum = ReasoningLevel(level_name)

            docs = await self._query_documents_for_level(
                db,
                query,
                level_name,
                conversation_context,
                max_distance,
                count,
            )

            seen_observations: set[str] = set()
            for doc in docs:
                normalized_content: str = doc.content.strip().lower()
                if normalized_content not in seen_observations:
                    metadata = self._extract_observation_metadata(doc)
                    observation = Observation(
                        content=doc.content,
                        metadata=metadata,
                        created_at=doc.created_at,
                    )
                    context.add_observation(observation, level_enum)
                    seen_observations.add(normalized_content)

        return context

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
        combined_query = self._build_truncated_query(query, conversation_context)

        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            peer_name=self.peer_name,
            collection_name=self.collection_name,
            query=combined_query,
            max_distance=max_distance,
            top_k=count * 3,
            filters=self._build_filter_conditions(level),
        )

        # Sort by creation time and return top count
        docs_sorted: list[models.Document] = sorted(
            list(documents), key=lambda x: x.created_at, reverse=True
        )
        return docs_sorted[:count]

    def _extract_observation_metadata(self, doc: models.Document) -> Any:
        """Extract metadata from a document for ObservationMetadata."""
        metadata = ObservationMetadata()
        if doc.internal_metadata:
            metadata.session_context = doc.internal_metadata.get("session_context", "")
            metadata.summary_id = doc.internal_metadata.get("summary_id", "")
            metadata.message_id = doc.internal_metadata.get("message_id")
            metadata.level = doc.internal_metadata.get("level")
            metadata.session_name = doc.internal_metadata.get("session_name")
            metadata.premises = doc.internal_metadata.get("premises", [])
        return metadata

    async def remove_duplicates(
        self,
        facts: list[str],
        *,
        similarity_threshold: float = 0.85,
    ) -> list[str]:
        """Remove duplicate observations based on similarity threshold.

        Args:
            facts: List of observation strings
            similarity_threshold: Threshold for considering observations similar

        Returns:
            List of unique observations
        """
        if not facts:
            return []

        # Batch generate embeddings for all facts at once
        embeddings: list[list[float]] = []
        batch_size: int = 2048  # OpenAI batch limit

        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            response: CreateEmbeddingResponse = (
                await embedding_client.client.embeddings.create(
                    input=batch, model="text-embedding-3-small"
                )
            )
            embeddings.extend([data.embedding for data in response.data])

        # Now check each fact for duplicates using query_documents with pre-computed embeddings
        unique_observations: list[str] = []

        async with tracked_db("embedding_store.remove_duplicates") as db:
            for fact, embedding in zip(facts, embeddings, strict=True):
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    peer_name=self.peer_name,
                    collection_name=self.collection_name,
                    query=fact,
                    max_distance=1.0 - similarity_threshold,
                    top_k=1,
                    embedding=embedding,  # Pass pre-computed embedding
                )

                docs_list: list[models.Document] = list(documents)
                if not docs_list:
                    unique_observations.append(fact)

        logger.debug(
            "Batch remove duplicates: %s input facts, %s unique after deduplication",
            len(facts),
            len(unique_observations),
        )

        return unique_observations

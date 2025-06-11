import datetime
import logging
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from src.utils import update_document_access_metadata
from src.utils.deriver import format_datetime_simple

from ... import crud, models, schemas
from ...dependencies import tracked_db
from ...utils.history import SummaryType, get_session_summaries
from ..models import (
    Observation,
    ObservationContext,
    ObservationMetadata,
    ReasoningLevel,
)

logger = logging.getLogger(__name__)


class CollectionEmbeddingStore:
    # Default number of observations to retrieve for each reasoning level
    DEFAULT_ABDUCTIVE_OBSERVATIONS_COUNT = 1
    DEFAULT_INDUCTIVE_OBSERVATIONS_COUNT = 3
    DEFAULT_DEDUCTIVE_OBSERVATIONS_COUNT = 5
    DEFAULT_EXPLICIT_OBSERVATIONS_COUNT = 7


    def __init__(self, db: AsyncSession, app_id: str, user_id: str, collection_id: str):
        self.db = db
        self.app_id = app_id
        self.user_id = user_id
        self.collection_id = collection_id
        # Initialize observation counts with defaults
        self.explicit_observations_count = self.DEFAULT_EXPLICIT_OBSERVATIONS_COUNT
        self.abductive_observations_count = self.DEFAULT_ABDUCTIVE_OBSERVATIONS_COUNT
        self.inductive_observations_count = self.DEFAULT_INDUCTIVE_OBSERVATIONS_COUNT
        self.deductive_observations_count = self.DEFAULT_DEDUCTIVE_OBSERVATIONS_COUNT

    def set_observation_counts(
        self,
        explicit: int | None = None,
        abductive: int | None = None,
        inductive: int | None = None,
        deductive: int | None = None,
    ):
        """Set the number of observations to retrieve for each reasoning level.

        Args:
            explicit: Number of explicit observations to retrieve
            abductive: Number of abductive observations to retrieve
            inductive: Number of inductive observations to retrieve
            deductive: Number of deductive observations to retrieve
        """
        if explicit is not None:
            self.explicit_observations_count = explicit
        if abductive is not None:
            self.abductive_observations_count = abductive
        if inductive is not None:
            self.inductive_observations_count = inductive
        if deductive is not None:
            self.deductive_observations_count = deductive

    async def get_relevant_observations_for_reasoning_with_context(
        self,
        current_message: str,
        conversation_context: str = "",
        max_distance: float = 0.8,
    ) -> ObservationContext:
        """Retrieve semantically relevant observations with their session context for reasoning.

        Args:
            current_message: The current message being processed
            conversation_context: Recent conversation history for context
            max_distance: Maximum distance for semantic similarity

        Returns:
            ObservationContext with typed observations organized by reasoning level
        """
        try:
            # Create a combined query from message and context
            if conversation_context:
                combined_query = f"Current message: {current_message}\nContext: {conversation_context}"
            else:
                combined_query = current_message

            context = ObservationContext()

            # Search within each reasoning level separately
            for level_name in ["explicit", "deductive", "inductive", "abductive"]:
                count = getattr(self, f"{level_name}_observations_count")
                level = ReasoningLevel(level_name)

                # Get semantically relevant documents for this level
                documents = await crud.query_documents(
                    self.db,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    query=combined_query,
                    max_distance=max_distance,
                    top_k=count * 3,  # Get more candidates than needed
                    filter={"level": level_name},  # Filter to specific reasoning level
                )

                # Convert to list and sort by recency within semantic results
                docs_list = list(documents)
                docs_sorted = sorted(
                    docs_list, key=lambda x: x.created_at, reverse=True
                )

                # Take the most recent among semantically relevant results
                selected_docs = docs_sorted[:count]

                # Create typed observations and deduplicate
                seen_observations = set()
                for doc in selected_docs:
                    # Normalize observation content for deduplication
                    normalized_content = doc.content.strip().lower()

                    # Skip if we've seen a very similar observation
                    if normalized_content not in seen_observations:
                        # Extract metadata from doc.h_metadata
                        metadata = ObservationMetadata()
                        if doc.h_metadata:
                            metadata.session_context = doc.h_metadata.get(
                                "session_context", ""
                            )
                            metadata.summary_id = doc.h_metadata.get("summary_id", "")
                            metadata.last_accessed = doc.h_metadata.get("last_accessed")
                            metadata.access_count = doc.h_metadata.get(
                                "access_count", 0
                            )
                            metadata.accessed_sessions = doc.h_metadata.get(
                                "accessed_sessions", []
                            )
                            metadata.message_id = doc.h_metadata.get("message_id")
                            metadata.level = doc.h_metadata.get("level")
                            metadata.session_id = doc.h_metadata.get("session_id")
                            metadata.premises = doc.h_metadata.get("premises", [])

                        observation = Observation(
                            content=doc.content,
                            metadata=metadata,
                            created_at=doc.created_at,
                        )
                        context.add_observation(observation, level)
                        seen_observations.add(normalized_content)
                    else:
                        logger.debug(
                            f"Skipping duplicate observation during retrieval: {doc.content[:50]}..."
                        )

            return context

        except Exception as e:
            logger.error(
                f"Error retrieving relevant observations with context for reasoning: {e}"
            )
            # Fallback to simple observations retrieval
            logger.warning("Falling back to simple observations retrieval")
            simple_context = await self.get_relevant_observations_for_reasoning(
                current_message, conversation_context, max_distance
            )

            # Convert simple context to ObservationContext
            context = ObservationContext()
            for level_name, observations in simple_context:
                if level_name in {"abductive", "inductive", "deductive"}:
                    level = ReasoningLevel(level_name)
                    for obs_content in observations:
                        observation = Observation(
                            content=obs_content,
                            created_at=datetime.datetime.now()
                        )
                        context.add_observation(observation, level)

            return context

    async def get_relevant_observations_for_reasoning(
        self,
        current_message: str,
        conversation_context: str = "",
        max_distance: float = 0.8,
    ) -> ObservationContext:
        """Retrieve semantically relevant observations for reasoning about the current message.

        Args:
            current_message: The current message being processed
            conversation_context: Recent conversation history for context
            max_distance: Maximum distance for semantic similarity

        Returns:
            ObservationContext with observations organized by reasoning level
        """
        try:
            # Create a combined query from message and context
            if conversation_context:
                combined_query = f"Current message: {current_message}\nContext: {conversation_context}"
            else:
                combined_query = current_message

            context = ObservationContext()

            # Search within each reasoning level separately
            for level_name in ["explicit", "deductive", "inductive", "abductive"]:
                count = getattr(self, f"{level_name}_observations_count")
                level = ReasoningLevel(level_name)

                # Get semantically relevant documents for this level
                documents = await crud.query_documents(
                    self.db,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    query=combined_query,
                    max_distance=max_distance,
                    top_k=count * 3,  # Get more candidates than needed
                    filter={"level": level_name},  # Filter to specific reasoning level
                )

                # Convert to list and sort by recency within semantic results
                docs_list = list(documents)
                docs_sorted = sorted(
                    docs_list, key=lambda x: x.created_at, reverse=True
                )

                # Take the most recent among semantically relevant results and deduplicate
                selected_docs = docs_sorted[
                    : count * 2
                ]  # Get more candidates for deduplication

                # Deduplicate observations for this level
                seen_observations = set()
                added_count = 0

                for doc in selected_docs:
                    if (
                        added_count >= count
                    ):  # Stop when we have enough unique observations
                        break

                    normalized_content = doc.content.strip().lower()
                    if normalized_content not in seen_observations:
                        observation = Observation(
                            content=doc.content, created_at=doc.created_at
                        )
                        context.add_observation(observation, level)
                        seen_observations.add(normalized_content)
                        added_count += 1
                    else:
                        logger.debug(
                            f"Skipping duplicate observation during retrieval: {doc.content[:50]}..."
                        )

            return context

        except Exception as e:
            logger.error(f"Error retrieving relevant observations for reasoning: {e}")
            # Fallback to recent observations if semantic search fails
            logger.warning("Falling back to recent observations retrieval")
            return await self.get_most_recent_observations()

    async def execute_queries(self, queries: list[str]) -> "QueryExecutionResult":
        """Execute multiple semantic queries and return strongly-typed results.
        
        Args:
            queries: List of query strings to search for
            
        Returns:
            QueryExecutionResult with results organized by query and level
        """
        from src.deriver.models import QueryExecutionResult, QueryResult, QueryResultsByLevel
        
        try:
            # Execute queries concurrently
            query_results_list = []
            for query in queries:
                try:
                    # Get observations for this query
                    context = await self.get_relevant_observations_for_reasoning(
                        current_message=query
                    )
                    
                    # Extract observation content strings
                    observations = []
                    for level_name in ["explicit", "deductive", "inductive", "abductive"]:
                        level_obs = getattr(context, level_name, [])
                        for obs in level_obs:
                            observations.append(obs.content)
                    
                    query_results_list.append(QueryResult(
                        query=query,
                        observations=observations
                    ))
                except Exception as e:
                    logger.error(f"Error executing query '{query}': {e}")
                    query_results_list.append(QueryResult(
                        query=query,
                        observations=[]
                    ))
            
            # Organize results by level
            results_by_level = QueryResultsByLevel()
            total_observations = 0
            
            for query_result in query_results_list:
                total_observations += len(query_result.observations)
                
                # For now, we'll add all observations to abductive level
                # This could be enhanced to categorize by level in the future
                results_by_level.abductive.extend(query_result.observations)
            
            return QueryExecutionResult(
                queries=queries,
                results_by_query=query_results_list,
                results_by_level=results_by_level,
                total_observations=total_observations
            )
            
        except Exception as e:
            logger.error(f"Error executing queries: {e}")
            return QueryExecutionResult(
                queries=queries,
                results_by_query=[],
                results_by_level=QueryResultsByLevel(),
                total_observations=0
            )

    async def get_most_recent_observations(self) -> ObservationContext:
        """Retrieve the most recent observations for each reasoning level.

        Returns:
            ObservationContext with the most recent observations for each reasoning level.
        """
        try:
            context = ObservationContext()

            # Process each reasoning level
            for level_name in ["explicit", "deductive", "inductive", "abductive"]:
                count = getattr(self, f"{level_name}_observations_count")
                level = ReasoningLevel(level_name)

                # Get most recent observations for this level
                stmt = await crud.get_documents(
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    filter={"level": level_name},
                    reverse=True,  # Sort by created_at in descending order
                )
                docs = (await self.db.execute(stmt)).scalars().all()[:count]

                # Convert to Observation objects
                for doc in docs:
                    observation = Observation(
                        content=doc.content, created_at=doc.created_at
                    )
                    context.add_observation(observation, level)

            logger.debug(
                f"Retrieved observations - Explicit: {len(context.explicit)}, "
                f"Abductive: {len(context.abductive)}, "
                f"Inductive: {len(context.inductive)}, "
                f"Deductive: {len(context.deductive)}"
            )

            return context

        except Exception as e:
            logger.error(f"Error retrieving observations from collection: {e}")
            # Return empty context if retrieval fails
            return ObservationContext()

    # Backward compatibility alias
    async def get_most_recent_facts(self) -> ObservationContext:
        """Backward compatibility alias for get_most_recent_observations."""
        return await self.get_most_recent_observations()

    async def get_contextualized_observations_for_dialectic(
        self,
    ) -> ObservationContext:
        """Retrieve observations with their session context for dialectic API calls.

        Returns:
            ObservationContext with observations organized by reasoning level.
            This method creates simple observations without the complex formatting.
        """
        try:
            context = ObservationContext()

            for level_name in ["explicit", "deductive", "inductive", "abductive"]:
                count = getattr(self, f"{level_name}_observations_count")
                level = ReasoningLevel(level_name)

                stmt = await crud.get_documents(
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    filter={"level": level_name},
                    reverse=True,
                )
                docs = (await self.db.execute(stmt)).scalars().all()[:count]

                # Convert to Observation objects with full metadata
                for doc in docs:
                    metadata = ObservationMetadata()
                    if doc.h_metadata:
                        metadata.session_context = doc.h_metadata.get(
                            "session_context", ""
                        )
                        metadata.summary_id = doc.h_metadata.get("summary_id", "")
                        metadata.last_accessed = doc.h_metadata.get("last_accessed")
                        metadata.access_count = doc.h_metadata.get("access_count", 0)
                        metadata.accessed_sessions = doc.h_metadata.get(
                            "accessed_sessions", []
                        )
                        metadata.message_id = doc.h_metadata.get("message_id")
                        metadata.level = doc.h_metadata.get("level")
                        metadata.session_id = doc.h_metadata.get("session_id")
                        metadata.premises = doc.h_metadata.get("premises", [])

                    observation = Observation(
                        content=doc.content,
                        metadata=metadata,
                        created_at=doc.created_at,
                    )
                    context.add_observation(observation, level)

            logger.debug(
                f"Retrieved contextualized observations for dialectic - "
                f"Explicit: {len(context.explicit)}, "
                f"Abductive: {len(context.abductive)}, "
                f"Inductive: {len(context.inductive)}, "
                f"Deductive: {len(context.deductive)}"
            )

            return context

        except Exception as e:
            logger.error(f"Error retrieving contextualized observations: {e}")
            # Return empty context if retrieval fails
            return ObservationContext()

    # Backward compatibility alias
    async def get_contextualized_facts_for_dialectic(self) -> ObservationContext:
        """Backward compatibility alias for get_contextualized_observations_for_dialectic."""
        return await self.get_contextualized_observations_for_dialectic()

    async def save_observations(
        self,
        db: AsyncSession,
        observations: list[str],
        similarity_threshold: float = 0.85,
        message_id: str | None = None,
        level: str | None = None,
        session_id: str | None = None,
        premises: list[str] | None = None,
    ) -> None:
        """Save observations to the collection with summary linking.

        Args:
            db: Database session to use for saving
            observations: List of observations to save
            similarity_threshold: Observations with similarity above this threshold are considered duplicates
            message_id: Message ID to associate with the observations
            level: Reasoning level (abductive, inductive, deductive)
            session_id: Session ID to link with existing summary context
            premises: List of premises that support this observation (stored in metadata)
        """
        # Get latest short summary for context linking if session_id provided
        summary_id = None
        summary_content = None
        if session_id:
            try:
                latest_summary = cast(
                    models.Metamessage | None,
                    await get_session_summaries(
                        db, session_id, SummaryType.SHORT, only_latest=True
                    )
                )
                if latest_summary:  # latest_summary is a single Metamessage or None
                    summary_id = latest_summary.public_id
                    summary_content = latest_summary.content
            except Exception as e:
                logger.warning(
                    f"Could not retrieve latest summary for session {session_id}: {e}"
                )

        # Save observations directly (we're already in background processing context)
        saved_count = 0
        
        for observation in observations:
            try:
                # Check if similar observation already exists
                existing_similar = await self._find_similar_observation(
                    observation, similarity_threshold, level
                )

                if existing_similar:
                    # Update existing observation metadata instead of creating duplicate
                    await self._update_similar_observation_metadata(
                        existing_similar, session_id, message_id
                    )
                    logger.debug(
                        f"Updated existing similar observation: {observation[:50]}..."
                    )
                    continue

                # No similar observation found, create new one
                metadata = {}
                if message_id:
                    metadata["message_id"] = message_id
                if level:
                    metadata["level"] = level
                if session_id:
                    metadata["session_id"] = session_id
                if summary_id:
                    metadata["summary_id"] = summary_id
                if summary_content:
                    # Store truncated summary content for context
                    metadata["session_context"] = summary_content[:500]

                # Add premises metadata if provided
                if premises:
                    metadata["premises"] = premises

                # Save the observation directly
                document = schemas.DocumentCreate(content=observation, metadata=metadata)
                await crud.create_document(
                    db,
                    document=document,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    duplicate_threshold=1 - similarity_threshold,
                )
                
                saved_count += 1
                logger.debug(f"Saved observation: {observation[:50]}...")

            except Exception as e:
                logger.error(f"Error saving observation '{observation[:50]}...': {e}")
                continue

        logger.info(
            f"Saved {saved_count} observations directly (level: {level})"
        )

    def _format_temporal_metadata_for_dialectic(self, doc: models.Document) -> str:
        """
        Format temporal metadata for dialectic observation display.
        Shows both observation genesis and ongoing relevance patterns.

        Args:
            doc: Document with temporal metadata

        Returns:
            Formatted temporal context string
        """
        temporal_parts = []

        # Observation inception - when originally derived
        if doc.created_at:
            formatted_time = format_datetime_simple(doc.created_at)
            temporal_parts.append(f"derived {formatted_time}")

        # Access frequency and session spread
        access_count = doc.h_metadata.get("access_count", 0) if doc.h_metadata else 0
        accessed_sessions = (
            doc.h_metadata.get("accessed_sessions", []) if doc.h_metadata else []
        )

        if access_count > 0:
            if accessed_sessions:
                session_count = len(accessed_sessions)
                if session_count > 1:
                    temporal_parts.append(
                        f"accessed {access_count}x across {session_count} sessions"
                    )
                else:
                    temporal_parts.append(f"accessed {access_count}x in 1 session")
            else:
                temporal_parts.append(f"accessed {access_count}x")

        # Last accessed timestamp
        last_accessed = doc.h_metadata.get("last_accessed") if doc.h_metadata else None
        if last_accessed:
            formatted_last_accessed = format_datetime_simple(last_accessed)
            temporal_parts.append(f"last accessed {formatted_last_accessed}")

        if temporal_parts:
            return f" [{', '.join(temporal_parts)}]"
        else:
            return ""

    async def _find_similar_observation(
        self, observation: str, similarity_threshold: float, level: str | None = None
    ) -> models.Document | None:
        """Find an existing observation that is semantically similar to the given observation."""
        try:
            filter_dict = {}
            if level:
                filter_dict["level"] = level

            # Query for similar observations using semantic search
            documents = await crud.query_documents(
                self.db,
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                query=observation,
                max_distance=1 - similarity_threshold,  # Convert similarity to distance
                top_k=1,
                filter=filter_dict,
            )

            # Return the most similar document if one exists
            docs_list = list(documents)
            return docs_list[0] if docs_list else None

        except Exception as e:
            logger.warning(f"Error finding similar observation: {e}")
            return None

    async def _update_similar_observation_metadata(
        self, doc: models.Document, session_id: str | None, message_id: str | None
    ):
        """Update metadata for an existing similar observation."""
        await update_document_access_metadata(
            self.db,
            doc,
            self.app_id,
            self.user_id,
            self.collection_id,
            session_id=session_id,
            message_id=message_id,
        )

    async def get_relevant_observations(
        self, query: str, top_k: int = 5, max_distance: float = 0.3
    ) -> list[models.Document]:
        """Retrieve the most relevant observations for a given query.

        Args:
            query: The query text to find relevant observations for
            top_k: Maximum number of observations to return
            max_distance: Maximum distance for semantic similarity

        Returns:
            List of document objects sorted by relevance
        """
        documents = await crud.query_documents(
            self.db,
            app_id=self.app_id,
            user_id=self.user_id,
            collection_id=self.collection_id,
            query=query,
            max_distance=max_distance,
            top_k=top_k,
        )

        return list(documents)

    async def remove_duplicates(
        self, observations: list[str], similarity_threshold: float = 0.85
    ) -> list[str]:
        """Remove observations that are duplicates of existing observations in the vector store.

        Args:
            observations: List of observations to check for duplicates
            similarity_threshold: Observations with similarity above this threshold are considered duplicates

        Returns:
            List of observations that are not duplicates of existing observations
        """
        unique_observations = []

        for observation in observations:
            try:
                # Check for duplicates using the crud function with existing db session
                duplicates = await crud.get_duplicate_documents(
                    self.db,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    content=observation,
                    similarity_threshold=similarity_threshold,
                )

                if not duplicates:
                    # No duplicates found, add to unique observations
                    unique_observations.append(observation)
                else:
                    # Log duplicate found
                    logger.debug(
                        f"Duplicate found: {duplicates[0].content}. Ignoring observation: {observation}"
                    )
            except Exception as e:
                logger.error(f"Error checking for duplicates: {e}")
                # If there's an error, still include the observation to avoid losing information
                unique_observations.append(observation)

        return unique_observations

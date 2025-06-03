import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from ... import crud, models, schemas
from ...dependencies import tracked_db
from ...utils.history import SummaryType, get_session_summaries
from ..fact_saver import get_fact_saver_queue
from src.utils.deriver import format_datetime_simple

logger = logging.getLogger(__name__)


class CollectionEmbeddingStore:
    # Default number of observations to retrieve for each reasoning level
    DEFAULT_ABDUCTIVE_OBSERVATIONS_COUNT = 1
    DEFAULT_INDUCTIVE_OBSERVATIONS_COUNT = 3
    DEFAULT_DEDUCTIVE_OBSERVATIONS_COUNT = 5

    def __init__(self, db: AsyncSession, app_id: str, user_id: str, collection_id: str):
        self.db = db
        self.app_id = app_id
        self.user_id = user_id
        self.collection_id = collection_id
        # Initialize observation counts with defaults
        self.abductive_observations_count = self.DEFAULT_ABDUCTIVE_OBSERVATIONS_COUNT
        self.inductive_observations_count = self.DEFAULT_INDUCTIVE_OBSERVATIONS_COUNT
        self.deductive_observations_count = self.DEFAULT_DEDUCTIVE_OBSERVATIONS_COUNT

    def set_observation_counts(self, abductive: int | None = None, inductive: int | None = None, deductive: int | None = None):
        """Set the number of observations to retrieve for each reasoning level.
        
        Args:
            abductive: Number of abductive observations to retrieve
            inductive: Number of inductive observations to retrieve
            deductive: Number of deductive observations to retrieve
        """
        if abductive is not None:
            self.abductive_observations_count = abductive
        if inductive is not None:
            self.inductive_observations_count = inductive
        if deductive is not None:
            self.deductive_observations_count = deductive

    # Backward compatibility alias
    def set_fact_counts(self, abductive: int | None = None, inductive: int | None = None, deductive: int | None = None):
        """Backward compatibility alias for set_observation_counts."""
        return self.set_observation_counts(abductive, inductive, deductive)

    async def get_relevant_observations_for_reasoning_with_context(
        self, 
        current_message: str, 
        conversation_context: str = "",
        max_distance: float = 0.8
    ) -> dict[str, list[dict]]:
        """Retrieve semantically relevant observations with their session context for reasoning.
        
        Args:
            current_message: The current message being processed
            conversation_context: Recent conversation history for context
            max_distance: Maximum distance for semantic similarity
            
        Returns:
            Dictionary with reasoning levels as keys and lists of observation dicts as values.
            Each observation dict contains: {'content': str, 'session_context': str, 'created_at': datetime}
        """
        try:
            # Create a combined query from message and context
            if conversation_context:
                combined_query = f"Current message: {current_message}\nContext: {conversation_context}"
            else:
                combined_query = current_message
                
            context = {}
            
            # Search within each reasoning level separately
            for level in ["abductive", "inductive", "deductive"]:
                count = getattr(self, f"{level}_observations_count")
                
                # Get semantically relevant documents for this level
                documents = await crud.query_documents(
                    self.db,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    query=combined_query,
                    max_distance=max_distance,
                    top_k=count * 3,  # Get more candidates than needed
                    filter={"level": level}  # Filter to specific reasoning level
                )
                
                # Convert to list and sort by recency within semantic results
                docs_list = list(documents)
                docs_sorted = sorted(docs_list, key=lambda x: x.created_at, reverse=True)
                
                # Take the most recent among semantically relevant results
                selected_docs = docs_sorted[:count]
                
                # Format with session context and deduplicate
                context[level] = []
                seen_observations = set()
                
                for doc in selected_docs:
                    # Normalize observation content for deduplication
                    normalized_content = doc.content.strip().lower()
                    
                    # Skip if we've seen a very similar observation
                    if normalized_content not in seen_observations:
                        observation_info = {
                            'content': doc.content,
                            'session_context': doc.h_metadata.get("session_context", ""),
                            'summary_id': doc.h_metadata.get("summary_id", ""),
                            'created_at': doc.created_at,
                            'last_accessed': doc.h_metadata.get("last_accessed"),
                            'access_count': doc.h_metadata.get("access_count", 0),
                            'accessed_sessions': doc.h_metadata.get("accessed_sessions", [])
                        }
                        context[level].append(observation_info)
                        seen_observations.add(normalized_content)
                    else:
                        logger.debug(f"Skipping duplicate observation during retrieval: {doc.content[:50]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant observations with context for reasoning: {e}")
            # Fallback to simple observations retrieval
            logger.warning("Falling back to simple observations retrieval")
            simple_context = await self.get_relevant_observations_for_reasoning(current_message, conversation_context, max_distance)
            # Convert to context format
            enhanced_context = {}
            for level, observations in simple_context.items():
                enhanced_context[level] = [{'content': observation, 'session_context': '', 'summary_id': '', 'created_at': None} for observation in observations]
            return enhanced_context

    # Backward compatibility alias
    async def get_relevant_facts_for_reasoning_with_context(self, current_message: str, conversation_context: str = "", max_distance: float = 0.8) -> dict[str, list[dict]]:
        """Backward compatibility alias for get_relevant_observations_for_reasoning_with_context."""
        return await self.get_relevant_observations_for_reasoning_with_context(current_message, conversation_context, max_distance)

    async def get_relevant_observations_for_reasoning(
        self, 
        current_message: str, 
        conversation_context: str = "",
        max_distance: float = 0.8
    ) -> dict[str, list[str]]:
        """Retrieve semantically relevant observations for reasoning about the current message.
        
        Args:
            current_message: The current message being processed
            conversation_context: Recent conversation history for context
            max_distance: Maximum distance for semantic similarity
            
        Returns:
            Dictionary with reasoning levels as keys and lists of relevant observations as values.
        """
        try:
            # Create a combined query from message and context
            if conversation_context:
                combined_query = f"Current message: {current_message}\nContext: {conversation_context}"
            else:
                combined_query = current_message
                
            context = {}
            
            # Search within each reasoning level separately
            for level in ["abductive", "inductive", "deductive"]:
                count = getattr(self, f"{level}_observations_count")
                
                # Get semantically relevant documents for this level
                documents = await crud.query_documents(
                    self.db,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    query=combined_query,
                    max_distance=max_distance,
                    top_k=count * 3,  # Get more candidates than needed
                    filter={"level": level}  # Filter to specific reasoning level
                )
                
                # Convert to list and sort by recency within semantic results
                docs_list = list(documents)
                docs_sorted = sorted(docs_list, key=lambda x: x.created_at, reverse=True)
                
                # Take the most recent among semantically relevant results and deduplicate
                selected_docs = docs_sorted[:count * 2]  # Get more candidates for deduplication
                
                # Deduplicate observations for this level
                unique_observations = []
                seen_observations = set()
                
                for doc in selected_docs:
                    if len(unique_observations) >= count:  # Stop when we have enough unique observations
                        break
                        
                    normalized_content = doc.content.strip().lower()
                    if normalized_content not in seen_observations:
                        unique_observations.append(doc.content)
                        seen_observations.add(normalized_content)
                    else:
                        logger.debug(f"Skipping duplicate observation during retrieval: {doc.content[:50]}...")
                
                context[level] = unique_observations
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant observations for reasoning: {e}")
            # Fallback to recent observations if semantic search fails
            logger.warning("Falling back to recent observations retrieval")
            return await self.get_most_recent_observations()

    # Backward compatibility alias
    async def get_relevant_facts_for_reasoning(self, current_message: str, conversation_context: str = "", max_distance: float = 0.8) -> dict[str, list[str]]:
        """Backward compatibility alias for get_relevant_observations_for_reasoning."""
        return await self.get_relevant_observations_for_reasoning(current_message, conversation_context, max_distance)

    async def get_most_recent_observations(self) -> dict[str, list[str]]:
        """Retrieve the most recent observations for each reasoning level.
        
        Returns:
            Dictionary with reasoning levels as keys and lists of observations as values.
            Each list contains the most recent observations for that reasoning level.
        """
        try:
            # Get most recent abductive observations
            abductive_stmt = await crud.get_documents(
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                filter={"level": "abductive"},
                reverse=True  # Sort by created_at in descending order
            )
            abductive_docs = (await self.db.execute(abductive_stmt)).scalars().all()[:self.abductive_observations_count]
            
            # Get most recent inductive observations
            inductive_stmt = await crud.get_documents(
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                filter={"level": "inductive"},
                reverse=True
            )
            inductive_docs = (await self.db.execute(inductive_stmt)).scalars().all()[:self.inductive_observations_count]
            
            # Get most recent deductive observations
            deductive_stmt = await crud.get_documents(
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                filter={"level": "deductive"},
                reverse=True
            )
            deductive_docs = (await self.db.execute(deductive_stmt)).scalars().all()[:self.deductive_observations_count]
            
            # Initialize context with retrieved observations
            context = {
                "abductive": [doc.content for doc in abductive_docs],
                "inductive": [doc.content for doc in inductive_docs],
                "deductive": [doc.content for doc in deductive_docs]
            }
            
            logger.debug(
                f"Retrieved observations - Abductive: {len(abductive_docs)}, "
                f"Inductive: {len(inductive_docs)}, "
                f"Deductive: {len(deductive_docs)}"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving observations from collection: {e}")
            # Return empty context if retrieval fails
            return {
                "abductive": [],
                "inductive": [],
                "deductive": []
            }

    # Backward compatibility alias
    async def get_most_recent_facts(self) -> dict[str, list[str]]:
        """Backward compatibility alias for get_most_recent_observations."""
        return await self.get_most_recent_observations()

    async def get_contextualized_observations_for_dialectic(self) -> dict[str, list[str]]:
        """Retrieve observations with their session context for dialectic API calls.
        
        Returns:
            Dictionary with reasoning levels as keys and lists of contextualized observations.
            Observations are grouped by session summary to avoid redundancy and provide context.
        """
        try:
            levels = ["abductive", "inductive", "deductive"]
            context = {}
            
            for level in levels:
                count = getattr(self, f"{level}_observations_count")
                stmt = await crud.get_documents(
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    filter={"level": level},
                    reverse=True
                )
                docs = (await self.db.execute(stmt)).scalars().all()[:count]
                
                # Group observations by session summary to avoid redundancy
                summary_groups = {}
                docs_without_summary = []
                
                # Note: We skip metadata updates during retrieval to avoid concurrent session operations
                # Access tracking will be handled through other paths when observations are saved/updated
                for doc in docs:
                    
                    session_context = doc.h_metadata.get("session_context", "")
                    summary_id = doc.h_metadata.get("summary_id", "")
                    
                    if session_context and summary_id:
                        if summary_id not in summary_groups:
                            summary_groups[summary_id] = {
                                "context": session_context,
                                "observations": []
                            }
                        summary_groups[summary_id]["observations"].append(doc.content)
                    else:
                        docs_without_summary.append(doc)
                
                # Format the output to avoid redundancy
                formatted_observations = []
                
                # Add observations with session context (grouped to avoid repeating context)
                for _summary_id, group in summary_groups.items():
                    context_header = f"From session: {group['context'][:200]}..."
                    observations_list = "\n".join([f"  • {observation_content}" for observation_content in group["observations"]])
                    formatted_observations.append(f"{context_header}\n{observations_list}")
                
                # Add observations without session context - include temporal metadata
                for doc in docs_without_summary:
                    temporal_info = self._format_temporal_metadata_for_dialectic(doc)
                    formatted_observations.append(f"• {doc.content}{temporal_info}")
                
                context[level] = formatted_observations
            
            logger.debug(
                f"Retrieved contextualized observations for dialectic - "
                f"Abductive: {len(context['abductive'])}, "
                f"Inductive: {len(context['inductive'])}, "
                f"Deductive: {len(context['deductive'])}"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving contextualized observations: {e}")
            # Return empty context if retrieval fails
            return {
                "abductive": [],
                "inductive": [],
                "deductive": []
            }

    # Backward compatibility alias
    async def get_contextualized_facts_for_dialectic(self) -> dict[str, list[str]]:
        """Backward compatibility alias for get_contextualized_observations_for_dialectic."""
        return await self.get_contextualized_observations_for_dialectic()

    async def save_observations(
        self,
        observations: list[str],
        similarity_threshold: float = 0.85,
        message_id: str | None = None,
        level: str | None = None,
        session_id: str | None = None,
        premises: list[str] | None = None
    ) -> None:
        """Save observations to the collection with summary linking.

        Args:
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
                latest_summaries = await get_session_summaries(
                    self.db, session_id, SummaryType.SHORT, only_latest=True
                )
                if latest_summaries and len(latest_summaries) > 0:
                    latest_summary = latest_summaries[0]  # Get the first (latest) summary
                    summary_id = latest_summary.public_id
                    summary_content = latest_summary.content
            except Exception as e:
                logger.warning(f"Could not retrieve latest summary for session {session_id}: {e}")
        
        # Queue observations for asynchronous saving to avoid transaction conflicts
        fact_saver = get_fact_saver_queue()
        
        for observation in observations:
            try:
                # Check if similar observation already exists
                existing_similar = await self._find_similar_observation(observation, similarity_threshold, level)
                
                if existing_similar:
                    # Update existing observation metadata instead of creating duplicate
                    await self._update_similar_observation_metadata(existing_similar, session_id, message_id)
                    logger.debug(f"Updated existing similar observation: {observation[:50]}...")
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
                
                # Generate unique task ID for tracking
                task_id = f"{level}_{message_id}_{uuid.uuid4().hex[:8]}"
                
                # Queue the observation for saving
                await fact_saver.queue_fact(
                    content=observation,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    metadata=metadata,
                    duplicate_threshold=1 - similarity_threshold,
                    task_id=task_id
                )
                logger.debug(f"Queued observation for saving: {observation[:50]}...")
                
            except Exception as e:
                logger.error(f"Error queuing observation '{observation[:50]}...': {e}")
                continue
        
        logger.info(f"Queued {len(observations)} observations for saving (level: {level})")

    # Backward compatibility alias
    async def save_facts(self, facts: list[str], similarity_threshold: float = 0.85, message_id: str | None = None, level: str | None = None, session_id: str | None = None, premises: list[str] | None = None) -> None:
        """Backward compatibility alias for save_observations."""
        return await self.save_observations(facts, similarity_threshold, message_id, level, session_id, premises)

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
        access_count = doc.h_metadata.get('access_count', 0) if doc.h_metadata else 0
        accessed_sessions = doc.h_metadata.get('accessed_sessions', []) if doc.h_metadata else []
        
        if access_count > 0:
            if accessed_sessions:
                session_count = len(accessed_sessions)
                if session_count > 1:
                    temporal_parts.append(f"accessed {access_count}x across {session_count} sessions")
                else:
                    temporal_parts.append(f"accessed {access_count}x in 1 session")
            else:
                temporal_parts.append(f"accessed {access_count}x")
        
        # Last accessed timestamp
        last_accessed = doc.h_metadata.get('last_accessed') if doc.h_metadata else None
        if last_accessed:
            formatted_last_accessed = format_datetime_simple(last_accessed)
            temporal_parts.append(f"last accessed {formatted_last_accessed}")
        
        if temporal_parts:
            return f" [{', '.join(temporal_parts)}]"
        else:
            return ""

    async def _find_similar_observation(self, observation: str, similarity_threshold: float, level: str | None = None) -> models.Document | None:
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
                filter=filter_dict
            )
            
            # Return the most similar document if one exists
            docs_list = list(documents)
            return docs_list[0] if docs_list else None
            
        except Exception as e:
            logger.warning(f"Error finding similar observation: {e}")
            return None

    # Backward compatibility alias
    async def _find_similar_fact(self, fact: str, similarity_threshold: float, level: str | None = None) -> models.Document | None:
        """Backward compatibility alias for _find_similar_observation."""
        return await self._find_similar_observation(fact, similarity_threshold, level)

    async def _update_similar_observation_metadata(self, doc: models.Document, session_id: str | None, message_id: str | None):
        """Update metadata for an existing similar observation."""
        try:
            # Update the document's metadata with new access information
            updated_metadata = doc.h_metadata.copy() if doc.h_metadata else {}
            
            # Increment access count
            updated_metadata["access_count"] = updated_metadata.get("access_count", 0) + 1
            updated_metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
            
            # Track accessed sessions
            accessed_sessions = updated_metadata.get("accessed_sessions", [])
            if session_id and session_id not in accessed_sessions:
                accessed_sessions.append(session_id)
                updated_metadata["accessed_sessions"] = accessed_sessions
            
            # Update the document
            document_update = schemas.DocumentUpdate(metadata=updated_metadata)
            await crud.update_document(
                self.db,
                document=document_update,
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                document_id=doc.public_id
            )
            
        except Exception as e:
            logger.warning(f"Error updating similar observation metadata: {e}")

    # Backward compatibility alias
    async def _update_similar_fact_metadata(self, doc: models.Document, session_id: str | None, message_id: str | None):
        """Backward compatibility alias for _update_similar_observation_metadata."""
        return await self._update_similar_observation_metadata(doc, session_id, message_id)

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

    # Backward compatibility alias
    async def get_relevant_facts(self, query: str, top_k: int = 5, max_distance: float = 0.3) -> list[models.Document]:
        """Backward compatibility alias for get_relevant_observations."""
        return await self.get_relevant_observations(query, top_k, max_distance)

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

        async with tracked_db("embedding_store.remove_duplicates") as db:
            for observation in observations:
                try:
                    # Check for duplicates using the crud function
                    duplicates = await crud.get_duplicate_documents(
                        db,
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

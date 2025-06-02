import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from ... import crud, models, schemas
from ...dependencies import tracked_db
from ...utils.history import SummaryType, get_session_summaries
from ..fact_saver import get_fact_saver_queue

logger = logging.getLogger(__name__)


class CollectionEmbeddingStore:
    # Default number of facts to retrieve for each reasoning level
    DEFAULT_ABDUCTIVE_FACTS_COUNT = 1
    DEFAULT_INDUCTIVE_FACTS_COUNT = 3
    DEFAULT_DEDUCTIVE_FACTS_COUNT = 5

    def __init__(self, db: AsyncSession, app_id: str, user_id: str, collection_id: str):
        self.db = db
        self.app_id = app_id
        self.user_id = user_id
        self.collection_id = collection_id
        # Initialize fact counts with defaults
        self.abductive_facts_count = self.DEFAULT_ABDUCTIVE_FACTS_COUNT
        self.inductive_facts_count = self.DEFAULT_INDUCTIVE_FACTS_COUNT
        self.deductive_facts_count = self.DEFAULT_DEDUCTIVE_FACTS_COUNT

    def set_fact_counts(self, abductive: int | None = None, inductive: int | None = None, deductive: int | None = None):
        """Set the number of facts to retrieve for each reasoning level.
        
        Args:
            abductive: Number of abductive facts to retrieve
            inductive: Number of inductive facts to retrieve
            deductive: Number of deductive facts to retrieve
        """
        if abductive is not None:
            self.abductive_facts_count = abductive
        if inductive is not None:
            self.inductive_facts_count = inductive
        if deductive is not None:
            self.deductive_facts_count = deductive

    async def get_relevant_facts_for_reasoning_with_context(
        self, 
        current_message: str, 
        conversation_context: str = "",
        max_distance: float = 0.8
    ) -> dict[str, list[dict]]:
        """Retrieve semantically relevant facts with their session context for reasoning.
        
        Args:
            current_message: The current message being processed
            conversation_context: Recent conversation history for context
            max_distance: Maximum distance for semantic similarity
            
        Returns:
            Dictionary with reasoning levels as keys and lists of fact dicts as values.
            Each fact dict contains: {'content': str, 'session_context': str, 'created_at': datetime}
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
                count = getattr(self, f"{level}_facts_count")
                
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
                seen_facts = set()
                
                for doc in selected_docs:
                    # Normalize fact content for deduplication
                    normalized_content = doc.content.strip().lower()
                    
                    # Skip if we've seen a very similar fact
                    if normalized_content not in seen_facts:
                        fact_info = {
                            'content': doc.content,
                            'session_context': doc.h_metadata.get("session_context", ""),
                            'summary_id': doc.h_metadata.get("summary_id", ""),
                            'created_at': doc.created_at,
                            'last_accessed': doc.h_metadata.get("last_accessed"),
                            'access_count': doc.h_metadata.get("access_count", 0),
                            'accessed_sessions': doc.h_metadata.get("accessed_sessions", [])
                        }
                        context[level].append(fact_info)
                        seen_facts.add(normalized_content)
                    else:
                        logger.debug(f"Skipping duplicate fact during retrieval: {doc.content[:50]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant facts with context for reasoning: {e}")
            # Fallback to simple facts retrieval
            logger.warning("Falling back to simple facts retrieval")
            simple_context = await self.get_relevant_facts_for_reasoning(current_message, conversation_context, max_distance)
            # Convert to context format
            enhanced_context = {}
            for level, facts in simple_context.items():
                enhanced_context[level] = [{'content': fact, 'session_context': '', 'summary_id': '', 'created_at': None} for fact in facts]
            return enhanced_context

    async def get_relevant_facts_for_reasoning(
        self, 
        current_message: str, 
        conversation_context: str = "",
        max_distance: float = 0.8
    ) -> dict[str, list[str]]:
        """Retrieve semantically relevant facts for reasoning about the current message.
        
        Args:
            current_message: The current message being processed
            conversation_context: Recent conversation history for context
            max_distance: Maximum distance for semantic similarity
            
        Returns:
            Dictionary with reasoning levels as keys and lists of relevant facts as values.
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
                count = getattr(self, f"{level}_facts_count")
                
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
                
                # Deduplicate facts for this level
                unique_facts = []
                seen_facts = set()
                
                for doc in selected_docs:
                    if len(unique_facts) >= count:  # Stop when we have enough unique facts
                        break
                        
                    normalized_content = doc.content.strip().lower()
                    if normalized_content not in seen_facts:
                        unique_facts.append(doc.content)
                        seen_facts.add(normalized_content)
                    else:
                        logger.debug(f"Skipping duplicate fact during retrieval: {doc.content[:50]}...")
                
                context[level] = unique_facts
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant facts for reasoning: {e}")
            # Fallback to recent facts if semantic search fails
            logger.warning("Falling back to recent facts retrieval")
            return await self.get_most_recent_facts()

    async def get_most_recent_facts(self) -> dict[str, list[str]]:
        """Retrieve the most recent facts for each reasoning level.
        
        Returns:
            Dictionary with reasoning levels as keys and lists of facts as values.
            Each list contains the most recent facts for that reasoning level.
        """
        try:
            # Get most recent abductive facts
            abductive_stmt = await crud.get_documents(
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                filter={"level": "abductive"},
                reverse=True  # Sort by created_at in descending order
            )
            abductive_docs = (await self.db.execute(abductive_stmt)).scalars().all()[:self.abductive_facts_count]
            
            # Get most recent inductive facts
            inductive_stmt = await crud.get_documents(
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                filter={"level": "inductive"},
                reverse=True
            )
            inductive_docs = (await self.db.execute(inductive_stmt)).scalars().all()[:self.inductive_facts_count]
            
            # Get most recent deductive facts
            deductive_stmt = await crud.get_documents(
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                filter={"level": "deductive"},
                reverse=True
            )
            deductive_docs = (await self.db.execute(deductive_stmt)).scalars().all()[:self.deductive_facts_count]
            
            # Initialize context with retrieved facts
            context = {
                "abductive": [doc.content for doc in abductive_docs],
                "inductive": [doc.content for doc in inductive_docs],
                "deductive": [doc.content for doc in deductive_docs]
            }
            
            logger.debug(
                f"Retrieved facts - Abductive: {len(abductive_docs)}, "
                f"Inductive: {len(inductive_docs)}, "
                f"Deductive: {len(deductive_docs)}"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving facts from collection: {e}")
            # Return empty context if retrieval fails
            return {
                "abductive": [],
                "inductive": [],
                "deductive": []
            }

    async def get_contextualized_facts_for_dialectic(self) -> dict[str, list[str]]:
        """Retrieve facts with their session context for dialectic API calls.
        
        Returns:
            Dictionary with reasoning levels as keys and lists of contextualized facts.
            Facts are grouped by session summary to avoid redundancy and provide context.
        """
        try:
            levels = ["abductive", "inductive", "deductive"]
            context = {}
            
            for level in levels:
                count = getattr(self, f"{level}_facts_count")
                stmt = await crud.get_documents(
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    filter={"level": level},
                    reverse=True
                )
                docs = (await self.db.execute(stmt)).scalars().all()[:count]
                
                # Group facts by session summary to avoid redundancy
                summary_groups = {}
                docs_without_summary = []
                
                # Note: We skip metadata updates during retrieval to avoid concurrent session operations
                # Access tracking will be handled through other paths when facts are saved/updated
                for doc in docs:
                    
                    session_context = doc.h_metadata.get("session_context", "")
                    summary_id = doc.h_metadata.get("summary_id", "")
                    
                    if session_context and summary_id:
                        if summary_id not in summary_groups:
                            summary_groups[summary_id] = {
                                "context": session_context,
                                "facts": []
                            }
                        summary_groups[summary_id]["facts"].append(doc.content)
                    else:
                        docs_without_summary.append(doc)
                
                # Format the output to avoid redundancy
                formatted_facts = []
                
                # Add facts with session context (grouped to avoid repeating context)
                for _summary_id, group in summary_groups.items():
                    context_header = f"From session: {group['context'][:200]}..."
                    facts_list = "\n".join([f"  • {fact_content}" for fact_content in group["facts"]])
                    formatted_facts.append(f"{context_header}\n{facts_list}")
                
                # Add facts without session context - include temporal metadata
                for doc in docs_without_summary:
                    temporal_info = self._format_temporal_metadata_for_dialectic(doc)
                    formatted_facts.append(f"• {doc.content}{temporal_info}")
                
                context[level] = formatted_facts
            
            logger.debug(
                f"Retrieved contextualized facts for dialectic - "
                f"Abductive: {len(context['abductive'])}, "
                f"Inductive: {len(context['inductive'])}, "
                f"Deductive: {len(context['deductive'])}"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving contextualized facts: {e}")
            # Return empty context if retrieval fails
            return {
                "abductive": [],
                "inductive": [],
                "deductive": []
            }

    async def save_facts(
        self,
        facts: list[str],
        similarity_threshold: float = 0.85,
        message_id: str | None = None,
        level: str | None = None,
        session_id: str | None = None
    ) -> None:
        """Save facts to the collection with summary linking.

        Args:
            facts: List of facts to save
            similarity_threshold: Facts with similarity above this threshold are considered duplicates
            message_id: Message ID to associate with the facts
            level: Reasoning level (abductive, inductive, deductive)
            session_id: Session ID to link with existing summary context
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
        
        # Queue facts for asynchronous saving to avoid transaction conflicts
        fact_saver = get_fact_saver_queue()
        
        for fact in facts:
            try:
                # Check if similar fact already exists
                existing_similar = await self._find_similar_fact(fact, similarity_threshold, level)
                
                if existing_similar:
                    # Update existing fact metadata instead of creating duplicate
                    await self._update_similar_fact_metadata(existing_similar, session_id, message_id)
                    logger.debug(f"Updated existing similar fact: {fact[:50]}...")
                    continue
                
                # No similar fact found, create new one
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
                
                # Generate unique task ID for tracking
                task_id = f"{level}_{message_id}_{uuid.uuid4().hex[:8]}"
                
                # Queue the fact for saving
                await fact_saver.queue_fact(
                    content=fact,
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    metadata=metadata,
                    duplicate_threshold=1 - similarity_threshold,
                    task_id=task_id
                )
                logger.debug(f"Queued fact for saving: {fact[:50]}...")
                
            except Exception as e:
                logger.error(f"Error queuing fact '{fact[:50]}...': {e}")
                continue
        
        logger.info(f"Queued {len(facts)} facts for saving (level: {level})")

    def _format_temporal_metadata_for_dialectic(self, doc: models.Document) -> str:
        """
        Format temporal metadata for dialectic fact display.
        Shows both fact genesis and ongoing relevance patterns.
        
        Args:
            doc: Document with temporal metadata
            
        Returns:
            Formatted temporal context string
        """
        temporal_parts = []
        
        # Fact inception - when originally derived
        if doc.created_at:
            temporal_parts.append(f"derived {doc.created_at.isoformat()}")
        
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
            temporal_parts.append(f"last accessed {last_accessed}")
        
        if temporal_parts:
            return f" [{', '.join(temporal_parts)}]"
        else:
            return ""

    async def _update_fact_access_metadata(self, doc: models.Document) -> None:
        """Update metadata to track when and how often facts are accessed."""
        try:
            
            updated_metadata = doc.h_metadata.copy() if doc.h_metadata else {}
            updated_metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
            updated_metadata["access_count"] = updated_metadata.get("access_count", 0) + 1
            
            # Update the document with new metadata
            document_update = schemas.DocumentUpdate(metadata=updated_metadata)
            await crud.update_document(
                self.db,
                document_update,
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                document_id=doc.public_id
            )
            
        except Exception as e:
            logger.warning(f"Failed to update access metadata for document {doc.public_id}: {e}")

    async def _find_similar_fact(self, fact: str, similarity_threshold: float, level: str | None = None) -> models.Document | None:
        """Find an existing fact that is semantically similar to the given fact."""
        try:
            filter_dict = {}
            if level:
                filter_dict["level"] = level
            
            # Query for similar facts using semantic search
            documents = await crud.query_documents(
                self.db,
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                query=fact,
                max_distance=1 - similarity_threshold,  # Convert similarity to distance
                top_k=1,
                filter=filter_dict
            )
            
            # Return the most similar document if one exists
            docs_list = list(documents)
            return docs_list[0] if docs_list else None
            
        except Exception as e:
            logger.warning(f"Error finding similar fact: {e}")
            return None

    async def _update_similar_fact_metadata(self, doc: models.Document, session_id: str | None = None, message_id: str | None = None) -> None:
        """Update metadata of an existing similar fact to track recent access."""
        try:
            
            updated_metadata = doc.h_metadata.copy() if doc.h_metadata else {}
            updated_metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
            updated_metadata["access_count"] = updated_metadata.get("access_count", 0) + 1
            
            # Add current session/message to accessed_sessions list
            accessed_sessions = updated_metadata.get("accessed_sessions", [])
            if session_id and session_id not in accessed_sessions:
                accessed_sessions.append(session_id)
                updated_metadata["accessed_sessions"] = accessed_sessions[-10:]  # Keep last 10 sessions
            
            # Track latest message that referenced this fact
            if message_id:
                updated_metadata["latest_message_id"] = message_id
            
            # Update the document
            document_update = schemas.DocumentUpdate(metadata=updated_metadata)
            await crud.update_document(
                self.db,
                document_update,
                app_id=self.app_id,
                user_id=self.user_id,
                collection_id=self.collection_id,
                document_id=doc.public_id
            )
            
            logger.debug(f"Updated metadata for similar fact: {doc.content[:50]}...")
            
        except Exception as e:
            logger.warning(f"Failed to update similar fact metadata for document {doc.public_id}: {e}")

    async def get_relevant_facts(
        self, query: str, top_k: int = 5, max_distance: float = 0.3
    ) -> list[models.Document]:
        """Retrieve the most relevant facts for a given query.

        Args:
            query: The query text to find relevant facts for
            top_k: Maximum number of facts to return
            similarity_threshold: Minimum similarity score for a fact to be considered relevant

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
        self, facts: list[str], similarity_threshold: float = 0.85
    ) -> list[str]:
        """Remove facts that are duplicates of existing facts in the vector store.

        Args:
            facts: List of facts to check for duplicates
            similarity_threshold: Facts with similarity above this threshold are considered duplicates

        Returns:
            List of facts that are not duplicates of existing facts
        """
        unique_facts = []

        async with tracked_db("embedding_store.remove_duplicates") as db:
            for fact in facts:
                try:
                    # Check for duplicates using the crud function
                    duplicates = await crud.get_duplicate_documents(
                        db,
                        app_id=self.app_id,
                        user_id=self.user_id,
                        collection_id=self.collection_id,
                        content=fact,
                        similarity_threshold=similarity_threshold,
                    )

                    if not duplicates:
                        # No duplicates found, add to unique facts
                        unique_facts.append(fact)
                    else:
                        # Log duplicate found
                        logger.debug(
                            f"Duplicate found: {duplicates[0].content}. Ignoring fact: {fact}"
                        )
                except Exception as e:
                    logger.error(f"Error checking for duplicates: {e}")
                    # If there's an error, still include the fact to avoid losing information
                    unique_facts.append(fact)

        return unique_facts

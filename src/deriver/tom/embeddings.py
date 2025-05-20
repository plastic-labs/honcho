import logging

from sqlalchemy.ext.asyncio import AsyncSession

from ... import crud, schemas
from ...dependencies import tracked_db
from ... import crud, schemas, models

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

    async def save_facts(
        self,
        facts: list[str],
        similarity_threshold: float = 0.85,
        message_id: str | None = None,
        level: str | None = None
    ) -> None:
        """Save facts to the collection.

        Args:
            facts: List of facts to save
            replace_duplicates: If True, replace old duplicates with new facts. If False, discard new duplicates
            similarity_threshold: Facts with similarity above this threshold are considered duplicates
        """
        for fact in facts:
            # Create document with duplicate checking
            try:
                metadata = {}
                if message_id:
                    metadata["message_id"] = message_id
                if level:
                    metadata["level"] = level
                await crud.create_document(
                    self.db,
                    document=schemas.DocumentCreate(content=fact, metadata=metadata),
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    duplicate_threshold=1
                    - similarity_threshold,  # Convert similarity to distance
                )
            except Exception as e:
                logger.error(f"Error creating document: {e}")
                continue

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
        async with tracked_db("embedding_store.get_relevant_facts") as db:
            documents = await crud.query_documents(
                db,
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

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from ... import crud, schemas

logger = logging.getLogger(__name__)


class CollectionEmbeddingStore:
    def __init__(self, db: AsyncSession, app_id: str, user_id: str, collection_id: str):
        self.db = db
        self.app_id = app_id
        self.user_id = user_id
        self.collection_id = collection_id

    async def save_facts(
        self,
        facts: list[str],
        replace_duplicates: bool = True,
        similarity_threshold: float = 0.85,
        message_id: str = None,
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
    ) -> list[str]:
        """Retrieve the most relevant facts for a given query.

        Args:
            query: The query text to find relevant facts for
            top_k: Maximum number of facts to return
            similarity_threshold: Minimum similarity score for a fact to be considered relevant

        Returns:
            List of facts sorted by relevance
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

        return [doc.content for doc in documents]

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

        for fact in facts:
            try:
                # Check for duplicates using the crud function
                duplicates = await crud.get_duplicate_documents(
                    self.db,
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

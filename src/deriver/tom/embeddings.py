import json
import logging
from pathlib import Path
from typing import Optional

# from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession

from ... import crud, schemas

logger = logging.getLogger(__name__)

class LocalEmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path.cwd() / "storage" / "embeddings"
        self.model = SentenceTransformer(model_name)
        self.facts: list[str] = []
        self.embeddings: NDArray[np.float32] = np.array([])  # Store as numpy array for faster operations
        self.use_storage = storage_path is not None
        if self.use_storage:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(exist_ok=True)
            self.facts_file = self.storage_path / "facts.json"
            self.embeddings_file = self.storage_path / "embeddings.npy"
            self.load_storage()
    
    def load_storage(self) -> None:
        if self.facts_file.exists():
            with open(self.facts_file) as f:
                self.facts = json.load(f)
        if self.embeddings_file.exists():
            self.embeddings = np.load(self.embeddings_file)
    
    def save_storage(self) -> None:
        with open(self.facts_file, "w") as f:
            json.dump(self.facts, f)
        np.save(self.embeddings_file, self.embeddings)
    
    def compute_embeddings(self, texts: list[str]) -> NDArray[np.float32]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    
    def compute_similarities(self, new_embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(self.embeddings) == 0:
            return np.zeros((len(new_embeddings), 0), dtype=np.float32)
        # Compute dot product between all pairs of embeddings
        dot_product = np.dot(new_embeddings, self.embeddings.T)
        # Compute norms
        norms1 = np.linalg.norm(new_embeddings, axis=1)
        norms2 = np.linalg.norm(self.embeddings, axis=1)
        # Compute similarities in a vectorized way
        return dot_product / (norms1[:, np.newaxis] * norms2)
    
    async def save_facts(self, facts: list[str], replace_duplicates: bool = True, similarity_threshold: float = 0.85) -> None:
        """Save facts and handle duplicates.
        
        Args:
            facts: List of facts to save
            replace_duplicates: If True, replace old duplicates with new facts. If False, discard new duplicates
            similarity_threshold: Facts with similarity above this threshold are considered duplicates
        """
        if not facts:
            return

        if not self.facts:
            self.facts.extend(facts)
            self.embeddings = self.compute_embeddings(facts)
            if self.use_storage:
                self.save_storage()
            return
        
        # Compute embeddings for all new facts at once
        new_embeddings = self.compute_embeddings(facts)
        
        # Compute similarities between all new and existing embeddings at once
        similarities = self.compute_similarities(new_embeddings)
        
        # For each new fact, either add it or replace its duplicate
        facts_to_keep = []
        embeddings_to_keep = []
        
        # Track which existing facts to keep
        keep_existing_mask = np.ones(len(self.facts), dtype=bool)
        
        for i, (new_fact, new_embedding) in enumerate(zip(facts, new_embeddings)):
            max_similarity = np.max(similarities[i])
            if max_similarity > similarity_threshold:
                # Found a duplicate
                if replace_duplicates:
                    # Replace the old fact with the new one
                    duplicate_idx = np.argmax(similarities[i])
                    keep_existing_mask[duplicate_idx] = False
                    facts_to_keep.append(new_fact)
                    embeddings_to_keep.append(new_embedding)
            else:
                # No duplicate found, add as new fact
                facts_to_keep.append(new_fact)
                embeddings_to_keep.append(new_embedding)
        
        # Keep non-duplicate existing facts
        self.facts = [fact for i, fact in enumerate(self.facts) if keep_existing_mask[i]]
        self.embeddings = self.embeddings[keep_existing_mask]
        
        # Add new facts
        if facts_to_keep:
            self.facts.extend(facts_to_keep)
            new_embeddings_array = np.stack(embeddings_to_keep)
            if len(self.embeddings) == 0:
                self.embeddings = new_embeddings_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
        
        if self.use_storage:
            self.save_storage()

    async def get_relevant_facts(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> list[str]:
        """Retrieve the most relevant facts for a given query.
        
        Args:
            query: The query text to find relevant facts for
            top_k: Maximum number of facts to return
            similarity_threshold: Minimum similarity score for a fact to be considered relevant
            
        Returns:
            List of facts sorted by relevance
        """
        if not self.facts:
            return []
        
        # Compute query embedding
        query_embedding = self.compute_embeddings([query])
        
        # Compute similarities with all facts
        similarities = self.compute_similarities(query_embedding)
        similarities = similarities[0]  # Get similarities for the single query
        
        # Get indices of facts sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter by similarity threshold and get top_k results
        relevant_facts: list[str] = []
        for idx in sorted_indices:
            similarity = similarities[idx]
            if similarity < similarity_threshold or len(relevant_facts) >= top_k:
                break
            relevant_facts.append(self.facts[idx])
        
        return relevant_facts

class CollectionEmbeddingStore:
    def __init__(self, db: AsyncSession, app_id: str, user_id: str, collection_id: str):
        self.db = db
        self.app_id = app_id
        self.user_id = user_id
        self.collection_id = collection_id

    async def save_facts(self, facts: list[str], replace_duplicates: bool = True, similarity_threshold: float = 0.85) -> None:
        """Save facts to the collection.
        
        Args:
            facts: List of facts to save
            replace_duplicates: If True, replace old duplicates with new facts. If False, discard new duplicates
            similarity_threshold: Facts with similarity above this threshold are considered duplicates
        """
        for fact in facts:
            # Create document with duplicate checking
            try:
                await crud.create_document(
                    self.db,
                    document=schemas.DocumentCreate(content=fact, metadata={}),
                    app_id=self.app_id,
                    user_id=self.user_id,
                    collection_id=self.collection_id,
                    duplicate_threshold=1-similarity_threshold  # Convert similarity to distance
                )
            except Exception as e:
                logger.error(f"Error creating document: {e}")
                continue

    async def get_relevant_facts(self, query: str, top_k: int = 5, max_distance: float = 0.3) -> list[str]:
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
            top_k=top_k
        )
        
        return [doc.content for doc in documents]

    async def remove_duplicates(self, facts: list[str], similarity_threshold: float = 0.85) -> list[str]:
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
                    similarity_threshold=similarity_threshold
                )
                
                if not duplicates:
                    # No duplicates found, add to unique facts
                    unique_facts.append(fact)
                else:
                    # Log duplicate found
                    logger.debug(f"Duplicate found: {duplicates[0].content}. Ignoring fact: {fact}")
            except Exception as e:
                logger.error(f"Error checking for duplicates: {e}")
                # If there's an error, still include the fact to avoid losing information
                unique_facts.append(fact)
                
        return unique_facts 
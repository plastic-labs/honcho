import asyncio
import logging
from collections import defaultdict
from typing import NamedTuple

import tiktoken
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger(__name__)


class BatchItem(NamedTuple):
    """A single item in a batch with its metadata."""

    text: str
    text_id: str
    chunk_index: int


class EmbeddingClient:
    """
    Embedding client for OpenAI with chunking and batching support.
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = settings.LLM.OPENAI_API_KEY
        if not api_key:
            raise ValueError("API key is required")
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)
        self.encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
        self.max_embedding_tokens: int = settings.LLM.MAX_EMBEDDING_TOKENS
        self.max_embedding_tokens_per_request: int = (
            settings.LLM.MAX_EMBEDDING_TOKENS_PER_REQUEST
        )

    async def embed(self, query: str) -> list[float]:
        token_count = len(self.encoding.encode(query))

        if token_count > self.max_embedding_tokens:
            raise ValueError(
                f"Query exceeds maximum token limit of {self.max_embedding_tokens} tokens (got {token_count} tokens)"
            )

        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        return response.data[0].embedding

    async def batch_embed(
        self, id_resource_dict: dict[str, tuple[str, list[int]]]
    ) -> dict[str, list[list[float]]]:
        """
        Embed multiple texts, chunking long ones and batching API calls.

        Args:
            id_resource_dict: Maps text IDs to (text, encoded_tokens) tuples

        Returns:
            Maps text IDs to lists of embedding vectors (one per chunk)
        """
        if not id_resource_dict:
            return {}

        # 1. Prepare chunks for all texts if needed
        text_chunks = self._prepare_chunks(id_resource_dict)

        # 2. Create batches that fit API limits (max 2048 embeddings per request, max 300,000 tokens per request)
        batches = self._create_batches(text_chunks)

        # 3. Process all batches concurrently
        batch_results = await asyncio.gather(
            *[self._process_batch(batch) for batch in batches],
        )

        # 4. Accumulate results preserving chunk order
        return self._accumulate_embeddings(batch_results)

    def _prepare_chunks(
        self, id_resource_dict: dict[str, tuple[str, list[int]]]
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Chunk texts that exceed token limits.

        Args:
            id_resource_dict: Maps text IDs to (text, encoded_tokens) tuples

        Returns:
            Maps text IDs to lists of (chunk_text, token_count) tuples
        """
        return {
            text_id: (
                _chunk_text_with_tokens(
                    text, encoded_tokens, self.max_embedding_tokens, self.encoding
                )
                if len(encoded_tokens) > self.max_embedding_tokens
                else [(text, len(encoded_tokens))]
            )
            for text_id, (text, encoded_tokens) in id_resource_dict.items()
        }

    def _create_batches(
        self, text_chunks: dict[str, list[tuple[str, int]]]
    ) -> list[list[BatchItem]]:
        """
        Group chunks into batches that fit API limits.

        Args:
            text_chunks: Maps text IDs to lists of (chunk_text, token_count) tuples

        Returns:
            List of batches, each containing BatchItem objects
        """
        batches: list[list[BatchItem]] = []
        current_batch: list[BatchItem] = []
        current_tokens = 0

        for text_id, chunks in text_chunks.items():
            for chunk_idx, (chunk_text, chunk_tokens) in enumerate(chunks):
                # Check if adding this chunk would exceed limits
                would_exceed_tokens = (
                    current_tokens + chunk_tokens
                    > self.max_embedding_tokens_per_request
                )
                would_exceed_count = len(current_batch) >= 2048  # OpenAI's input limit

                if current_batch and (would_exceed_tokens or would_exceed_count):
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

                current_batch.append(BatchItem(chunk_text, text_id, chunk_idx))
                current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch(
        self, batch: list[BatchItem]
    ) -> dict[str, dict[int, list[float]]]:
        """
        Process a single batch through the embeddings API.

        Args:
            batch: List of BatchItem objects to embed

        Returns:
            Maps text IDs to {chunk_index: embedding_vector} dictionaries
        """
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small", input=[item.text for item in batch]
            )

            # Organize embeddings by text_id and chunk_index
            result: dict[str, dict[int, list[float]]] = defaultdict(dict)
            for item, embedding_data in zip(batch, response.data, strict=True):
                result[item.text_id][item.chunk_index] = embedding_data.embedding

            return dict(result)

        except Exception:
            logger.exception("Error processing batch")
            raise

    def _accumulate_embeddings(
        self, batch_results: list[dict[str, dict[int, list[float]]]]
    ) -> dict[str, list[list[float]]]:
        """
        Combine batch results into final output, preserving chunk order.

        Args:
            batch_results: List of batch results from _process_batch

        Returns:
            Maps text IDs to ordered lists of embedding vectors
        """
        all_embeddings: dict[str, dict[int, list[float]]] = defaultdict(dict)

        # Collect all embeddings by text_id and chunk_index
        for batch_result in batch_results:
            for text_id, chunk_dict in batch_result.items():
                all_embeddings[text_id].update(chunk_dict)

        # Convert to ordered lists
        return {
            text_id: [chunk_dict[i] for i in sorted(chunk_dict.keys())]
            for text_id, chunk_dict in all_embeddings.items()
        }


def _chunk_text_with_tokens(
    text: str,
    encoded_tokens: list[int],
    max_tokens: int,
    encoding: tiktoken.Encoding,
) -> list[tuple[str, int]]:
    """
    Split text into chunks that fit within token limits, with 20% overlap.

    Args:
        text: Original text to chunk
        encoded_tokens: Pre-encoded tokens for the text
        max_tokens: Maximum tokens per chunk
        encoding: Tiktoken encoding model

    Returns:
        List of (chunk_text, token_count) tuples
    """
    if len(encoded_tokens) <= max_tokens:
        return [(text, len(encoded_tokens))]

    # Use 20% overlap for better semantic continuity
    overlap_tokens = int(max_tokens * 0.2)
    step_size = max_tokens - overlap_tokens

    return [
        (
            encoding.decode(encoded_tokens[i : i + max_tokens]),
            min(max_tokens, len(encoded_tokens) - i),
        )
        for i in range(0, len(encoded_tokens), step_size)
        if i < len(encoded_tokens)  # Ensure we don't create empty chunks
    ]

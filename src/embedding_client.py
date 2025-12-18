import asyncio
import logging
import threading
from collections import defaultdict
from typing import NamedTuple

import tiktoken
from google import genai
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger(__name__)


class BatchItem(NamedTuple):
    """A single item in a batch with its metadata."""

    text: str
    text_id: str
    chunk_index: int


class _EmbeddingClient:
    """
    Embedding client supporting OpenAI and Gemini with chunking and batching support.
    """

    def __init__(self, api_key: str | None = None, provider: str | None = None):
        self.provider: str = provider or settings.LLM.EMBEDDING_PROVIDER

        if self.provider == "gemini":
            if api_key is None:
                api_key = settings.LLM.GEMINI_API_KEY
            if not api_key:
                raise ValueError("Gemini API key is required")
            self.client: genai.Client | AsyncOpenAI = genai.Client(api_key=api_key)
            self.model: str = "gemini-embedding-001"
            # Gemini has a 2048 token limit
            self.max_embedding_tokens: int = min(settings.MAX_EMBEDDING_TOKENS, 2048)
            # Gemini batch size is not documented, using conservative estimate
            self.max_batch_size: int = 100
        elif self.provider == "openrouter":
            if api_key is None:
                api_key = settings.LLM.OPENAI_COMPATIBLE_API_KEY
            if not api_key:
                raise ValueError(
                    "OpenRouter API key (LLM_OPENAI_COMPATIBLE_API_KEY) is required"
                )
            base_url = (
                settings.LLM.OPENAI_COMPATIBLE_BASE_URL
                or "https://openrouter.ai/api/v1"
            )
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.model = "openai/text-embedding-3-small"
            self.max_embedding_tokens = settings.MAX_EMBEDDING_TOKENS
            self.max_batch_size = 2048  # Same as OpenAI
        else:  # openai
            if api_key is None:
                api_key = settings.LLM.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = "text-embedding-3-small"
            self.max_embedding_tokens = settings.MAX_EMBEDDING_TOKENS
            self.max_batch_size = 2048  # OpenAI batch limit

        self.encoding: tiktoken.Encoding = tiktoken.get_encoding("o200k_base")
        self.max_embedding_tokens_per_request: int = (
            settings.MAX_EMBEDDING_TOKENS_PER_REQUEST
        )

    async def embed(self, query: str) -> list[float]:
        token_count = len(self.encoding.encode(query))

        if token_count > self.max_embedding_tokens:
            raise ValueError(
                f"Query exceeds maximum token limit of {self.max_embedding_tokens} tokens (got {token_count} tokens)"
            )

        if isinstance(self.client, genai.Client):
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=query,
                config={"output_dimensionality": 1536},
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise ValueError("No embedding returned from Gemini API")
            return response.embeddings[0].values
        else:  # openai
            response = await self.client.embeddings.create(
                model=self.model, input=query
            )
            return response.data[0].embedding

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Simple batch embedding for a list of text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors corresponding to input texts

        Raises:
            ValueError: If any text exceeds token limits
        """
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            try:
                if isinstance(self.client, genai.Client):
                    # Type cast needed due to genai type signature complexity
                    response = await self.client.aio.models.embed_content(
                        model=self.model,
                        contents=batch,  # pyright: ignore[reportArgumentType]
                        config={"output_dimensionality": 1536},
                    )
                    if response.embeddings:
                        for emb in response.embeddings:
                            if emb.values:
                                embeddings.append(emb.values)
                else:  # openai
                    response = await self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                    )
                    embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                # Check if it's a token limit error and re-raise as ValueError for consistency
                if "token" in str(e).lower():
                    raise ValueError(
                        f"Text content exceeds maximum token limit of {self.max_embedding_tokens}."
                    ) from e
                raise

        return embeddings

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
                would_exceed_count = len(current_batch) >= self.max_batch_size

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
        self, batch: list[BatchItem], max_retries: int = 3
    ) -> dict[str, dict[int, list[float]]]:
        """
        Process a single batch through the embeddings API with retry logic.

        Args:
            batch: List of BatchItem objects to embed
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Maps text IDs to {chunk_index: embedding_vector} dictionaries
        """
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Organize embeddings by text_id and chunk_index
                result: dict[str, dict[int, list[float]]] = defaultdict(dict)

                if isinstance(self.client, genai.Client):
                    response = await self.client.aio.models.embed_content(
                        model=self.model,
                        contents=[item.text for item in batch],
                        config={"output_dimensionality": 1536},
                    )
                    if response.embeddings:
                        for item, embedding in zip(
                            batch, response.embeddings, strict=True
                        ):
                            if embedding.values:
                                result[item.text_id][item.chunk_index] = (
                                    embedding.values
                                )
                else:  # openai / openrouter
                    response = await self.client.embeddings.create(
                        model=self.model, input=[item.text for item in batch]
                    )
                    for item, embedding_data in zip(batch, response.data, strict=True):
                        result[item.text_id][item.chunk_index] = (
                            embedding_data.embedding
                        )

                return dict(result)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    logger.warning(
                        f"Embedding batch failed (attempt {attempt + 1}/{max_retries}), "
                        + f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception("Error processing batch after all retries")

        raise last_exception or RuntimeError("Batch processing failed")

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


class EmbeddingClient:
    """
    Singleton wrapper for the embedding client with deferred loading.

    The actual client is only initialized on first use, improving startup time
    and allowing the application to start even if API keys are not yet configured.
    """

    _instance: "_EmbeddingClient | None" = None
    _lock: threading.Lock = threading.Lock()
    _wrapper_instance: "EmbeddingClient | None" = None

    def __new__(cls):
        """Ensure only one instance of EmbeddingClient exists."""
        # We always return the same wrapper instance
        if cls._wrapper_instance is None:
            cls._wrapper_instance = super().__new__(cls)
        return cls._wrapper_instance

    def _get_client(self) -> _EmbeddingClient:
        """
        Get or create the underlying embedding client instance.

        Uses double-checked locking for thread-safe lazy initialization.
        """
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    provider = settings.LLM.EMBEDDING_PROVIDER
                    if provider == "gemini":
                        api_key = settings.LLM.GEMINI_API_KEY
                    elif provider == "openrouter":
                        api_key = settings.LLM.OPENAI_COMPATIBLE_API_KEY
                    else:
                        api_key = settings.LLM.OPENAI_API_KEY

                    self._instance = _EmbeddingClient(
                        api_key=api_key, provider=provider
                    )
                    logger.debug(
                        f"Initialized embedding client with provider: {provider}"
                    )

        return self._instance

    async def embed(self, query: str) -> list[float]:
        """Embed a single query string."""
        return await self._get_client().embed(query)

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Simple batch embedding for a list of text strings."""
        return await self._get_client().simple_batch_embed(texts)

    async def batch_embed(
        self, id_resource_dict: dict[str, tuple[str, list[int]]]
    ) -> dict[str, list[list[float]]]:
        """Embed multiple texts, chunking long ones and batching API calls."""
        return await self._get_client().batch_embed(id_resource_dict)

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._get_client().provider

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._get_client().model

    @property
    def max_embedding_tokens(self) -> int:
        """Get the maximum embedding tokens."""
        return self._get_client().max_embedding_tokens

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Get the tiktoken encoding."""
        return self._get_client().encoding


# Shared singleton embedding client instance
embedding_client = EmbeddingClient()

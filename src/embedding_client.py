import asyncio
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Literal, NamedTuple, TypeVar

import tiktoken
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from .config import EmbeddingModelConfig, resolve_embedding_model_config, settings

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _emit_embedding_call(
    *,
    provider: str,
    model: str,
    texts: list[str],
    input_tokens_estimate: int,
    fn: Callable[[], Awaitable[_T]],
    is_final_attempt: bool = True,
) -> _T:
    """time a single embedding-provider call, emit
    `embedding.call.completed` on both success and exception, and return the
    call's result. Errors propagate unchanged — telemetry never bleeds into
    the caller's control flow.

    Caller-supplied `texts` is used only for `input_count`; we don't keep the
    list around for the event to avoid leaking content into telemetry.

    `is_final_attempt` defaults to True so one-shot callers (`embed`,
    `simple_batch_embed`) get correct semantics without changes. Retry-loop
    callers (`_process_batch`) pass the real attempt index so dashboards
    can distinguish exhausted retries from mid-retry failures.
    """
    start = time.perf_counter()
    error: BaseException | None = None
    try:
        return await fn()
    except BaseException as exc:
        error = exc
        raise
    finally:
        if error is None:
            outcome: Literal["success", "error", "cancelled"] = "success"
        elif isinstance(error, asyncio.CancelledError):
            outcome = "cancelled"
        else:
            outcome = "error"
        _publish_embedding_event(
            provider=provider,
            model=model,
            input_count=len(texts),
            input_tokens_estimate=input_tokens_estimate,
            duration_ms=(time.perf_counter() - start) * 1000,
            outcome=outcome,
            error=error,
            is_final_attempt=is_final_attempt,
        )


def _publish_embedding_event(
    *,
    provider: str,
    model: str,
    input_count: int,
    input_tokens_estimate: int,
    duration_ms: float,
    outcome: Literal["success", "error", "cancelled"],
    error: BaseException | None,
    is_final_attempt: bool,
) -> None:
    """Build and emit the EmbeddingCallCompletedEvent. Best-effort."""
    try:
        from src.telemetry.events import (
            EmbeddingCallCompletedEvent,
            EmbeddingCallPurpose,
            emit,
        )
        from src.utils.types import (
            get_embedding_call_purpose,
            get_embedding_parent_category,
            get_embedding_run_id,
            get_embedding_workspace_name,
        )

        # call_purpose travels via ContextVar so embedding callers don't have
        # to thread it through every call site. Unknown values drop to None
        # rather than raising — keeps telemetry resilient to drift.
        purpose_slug = get_embedding_call_purpose()
        call_purpose: EmbeddingCallPurpose | None = None
        if purpose_slug:
            try:
                call_purpose = EmbeddingCallPurpose(purpose_slug)
            except ValueError:
                logger.debug(
                    "Unknown embedding_call_purpose=%r; emitting without",
                    purpose_slug,
                )

        emit(
            EmbeddingCallCompletedEvent(
                workspace_name=get_embedding_workspace_name(),
                call_purpose=call_purpose,
                parent_category=get_embedding_parent_category(),
                provider=provider,
                model=model,
                input_count=input_count,
                input_tokens_estimate=input_tokens_estimate,
                duration_ms=duration_ms,
                outcome=outcome,
                is_final_attempt=is_final_attempt,
                error_class=type(error).__name__ if error is not None else None,
                run_id=get_embedding_run_id(),
            )
        )
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("Failed to emit EmbeddingCallCompletedEvent", exc_info=True)


class BatchItem(NamedTuple):
    """A single item in a batch with its metadata."""

    text: str
    text_id: str
    chunk_index: int
    token_count: int


class _EmbeddingClient:
    """
    Embedding client supporting OpenAI and Gemini with chunking and batching support.
    """

    def __init__(
        self,
        config: EmbeddingModelConfig,
        *,
        vector_dimensions: int,
        max_input_tokens: int,
        max_tokens_per_request: int,
        send_dimensions: bool,
    ):
        self.transport: str = config.transport
        self.model: str = config.model
        self.vector_dimensions: int = vector_dimensions
        self.send_dimensions: bool = send_dimensions

        if self.transport == "gemini":
            if not config.api_key:
                raise ValueError("Gemini API key is required")
            http_options = (
                genai_types.HttpOptions(base_url=config.base_url)
                if config.base_url
                else None
            )
            self.client: genai.Client | AsyncOpenAI = genai.Client(
                api_key=config.api_key,
                http_options=http_options,
            )
            # Gemini has a 2048 token limit
            self.max_embedding_tokens: int = min(max_input_tokens, 2048)
            # Gemini batch size is not documented, using conservative estimate
            self.max_batch_size: int = 100
        else:  # openai
            if not config.api_key:
                raise ValueError("OpenAI API key is required")
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            self.max_embedding_tokens = max_input_tokens
            self.max_batch_size = 2048  # OpenAI batch limit

        try:
            self.encoding: tiktoken.Encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_embedding_tokens_per_request: int = max_tokens_per_request

    @property
    def provider(self) -> str:
        return self.transport

    def _validate_embedding_dimensions(self, embedding: list[float]) -> list[float]:
        if len(embedding) != self.vector_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch for {self.transport}:{self.model}. "
                + f"Expected {self.vector_dimensions}, got {len(embedding)}."
            )
        return embedding

    async def embed(self, query: str) -> list[float]:
        token_count = len(self.encoding.encode(query))

        if token_count > self.max_embedding_tokens:
            raise ValueError(
                f"Query exceeds maximum token limit of {self.max_embedding_tokens} tokens (got {token_count} tokens)"
            )

        # Bind the typed client at the dispatch site so pyright can narrow it
        # for the closures without needing `assert isinstance(...)` (bandit
        # B101). The closures close over the narrowed local, not `self.client`.
        if isinstance(self.client, genai.Client):
            gemini_client = self.client

            async def _call_gemini() -> list[float]:
                response = await gemini_client.aio.models.embed_content(
                    model=self.model,
                    contents=query,
                    config={"output_dimensionality": self.vector_dimensions},
                )
                if not response.embeddings or not response.embeddings[0].values:
                    raise ValueError("No embedding returned from Gemini API")
                return self._validate_embedding_dimensions(
                    response.embeddings[0].values
                )

            return await _emit_embedding_call(
                provider=self.transport,
                model=self.model,
                texts=[query],
                input_tokens_estimate=token_count,
                fn=_call_gemini,
            )

        openai_client = self.client

        async def _call_openai() -> list[float]:
            openai_kwargs: dict[str, Any] = {"model": self.model, "input": [query]}
            if self.send_dimensions:
                openai_kwargs["dimensions"] = self.vector_dimensions
            response = await openai_client.embeddings.create(**openai_kwargs)
            return self._validate_embedding_dimensions(response.data[0].embedding)

        return await _emit_embedding_call(
            provider=self.transport,
            model=self.model,
            texts=[query],
            input_tokens_estimate=token_count,
            fn=_call_openai,
        )

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed a list of text strings. Each input must already fit within
        `max_embedding_tokens`; this method does not sub-chunk oversized inputs.

        Internally goes through the same token-aware batching pipeline as
        `batch_embed()` so the per-request token cap is respected.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text (in order)

        Raises:
            ValueError: If any text exceeds token limits
        """
        if not texts:
            return []

        # Validate per-input token limit and collect token counts for batching
        token_counts: list[int] = []
        for idx, text in enumerate(texts):
            tokens = len(self.encoding.encode(text))
            if tokens > self.max_embedding_tokens:
                raise ValueError(
                    f"Text at index {idx} exceeds maximum token limit of {self.max_embedding_tokens} tokens (got {tokens} tokens)"
                )
            token_counts.append(tokens)

        # Use positional indices as text_ids so we can reassemble in input order.
        text_chunks: dict[str, list[tuple[str, int]]] = {
            str(i): [(text, token_counts[i])] for i, text in enumerate(texts)
        }

        batches = self._create_batches(text_chunks)
        batch_results = await asyncio.gather(
            *[self._process_batch(batch) for batch in batches],
        )

        combined: dict[str, list[list[float]]] = self._accumulate_embeddings(
            batch_results
        )
        return [combined[str(i)][0] for i in range(len(texts))]

    def prepare_chunks(self, id_resource_dict: dict[str, str]) -> dict[str, list[str]]:
        """
        Public helper: tokenize and chunk texts using the same rules as
        `batch_embed()`. Returns ordered chunk texts per input id.

        Intended for callers that want to persist embeddable chunks
        before later embedding them off the request path.
        """
        return {
            text_id: [chunk_text for chunk_text, _ in chunks]
            for text_id, chunks in self._prepare_chunks(id_resource_dict).items()
        }

    async def batch_embed(
        self, id_resource_dict: dict[str, str]
    ) -> dict[str, list[list[float]]]:
        """
        Embed multiple texts, chunking long ones and batching API calls.

        Args:
            id_resource_dict: Maps text IDs to text content

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
        self, id_resource_dict: dict[str, str]
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Chunk texts that exceed token limits.

        Args:
            id_resource_dict: Maps text IDs to text content. We tokenize with
                the embedding client's own encoding so token IDs match the
                decoder vocabulary used by the target embedding API.

        Returns:
            Maps text IDs to lists of (chunk_text, token_count) tuples
        """
        out: dict[str, list[tuple[str, int]]] = {}
        for text_id, text in id_resource_dict.items():
            tokens = self.encoding.encode(text)
            if len(tokens) > self.max_embedding_tokens:
                out[text_id] = _chunk_text_with_tokens(
                    text, tokens, self.max_embedding_tokens, self.encoding
                )
            else:
                out[text_id] = [(text, len(tokens))]
        return out

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

                current_batch.append(
                    BatchItem(chunk_text, text_id, chunk_idx, chunk_tokens)
                )
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

        async def _call_provider() -> dict[str, dict[int, list[float]]]:
            """One provider call. Lifted out of the retry loop so
            _emit_embedding_call emits a separate event per attempt — each
            attempt is a distinct provider hit and shows up as its own line
            item in analytics."""
            result: dict[str, dict[int, list[float]]] = defaultdict(dict)
            if isinstance(self.client, genai.Client):
                response = await self.client.aio.models.embed_content(
                    model=self.model,
                    contents=[item.text for item in batch],
                    config={"output_dimensionality": self.vector_dimensions},
                )
                if response.embeddings:
                    for item, embedding in zip(batch, response.embeddings, strict=True):
                        if embedding.values:
                            result[item.text_id][item.chunk_index] = (
                                self._validate_embedding_dimensions(embedding.values)
                            )
            else:  # openai
                openai_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": [item.text for item in batch],
                }
                if self.send_dimensions:
                    openai_kwargs["dimensions"] = self.vector_dimensions
                response = await self.client.embeddings.create(**openai_kwargs)
                for item, embedding_data in zip(batch, response.data, strict=True):
                    result[item.text_id][item.chunk_index] = (
                        self._validate_embedding_dimensions(embedding_data.embedding)
                    )
            return result

        # Token counts were computed during chunk prep; reuse them here so the
        # provider call doesn't re-encode every chunk just for the size proxy.
        batch_tokens_estimate = sum(item.token_count for item in batch)
        batch_texts = [item.text for item in batch]

        for attempt in range(max_retries):
            try:
                result = await _emit_embedding_call(
                    provider=self.transport,
                    model=self.model,
                    texts=batch_texts,
                    input_tokens_estimate=batch_tokens_estimate,
                    fn=_call_provider,
                    is_final_attempt=(attempt >= max_retries - 1),
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
    _instance_signature: tuple[object, ...] | None = None
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
        signature = self._get_settings_signature()
        if self._instance is None or self._instance_signature != signature:
            with self._lock:
                if self._instance is None or self._instance_signature != signature:
                    runtime_config = self._resolve_runtime_config()
                    self._instance = _EmbeddingClient(
                        runtime_config,
                        vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
                        max_input_tokens=settings.EMBEDDING.MAX_INPUT_TOKENS,
                        max_tokens_per_request=settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
                        send_dimensions=settings.EMBEDDING.resolve_send_dimensions(),
                    )
                    self._instance_signature = signature
                    logger.debug(
                        "Initialized embedding client with transport: %s model: %s",
                        runtime_config.transport,
                        runtime_config.model,
                    )

        return self._instance

    def _resolve_runtime_config(self) -> EmbeddingModelConfig:
        return resolve_embedding_model_config(settings.EMBEDDING.MODEL_CONFIG)

    def _get_settings_signature(self) -> tuple[object, ...]:
        runtime_config = self._resolve_runtime_config()
        return (
            runtime_config.transport,
            runtime_config.model,
            runtime_config.api_key,
            runtime_config.base_url,
            settings.EMBEDDING.VECTOR_DIMENSIONS,
            settings.EMBEDDING.MAX_INPUT_TOKENS,
            settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
            settings.EMBEDDING.resolve_send_dimensions(),
        )

    async def embed(self, query: str) -> list[float]:
        """Embed a single query string."""
        return await self._get_client().embed(query)

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed a list of text strings (each must fit token limit)."""
        return await self._get_client().simple_batch_embed(texts)

    def prepare_chunks(self, id_resource_dict: dict[str, str]) -> dict[str, list[str]]:
        """Chunk texts using the same rules as `batch_embed` (no network)."""
        return self._get_client().prepare_chunks(id_resource_dict)

    async def batch_embed(
        self, id_resource_dict: dict[str, str]
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
    def transport(self) -> str:
        """Get the transport name."""
        return self._get_client().transport

    @property
    def max_embedding_tokens(self) -> int:
        """Get the maximum embedding tokens."""
        return self._get_client().max_embedding_tokens

    @property
    def vector_dimensions(self) -> int:
        """Get the configured embedding dimensions."""
        return self._get_client().vector_dimensions

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Get the tiktoken encoding."""
        return self._get_client().encoding


# Shared singleton embedding client instance
embedding_client = EmbeddingClient()

"""OpenAI-compatible ``/embeddings``, answered from a content hash."""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Any

from fastapi import APIRouter

from src.mock_provider.schemas import EmbeddingsRequest

router = APIRouter(tags=["mock-provider"])

# Honcho's default. EmbeddingClient._validate_embedding_dimensions raises when a
# vector comes back at the wrong width, and validate_embedding_schema refuses to
# boot when the width disagrees with the pgvector column, so the request's own
# `dimensions` is honoured whenever it is present.
DEFAULT_DIMENSIONS = 1536


def content_to_embedding(content: str, dimensions: int) -> list[float]:
    """A deterministic vector for ``content``.

    Identical input yields an identical vector, and different inputs differ —
    which is what deduplication logic needs. It carries no semantic similarity:
    two paraphrases are as far apart as two unrelated strings. Anything
    asserting on ranking quality must not use this provider.

    Mirrors ``_content_to_embedding`` in tests/conftest.py.
    """
    digest = hashlib.sha256(content.encode()).digest()
    return [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(dimensions)]


def _encode_base64(vector: list[float]) -> str:
    """Little-endian float32, which is what the OpenAI SDK decodes."""
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()


def _normalize_input(
    raw: str | list[str] | list[int] | list[list[int]] | None,
) -> list[str]:
    """Flatten the request input into one string per embedding to return.

    Token-array inputs are rendered back to a stable string rather than
    rejected — the vector only has to be deterministic, not meaningful.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    # A flat list of ints is one tokenized input, not many single-token ones.
    if raw and all(isinstance(item, int) for item in raw):
        return [",".join(str(item) for item in raw)]

    texts: list[str] = []
    for item in raw:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, list):
            texts.append(",".join(str(part) for part in item))
        else:
            texts.append(str(item))
    return texts


@router.post("/embeddings")
async def embeddings(body: EmbeddingsRequest) -> Any:
    texts = _normalize_input(body.input)
    # A non-positive width is rejected by the request model, so absent is the
    # only case left to fill in.
    dimensions = body.dimensions if body.dimensions is not None else DEFAULT_DIMENSIONS

    data: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        vector = content_to_embedding(text, dimensions)
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": (
                    vector
                    if body.encoding_format == "float"
                    else _encode_base64(vector)
                ),
            }
        )

    prompt_tokens = max(1, sum(len(text) for text in texts) // 4)
    return {
        "object": "list",
        "data": data,
        "model": body.model or "mock-embedding",
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }

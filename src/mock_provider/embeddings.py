"""OpenAI-compatible ``/embeddings``, answered from a content hash."""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Any

from fastapi import APIRouter, Request

from src.mock_provider.coerce import as_dict, as_int, as_list, as_str

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


def _normalize_input(raw: object) -> list[str]:
    """Accept every input shape the OpenAI embeddings API allows.

    Token-array inputs (list[int] / list[list[int]]) are rendered back to a
    stable string rather than rejected — the vector only has to be deterministic.
    """
    if raw is None:
        return []
    if (text := as_str(raw)) is not None:
        return [text]

    items = as_list(raw)
    if items is None:
        return [str(raw)]

    # A flat list of ints is one tokenized input, not many single-token ones.
    if items and all(as_int(item) is not None for item in items):
        return [",".join(str(item) for item in items)]

    texts: list[str] = []
    for item in items:
        if (item_text := as_str(item)) is not None:
            texts.append(item_text)
        elif (parts := as_list(item)) is not None:
            texts.append(",".join(str(part) for part in parts))
        else:
            texts.append(str(item))
    return texts


@router.post("/embeddings")
async def embeddings(request: Request) -> Any:
    payload = as_dict(await request.json()) or {}

    texts = _normalize_input(payload.get("input"))
    requested = as_int(payload.get("dimensions"))
    dimensions = requested if requested and requested > 0 else DEFAULT_DIMENSIONS

    # The SDK omits encoding_format only when it wants base64 (it sets the
    # parameter explicitly for float), so absent means base64.
    encoding_format = as_str(payload.get("encoding_format")) or "base64"

    data: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        vector = content_to_embedding(text, dimensions)
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": (
                    vector if encoding_format == "float" else _encode_base64(vector)
                ),
            }
        )

    prompt_tokens = max(1, sum(len(text) for text in texts) // 4)
    return {
        "object": "list",
        "data": data,
        "model": as_str(payload.get("model")) or "mock-embedding",
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }

"""
Vector store utility functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from src.vector_store import VectorRecord, VectorStore, VectorUpsertResult

logger = logging.getLogger(__name__)


async def upsert_with_retry(
    vector_store: VectorStore,
    namespace: str,
    vector_records: list[VectorRecord],
    max_attempts: int = 3,
) -> VectorUpsertResult | None:
    """
    Upsert vectors with exponential backoff retry.

    Retries on any exception or when secondary store fails (partial success).

    Args:
        vector_store: The vector store to upsert into
        namespace: The namespace for the vectors
        vector_records: List of VectorRecord objects to upsert
        max_attempts: Maximum number of retry attempts (default 3)

    Returns:
        VectorUpsertResult on success, or None if vector_records is empty

    Raises:
        Exception: If all retries fail
    """
    if not vector_records:
        return None

    result: VectorUpsertResult | None = None
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception_type(Exception)
        | retry_if_result(lambda res: res is not None and res.secondary_ok is False),
        reraise=True,
    ):
        with attempt:
            result = await vector_store.upsert_many(namespace, vector_records)

    return result

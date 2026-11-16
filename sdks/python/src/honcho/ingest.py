"""Batching shared by the sync and async ``ingest`` methods."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .api_types import IngestRecord, IngestRecordResult, IngestResponse

INGEST_BATCH_SIZE = 100
"""Records per request: the most the ingest route takes at once."""


def ingest_batches(
    records: Iterable[IngestRecord | Mapping[str, Any]],
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield ``(offset, batch)`` pairs of request-ready records.

    Each record is validated as it is read, so a bad record raises before its
    batch is sent, and an iterator is read lazily, one batch at a time.
    """
    batch: list[dict[str, Any]] = []
    offset = 0
    for record in records:
        model = (
            record
            if isinstance(record, IngestRecord)
            else IngestRecord.model_validate(record)
        )
        batch.append(model.model_dump(mode="json", exclude_none=True))
        if len(batch) == INGEST_BATCH_SIZE:
            yield offset, batch
            offset += len(batch)
            batch = []
    if batch:
        yield offset, batch


def offset_results(data: Any, offset: int) -> list[IngestRecordResult]:
    """Parse one batch's response, with indexes counted across all batches."""
    return [
        result.model_copy(update={"index": result.index + offset})
        for result in IngestResponse.model_validate(data).records
    ]

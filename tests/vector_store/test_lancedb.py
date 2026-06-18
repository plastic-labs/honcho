"""Tests for LanceDBVectorStore query projection behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.vector_store.lancedb import LanceDBVectorStore


def _build_query_chain(rows: list[dict[str, Any]]) -> MagicMock:
    """Build a chainable mock that mirrors LanceDB's async query builder."""
    chain = MagicMock()
    chain.distance_type.return_value = chain
    chain.limit.return_value = chain
    chain.select.return_value = chain
    chain.where.return_value = chain
    chain.to_list = AsyncMock(return_value=rows)
    return chain


def _patch_table(
    store: LanceDBVectorStore, rows: list[dict[str, Any]]
) -> tuple[MagicMock, MagicMock]:
    """Patch _get_table to return a mock whose vector_search yields the chain."""
    chain = _build_query_chain(rows)
    table = MagicMock()
    table.vector_search = MagicMock(return_value=chain)
    store._get_table = AsyncMock(return_value=table)  # pyright: ignore[reportPrivateUsage]
    return table, chain


@pytest.fixture
def store() -> LanceDBVectorStore:
    return LanceDBVectorStore()


@pytest.mark.asyncio
async def test_query_returns_empty_when_table_missing(
    store: LanceDBVectorStore,
) -> None:
    store._get_table = AsyncMock(return_value=None)  # pyright: ignore[reportPrivateUsage]

    results = await store.query("honcho.msg.missing", [0.1, 0.2, 0.3, 0.4])

    assert results == []


@pytest.mark.asyncio
async def test_query_default_does_not_project(store: LanceDBVectorStore) -> None:
    _table, chain = _patch_table(store, rows=[])

    await store.query("honcho.msg.test", [0.1, 0.2, 0.3, 0.4])

    chain.select.assert_not_called()


@pytest.mark.asyncio
async def test_query_with_include_attributes_false_selects_only_id(
    store: LanceDBVectorStore,
) -> None:
    _table, chain = _patch_table(store, rows=[])

    await store.query(
        "honcho.doc.test",
        [0.1, 0.2, 0.3, 0.4],
        include_attributes=False,
    )

    chain.select.assert_called_once_with(["id"])


@pytest.mark.asyncio
async def test_query_with_attribute_list_projects_id_plus_listed(
    store: LanceDBVectorStore,
) -> None:
    _table, chain = _patch_table(store, rows=[])

    await store.query(
        "honcho.msg.test",
        [0.1, 0.2, 0.3, 0.4],
        include_attributes=["message_id"],
    )

    chain.select.assert_called_once_with(["id", "message_id"])


@pytest.mark.asyncio
async def test_query_attribute_list_dedupes_explicit_id(
    store: LanceDBVectorStore,
) -> None:
    _table, chain = _patch_table(store, rows=[])

    await store.query(
        "honcho.msg.test",
        [0.1, 0.2, 0.3, 0.4],
        include_attributes=["id", "message_id"],
    )

    chain.select.assert_called_once_with(["id", "message_id"])


@pytest.mark.asyncio
async def test_query_converts_rows_to_results_with_score_and_metadata(
    store: LanceDBVectorStore,
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "id": "vec_1",
            "_distance": 0.12,
            "vector": [0.0, 0.0, 0.0, 0.0],
            "message_id": "msg_1",
            "session_name": "sess_a",
        },
        {
            "id": "vec_2",
            "_distance": 0.34,
            "message_id": "msg_2",
        },
    ]
    _patch_table(store, rows=rows)

    results = await store.query("honcho.msg.test", [0.1, 0.2, 0.3, 0.4])

    assert [r.id for r in results] == ["vec_1", "vec_2"]
    assert [r.score for r in results] == [0.12, 0.34]
    # id, vector, _distance must not leak into metadata
    assert results[0].metadata == {
        "message_id": "msg_1",
        "session_name": "sess_a",
    }
    assert results[1].metadata == {"message_id": "msg_2"}


@pytest.mark.asyncio
async def test_query_filters_by_max_distance(store: LanceDBVectorStore) -> None:
    rows: list[dict[str, Any]] = [
        {"id": "vec_close", "_distance": 0.05, "message_id": "msg_1"},
        {"id": "vec_far", "_distance": 0.9, "message_id": "msg_2"},
    ]
    _patch_table(store, rows=rows)

    results = await store.query(
        "honcho.msg.test",
        [0.1, 0.2, 0.3, 0.4],
        max_distance=0.5,
    )

    assert [r.id for r in results] == ["vec_close"]

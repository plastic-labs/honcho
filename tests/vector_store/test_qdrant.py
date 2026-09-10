from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from src.exceptions import VectorStoreError
from src.vector_store import VectorRecord
from src.vector_store.qdrant import (
    QdrantVectorStore,
    _point_id,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> QdrantVectorStore:
    monkeypatch.setattr("src.vector_store.qdrant.AsyncQdrantClient", MagicMock())
    return QdrantVectorStore()


def _mock_client(store: QdrantVectorStore) -> MagicMock:
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.get_collection = AsyncMock()
    store._client = client  # pyright: ignore[reportPrivateUsage]
    return client


def _hit(*, id: str, score: float, payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=id, score=score, payload=payload)


def test_point_id_is_deterministic_and_a_valid_uuid() -> None:
    assert _point_id("user_123") == _point_id("user_123")
    assert _point_id("user_123") != _point_id("user_456")
    uuid.UUID(_point_id("user_123"))


def test_build_filter_membership(store: QdrantVectorStore) -> None:
    f = store._build_filter({"session_name": {"in": ["s1", "s2"]}})  # pyright: ignore[reportPrivateUsage]
    assert f is not None and f.must is not None


@pytest.mark.asyncio
async def test_query_returns_empty_when_collection_missing(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.collection_exists = AsyncMock(return_value=False)

    results = await store.query("honcho.msg.missing", [0.1, 0.2, 0.3, 0.4])

    assert results == []
    client.query_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_short_circuits_on_nonpositive_top_k(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.query_points = AsyncMock()

    for top_k in (0, -1):
        assert (
            await store.query("honcho.msg.test", [0.1, 0.2, 0.3, 0.4], top_k=top_k)
            == []
        )

    client.collection_exists.assert_not_awaited()
    client.query_points.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_raises_vector_store_error_on_failure(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.query_points = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(VectorStoreError):
        await store.query("honcho.msg.test", [0.1, 0.2, 0.3, 0.4])


@pytest.mark.asyncio
async def test_query_include_attributes_false_still_recovers_id(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)

    await store.query("honcho.msg.test", [0.1, 0.2, 0.3, 0.4], include_attributes=False)

    assert client.query_points.await_args.kwargs["with_payload"] == ["_id"]


@pytest.mark.asyncio
async def test_query_attribute_list_projects_id_plus_listed(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)

    await store.query(
        "honcho.msg.test",
        [0.1, 0.2, 0.3, 0.4],
        include_attributes=["message_id"],
    )

    assert client.query_points.await_args.kwargs["with_payload"] == [
        "_id",
        "message_id",
    ]


@pytest.mark.asyncio
async def test_query_converts_hits_to_results_with_distance_and_metadata(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.query_points = AsyncMock(
        return_value=SimpleNamespace(
            points=[
                _hit(
                    id="<uuid-1>",
                    score=0.88,
                    payload={"_id": "vec_1", "message_id": "msg_1"},
                ),
                _hit(id="<uuid-2>", score=0.66, payload={"_id": "vec_2"}),
            ]
        )
    )

    results = await store.query("honcho.msg.test", [0.1, 0.2, 0.3, 0.4])

    assert [r.id for r in results] == ["vec_1", "vec_2"]
    assert results[0].score == 1.0 - 0.88
    assert results[1].score == 1.0 - 0.66
    assert results[0].metadata == {"message_id": "msg_1"}
    assert results[1].metadata == {}


@pytest.mark.asyncio
async def test_query_filters_by_max_distance(store: QdrantVectorStore) -> None:
    client = _mock_client(store)
    client.query_points = AsyncMock(
        return_value=SimpleNamespace(
            points=[
                _hit(id="<uuid-1>", score=0.95, payload={"_id": "vec_close"}),
                _hit(id="<uuid-2>", score=0.1, payload={"_id": "vec_far"}),
            ]
        )
    )

    results = await store.query(
        "honcho.msg.test",
        [0.1, 0.2, 0.3, 0.4],
        max_distance=0.5,
    )

    assert [r.id for r in results] == ["vec_close"]


@pytest.mark.asyncio
async def test_probe_returns_none_for_missing_collection(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.collection_exists = AsyncMock(return_value=False)

    assert await store.probe_namespace_dim("does_not_exist") is None


@pytest.mark.asyncio
async def test_probe_returns_declared_dim(store: QdrantVectorStore) -> None:
    client = _mock_client(store)
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=models.VectorParams(
                        size=768, distance=models.Distance.COSINE
                    )
                )
            )
        )
    )

    assert await store.probe_namespace_dim("probe_test") == 768


@pytest.mark.asyncio
async def test_probe_raises_when_vector_config_missing(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=None))
        )
    )

    with pytest.raises(VectorStoreError):
        await store.probe_namespace_dim("corrupt")


@pytest.mark.asyncio
async def test_upsert_many_raises_vector_store_error_on_failure(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.upsert = AsyncMock(side_effect=RuntimeError("boom"))
    record = VectorRecord(id="doc_1", embedding=[0.1, 0.2, 0.3, 0.4])

    with pytest.raises(VectorStoreError):
        await store.upsert_many("honcho.doc.test", [record])


@pytest.mark.asyncio
async def test_delete_many_raises_vector_store_error_on_failure(
    store: QdrantVectorStore,
) -> None:
    client = _mock_client(store)
    client.delete = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(VectorStoreError):
        await store.delete_many("honcho.doc.test", ["doc_1"])

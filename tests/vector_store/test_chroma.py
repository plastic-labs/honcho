"""Tests for ChromaVectorStore filter translation, batching, and error handling."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.exceptions import VectorStoreError
from src.vector_store import VectorRecord

chromadb = pytest.importorskip("chromadb")
from chromadb.errors import NotFoundError  # noqa: E402

from src.vector_store.chroma import ChromaVectorStore  # noqa: E402

# Chroma collection naming rules: 3-512 chars, starts/ends with a lowercase
# alphanumeric, dots/dashes/underscores allowed in between.
_CHROMA_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,510}[a-z0-9]$")


def _query_response(
    rows: list[tuple[str, float, dict[str, Any] | None]],
) -> dict[str, Any]:
    """Build a column-major Chroma query response for one query embedding."""
    return {
        "ids": [[r[0] for r in rows]],
        "distances": [[r[1] for r in rows]],
        "metadatas": [[r[2] for r in rows]],
    }


def _patch_collection(
    store: ChromaVectorStore, collection: MagicMock, batch_limit: int = 300
) -> None:
    """Patch collection/client access so no real Chroma client is created."""
    store._get_collection = AsyncMock(return_value=collection)  # pyright: ignore[reportPrivateUsage]
    store._get_or_create_collection = AsyncMock(return_value=collection)  # pyright: ignore[reportPrivateUsage]
    store._get_client = AsyncMock(return_value=MagicMock())  # pyright: ignore[reportPrivateUsage]
    store._max_batch_size = batch_limit  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def store() -> ChromaVectorStore:
    return ChromaVectorStore()


@pytest.fixture
def record() -> VectorRecord:
    return VectorRecord(
        id="doc_1", embedding=[0.1, 0.2, 0.3, 0.4], metadata={"foo": "bar"}
    )


# === Collection name mapping ===


def test_collection_name_is_valid_and_deterministic(store: ChromaVectorStore) -> None:
    ns = store.get_vector_namespace("document", "ws1", observer="a", observed="b")
    name = store._collection_name(ns)  # pyright: ignore[reportPrivateUsage]

    assert _CHROMA_NAME_PATTERN.match(name), name
    assert name == store._collection_name(ns)  # pyright: ignore[reportPrivateUsage]


def test_collection_name_distinct_for_distinct_namespaces(
    store: ChromaVectorStore,
) -> None:
    ns_a = store.get_vector_namespace("document", "ws1", observer="a", observed="b")
    ns_b = store.get_vector_namespace("document", "ws1", observer="b", observed="a")

    assert store._collection_name(ns_a) != store._collection_name(ns_b)  # pyright: ignore[reportPrivateUsage]


# === Filter translation ===


def test_build_where_single_equality_is_bare_clause(store: ChromaVectorStore) -> None:
    where = store._build_where({"level": "explicit"})  # pyright: ignore[reportPrivateUsage]
    assert where == {"level": {"$eq": "explicit"}}


def test_build_where_in_operator(store: ChromaVectorStore) -> None:
    where = store._build_where({"session_name": {"in": ["s1", "s2"]}})  # pyright: ignore[reportPrivateUsage]
    assert where == {"session_name": {"$in": ["s1", "s2"]}}


def test_build_where_multiple_clauses_combine_with_and(
    store: ChromaVectorStore,
) -> None:
    where = store._build_where(  # pyright: ignore[reportPrivateUsage]
        {"level": "explicit", "session_name": {"in": ["s1"]}}
    )
    assert where == {
        "$and": [
            {"level": {"$eq": "explicit"}},
            {"session_name": {"$in": ["s1"]}},
        ]
    }


def test_build_where_empty_returns_none(store: ChromaVectorStore) -> None:
    assert store._build_where({}) is None  # pyright: ignore[reportPrivateUsage]


def test_build_where_null_value_raises(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="null"):
        store._build_where({"session_name": None})  # pyright: ignore[reportPrivateUsage]


# === Metadata sanitization ===


def test_sanitize_metadata_drops_none_and_reserved_keys() -> None:
    sanitized = ChromaVectorStore._sanitize_metadata(  # pyright: ignore[reportPrivateUsage]
        {"level": "explicit", "session_name": None, "id": "x", "embedding": [0.1]}
    )
    assert sanitized == {"level": "explicit"}


def test_sanitize_metadata_empty_returns_none() -> None:
    assert ChromaVectorStore._sanitize_metadata({"session_name": None}) is None  # pyright: ignore[reportPrivateUsage]


# === Upsert ===


@pytest.mark.asyncio
async def test_upsert_many_short_circuits_on_empty(store: ChromaVectorStore) -> None:
    collection = MagicMock()
    _patch_collection(store, collection)

    await store.upsert_many("honcho.doc.test", [])

    collection.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_many_splits_batches(store: ChromaVectorStore) -> None:
    collection = MagicMock()
    _patch_collection(store, collection, batch_limit=2)
    vectors = [
        VectorRecord(id=f"doc_{i}", embedding=[0.1, 0.2], metadata={"level": "e"})
        for i in range(5)
    ]

    await store.upsert_many("honcho.doc.test", vectors)

    assert collection.upsert.call_count == 3
    first_call = collection.upsert.call_args_list[0]
    assert first_call.kwargs["ids"] == ["doc_0", "doc_1"]
    last_call = collection.upsert.call_args_list[-1]
    assert last_call.kwargs["ids"] == ["doc_4"]


@pytest.mark.asyncio
async def test_upsert_many_raises_vector_store_error_on_transport_failure(
    store: ChromaVectorStore, record: VectorRecord
) -> None:
    collection = MagicMock()
    collection.upsert.side_effect = httpx.ConnectError("connection refused")
    _patch_collection(store, collection)

    with pytest.raises(VectorStoreError) as excinfo:
        await store.upsert_many("honcho.doc.test", [record])

    assert "honcho.doc.test" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


# === Query ===


@pytest.mark.asyncio
async def test_query_returns_empty_when_collection_missing(
    store: ChromaVectorStore,
) -> None:
    store._get_collection = AsyncMock(return_value=None)  # pyright: ignore[reportPrivateUsage]

    results = await store.query("honcho.msg.missing", [0.1, 0.2, 0.3, 0.4])

    assert results == []


@pytest.mark.asyncio
async def test_query_returns_results_with_metadata(store: ChromaVectorStore) -> None:
    collection = MagicMock()
    collection.query.return_value = _query_response(
        [("doc_1", 0.1, {"level": "explicit"}), ("doc_2", 0.5, None)]
    )
    _patch_collection(store, collection)

    results = await store.query("honcho.doc.test", [0.1, 0.2, 0.3, 0.4])

    assert [(r.id, r.score, r.metadata) for r in results] == [
        ("doc_1", 0.1, {"level": "explicit"}),
        ("doc_2", 0.5, {}),
    ]


@pytest.mark.asyncio
async def test_query_applies_max_distance_client_side(
    store: ChromaVectorStore,
) -> None:
    collection = MagicMock()
    collection.query.return_value = _query_response(
        [("near", 0.1, None), ("far", 0.9, None)]
    )
    _patch_collection(store, collection)

    results = await store.query(
        "honcho.doc.test", [0.1, 0.2, 0.3, 0.4], max_distance=0.5
    )

    assert [r.id for r in results] == ["near"]


@pytest.mark.asyncio
async def test_query_include_attributes_false_skips_metadata(
    store: ChromaVectorStore,
) -> None:
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["doc_1"]],
        "distances": [[0.1]],
        "metadatas": None,
    }
    _patch_collection(store, collection)

    results = await store.query(
        "honcho.doc.test", [0.1, 0.2, 0.3, 0.4], include_attributes=False
    )

    assert collection.query.call_args.kwargs["include"] == ["distances"]
    assert results[0].metadata == {}


@pytest.mark.asyncio
async def test_query_attribute_list_projects_client_side(
    store: ChromaVectorStore,
) -> None:
    collection = MagicMock()
    collection.query.return_value = _query_response(
        [("doc_1", 0.1, {"level": "explicit", "session_name": "s1"})]
    )
    _patch_collection(store, collection)

    results = await store.query(
        "honcho.doc.test", [0.1, 0.2, 0.3, 0.4], include_attributes=["level"]
    )

    assert results[0].metadata == {"level": "explicit"}


@pytest.mark.asyncio
async def test_query_passes_translated_filters(store: ChromaVectorStore) -> None:
    collection = MagicMock()
    collection.query.return_value = _query_response([])
    _patch_collection(store, collection)

    await store.query(
        "honcho.doc.test",
        [0.1, 0.2, 0.3, 0.4],
        filters={"session_name": {"in": ["s1", "s2"]}},
    )

    assert collection.query.call_args.kwargs["where"] == {
        "session_name": {"$in": ["s1", "s2"]}
    }


@pytest.mark.asyncio
async def test_query_returns_empty_on_transport_failure(
    store: ChromaVectorStore,
) -> None:
    collection = MagicMock()
    collection.query.side_effect = httpx.ConnectError("connection refused")
    _patch_collection(store, collection)

    results = await store.query("honcho.doc.test", [0.1, 0.2, 0.3, 0.4])

    assert results == []


# === Delete ===


@pytest.mark.asyncio
async def test_delete_many_short_circuits_on_empty(store: ChromaVectorStore) -> None:
    collection = MagicMock()
    _patch_collection(store, collection)

    await store.delete_many("honcho.doc.test", [])

    collection.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_many_noop_when_collection_missing(
    store: ChromaVectorStore,
) -> None:
    store._get_collection = AsyncMock(return_value=None)  # pyright: ignore[reportPrivateUsage]

    await store.delete_many("honcho.doc.missing", ["doc_1"])


@pytest.mark.asyncio
async def test_delete_many_raises_vector_store_error_on_transport_failure(
    store: ChromaVectorStore,
) -> None:
    collection = MagicMock()
    collection.delete.side_effect = httpx.ConnectError("connection refused")
    _patch_collection(store, collection)

    with pytest.raises(VectorStoreError):
        await store.delete_many("honcho.doc.test", ["doc_1"])


@pytest.mark.asyncio
async def test_delete_namespace_noop_when_collection_missing(
    store: ChromaVectorStore,
) -> None:
    client = MagicMock()
    client.delete_collection.side_effect = NotFoundError("does not exist")
    store._get_client = AsyncMock(return_value=client)  # pyright: ignore[reportPrivateUsage]

    await store.delete_namespace("honcho.doc.missing")


# === Real embedded round-trip ===


@pytest.mark.asyncio
async def test_persistent_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Exercise the real embedded client end-to-end in a tmp dir."""
    monkeypatch.setattr(
        "src.config.settings.VECTOR_STORE.CHROMA_CLIENT_MODE", "persistent"
    )
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.CHROMA_PATH", str(tmp_path))

    store = ChromaVectorStore()
    ns = store.get_vector_namespace("document", "ws1", observer="a", observed="b")
    try:
        await store.upsert_many(
            ns,
            [
                VectorRecord(
                    id="doc_1",
                    embedding=[0.1, 0.2, 0.3, 0.4],
                    metadata={"level": "explicit", "session_name": "s1"},
                ),
                VectorRecord(
                    id="doc_2",
                    embedding=[0.9, 0.8, 0.7, 0.6],
                    metadata={"level": "deductive", "session_name": None},
                ),
            ],
        )

        results = await store.query(ns, [0.1, 0.2, 0.3, 0.4], top_k=5)
        assert [r.id for r in results] == ["doc_1", "doc_2"]
        assert results[0].score < results[1].score
        # None-valued metadata keys are stripped on write
        assert "session_name" not in results[1].metadata

        filtered = await store.query(
            ns, [0.1, 0.2, 0.3, 0.4], top_k=5, filters={"level": "deductive"}
        )
        assert [r.id for r in filtered] == ["doc_2"]

        assert await store.probe_namespace_dim(ns) == 4

        await store.delete_many(ns, ["doc_1"])
        remaining = await store.query(ns, [0.1, 0.2, 0.3, 0.4], top_k=5)
        assert [r.id for r in remaining] == ["doc_2"]

        await store.delete_namespace(ns)
        assert await store.probe_namespace_dim(ns) is None
    finally:
        await store.close()


@pytest.mark.parametrize("membership", [["explicit"], ("explicit",), {"explicit"}])
async def test_persistent_bare_membership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, membership: Any
) -> None:
    monkeypatch.setattr(
        "src.config.settings.VECTOR_STORE.CHROMA_CLIENT_MODE", "persistent"
    )
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.CHROMA_PATH", str(tmp_path))
    store = ChromaVectorStore()
    try:
        await store.upsert_many(
            "membership",
            [
                VectorRecord(
                    id="match", embedding=[1.0, 0.0], metadata={"level": "explicit"}
                ),
                VectorRecord(
                    id="other", embedding=[1.0, 0.0], metadata={"level": "deductive"}
                ),
            ],
        )
        results = await store.query(
            "membership", [1.0, 0.0], filters={"level": membership}
        )
        assert [result.id for result in results] == ["match"]
    finally:
        await store.close()


@pytest.mark.parametrize("membership", [{"in": []}, [], (), set[str]()])
async def test_persistent_empty_membership_matches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, membership: Any
) -> None:
    monkeypatch.setattr(
        "src.config.settings.VECTOR_STORE.CHROMA_CLIENT_MODE", "persistent"
    )
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.CHROMA_PATH", str(tmp_path))
    store = ChromaVectorStore()
    try:
        await store.upsert_many(
            "membership",
            [
                VectorRecord(
                    id="present",
                    embedding=[1.0, 0.0],
                    metadata={"level": "explicit", "session_name": "s1"},
                ),
                VectorRecord(
                    id="empty",
                    embedding=[1.0, 0.0],
                    metadata={"level": "explicit", "session_name": ""},
                ),
                VectorRecord(
                    id="missing", embedding=[1.0, 0.0], metadata={"level": "explicit"}
                ),
            ],
        )
        assert (
            await store.query(
                "membership", [1.0, 0.0], filters={"session_name": membership}
            )
            == []
        )
        assert (
            await store.query(
                "membership",
                [1.0, 0.0],
                filters={"level": "explicit", "session_name": membership},
            )
            == []
        )
    finally:
        await store.close()


@pytest.mark.parametrize("top_k", [0, -1])
async def test_nonpositive_top_k_skips_client(
    store: ChromaVectorStore, top_k: int
) -> None:
    client = AsyncMock(side_effect=AssertionError("must not connect"))
    store._get_client = client  # pyright: ignore[reportPrivateUsage]
    assert await store.query("unused", [1.0, 0.0], top_k=top_k) == []
    client.assert_not_called()


async def test_probe_retains_dimension_after_deleting_every_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setattr(
        "src.config.settings.VECTOR_STORE.CHROMA_CLIENT_MODE", "persistent"
    )
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.CHROMA_PATH", str(tmp_path))
    store = ChromaVectorStore()
    await store.upsert_many("dimension", [VectorRecord(id="a", embedding=[0.1] * 4)])
    await store.delete_many("dimension", ["a"])
    await store.close()

    reopened = ChromaVectorStore()
    try:
        assert await reopened.probe_namespace_dim("dimension") == 4
        # The probe must agree with the dimension still enforced by Chroma.
        with pytest.raises(chromadb.errors.InvalidArgumentError, match="dimension"):
            await reopened.upsert_many(
                "dimension", [VectorRecord(id="b", embedding=[0.1] * 8)]
            )
    finally:
        await reopened.close()


async def test_probe_returns_none_for_never_written_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setattr(
        "src.config.settings.VECTOR_STORE.CHROMA_CLIENT_MODE", "persistent"
    )
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.CHROMA_PATH", str(tmp_path))
    store = ChromaVectorStore()
    try:
        await store._get_or_create_collection("fresh")  # pyright: ignore[reportPrivateUsage]
        assert await store.probe_namespace_dim("fresh") is None
    finally:
        await store.close()

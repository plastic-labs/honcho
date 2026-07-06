"""Tests for MilvusVectorStore behavior."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

import pytest

from src.config import settings
from src.vector_store import VectorRecord
from src.vector_store.milvus import (
    MAX_COLLECTION_NAME_LENGTH,
    MilvusVectorStore,
)

logging.getLogger("milvus_lite.server_manager").disabled = True


def _helper_store() -> MilvusVectorStore:
    """Create a MilvusVectorStore instance without opening a client."""
    return object.__new__(MilvusVectorStore)


class _ScoreOnlySearchClient:
    def has_collection(self, *, collection_name: str) -> bool:
        assert collection_name
        return True

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        assert kwargs["search_params"] == {"metric_type": "COSINE"}
        return [
            [
                {
                    "id": "vec_close",
                    "score": 0.9,
                    "entity": {"message_id": "msg_1"},
                },
                {
                    "id": "vec_far",
                    "score": 0.2,
                    "entity": {"message_id": "msg_2"},
                },
            ]
        ]


def test_collection_name_is_valid_stable_and_bounded() -> None:
    store = _helper_store()
    namespace = "honcho.doc." + ("workspace.with-dashes" * 30)

    first = store._collection_name(namespace)  # pyright: ignore[reportPrivateUsage]
    second = store._collection_name(namespace)  # pyright: ignore[reportPrivateUsage]

    assert first == second
    assert len(first) <= MAX_COLLECTION_NAME_LENGTH
    assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", first)


def test_filter_expression_escapes_and_combines_conditions() -> None:
    store = _helper_store()

    expression = store._build_filter_expression(  # pyright: ignore[reportPrivateUsage]
        {
            "session_name": 'quoted "session"',
            "level": {"in": ["explicit", "deductive"]},
            "peer_name": None,
        }
    )

    assert expression == (
        'session_name == "quoted \\"session\\""'
        ' and level in ["explicit", "deductive"]'
        " and peer_name is null"
    )


def test_filter_expression_rejects_invalid_keys() -> None:
    store = _helper_store()

    with pytest.raises(ValueError, match="Invalid filter key"):
        store._build_filter_expression(  # pyright: ignore[reportPrivateUsage]
            {"session-name": "unsafe"}
        )


def test_projection_settings_map_to_milvus_output_fields() -> None:
    store = _helper_store()

    assert store._output_fields(True) is None  # pyright: ignore[reportPrivateUsage]
    assert store._output_fields(False) == ["id"]  # pyright: ignore[reportPrivateUsage]
    assert store._output_fields(["id", "message_id"]) == [  # pyright: ignore[reportPrivateUsage]
        "message_id"
    ]


@pytest.mark.asyncio
async def test_query_converts_milvus_score_to_cosine_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 4)
    store = _helper_store()
    store.client = cast(Any, _ScoreOnlySearchClient())

    results = await store.query(
        "honcho.msg.test",
        [1.0, 0.0, 0.0, 0.0],
        max_distance=0.5,
    )

    assert [result.id for result in results] == ["vec_close"]
    assert abs(results[0].score - 0.1) < 1e-12
    assert results[0].metadata == {"message_id": "msg_1"}


@pytest.mark.asyncio
async def test_milvus_lite_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        settings.VECTOR_STORE, "MILVUS_URI", str(tmp_path / "milvus.db")
    )
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_TOKEN", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_DB_NAME", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_CONSISTENCY_LEVEL", "Strong")
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 4)

    store = MilvusVectorStore()
    namespace = "honcho.msg.workspace-with.invalid-chars"
    try:
        assert await store.query(namespace, [1.0, 0.0, 0.0, 0.0]) == []

        await store.upsert_many(
            namespace,
            [
                VectorRecord(
                    id="vec_1",
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    metadata={
                        "message_id": "msg_1",
                        "session_name": "session_a",
                        "peer_name": "alice",
                        "custom": "kept",
                    },
                ),
                VectorRecord(
                    id="vec_2",
                    embedding=[0.0, 1.0, 0.0, 0.0],
                    metadata={
                        "message_id": "msg_2",
                        "session_name": "session_b",
                        "peer_name": "bob",
                    },
                ),
            ],
        )

        assert await store.probe_namespace_dim(namespace) == 4

        all_results = await store.query(
            namespace,
            [1.0, 0.0, 0.0, 0.0],
            top_k=2,
        )
        assert [result.id for result in all_results] == ["vec_1", "vec_2"]
        assert all_results[0].score == 0.0
        assert all_results[0].metadata["message_id"] == "msg_1"
        assert all_results[0].metadata["custom"] == "kept"

        projected = await store.query(
            namespace,
            [1.0, 0.0, 0.0, 0.0],
            include_attributes=["message_id"],
        )
        assert projected[0].metadata == {"message_id": "msg_1"}

        id_only = await store.query(
            namespace,
            [1.0, 0.0, 0.0, 0.0],
            include_attributes=False,
        )
        assert id_only[0].metadata == {}

        filtered = await store.query(
            namespace,
            [1.0, 0.0, 0.0, 0.0],
            filters={"session_name": {"in": ["session_b"]}},
        )
        assert [result.id for result in filtered] == ["vec_2"]

        close_only = await store.query(
            namespace,
            [1.0, 0.0, 0.0, 0.0],
            max_distance=0.5,
        )
        assert [result.id for result in close_only] == ["vec_1"]

        await store.delete_many(namespace, ["vec_1"])
        after_delete = await store.query(
            namespace,
            [1.0, 0.0, 0.0, 0.0],
        )
        assert [result.id for result in after_delete] == ["vec_2"]

        await store.delete_namespace(namespace)
        assert await store.probe_namespace_dim(namespace) is None
        assert await store.query(namespace, [1.0, 0.0, 0.0, 0.0]) == []
    finally:
        await store.close()

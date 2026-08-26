"""Tests for MilvusVectorStore behavior."""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pytest
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from src.config import VectorStoreSettings, settings
from src.exceptions import VectorStoreError
from src.vector_store import VectorRecord, get_external_vector_store
from src.vector_store.milvus import (
    MAX_COLLECTION_NAME_LENGTH,
    STANDARD_METADATA_FIELDS,
    MilvusVectorStore,
)

logging.getLogger("milvus_lite.server_manager").disabled = True


def _helper_store() -> MilvusVectorStore:
    """Create a MilvusVectorStore instance without opening a client."""
    return object.__new__(MilvusVectorStore)


def test_default_consistency_level_is_session() -> None:
    consistency_field = VectorStoreSettings.model_fields["MILVUS_CONSISTENCY_LEVEL"]
    assert consistency_field.default == "Session"


def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def missing_milvus(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "src.vector_store.milvus":
            raise ImportError("No module named 'pymilvus'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", missing_milvus)
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "milvus")
    get_external_vector_store.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="uv sync --extra milvus"):
            get_external_vector_store()
    finally:
        get_external_vector_store.cache_clear()


def test_local_uri_requires_milvus_lite_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_lite(_: str) -> str:
        raise PackageNotFoundError("milvus-lite")

    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_URI", "./milvus.db")
    monkeypatch.setattr("src.vector_store.milvus.package_version", missing_lite)

    with pytest.raises(RuntimeError, match="uv sync --extra milvus-lite"):
        MilvusVectorStore()


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

    assert store._output_fields(True) == [  # pyright: ignore[reportPrivateUsage]
        "metadata"
    ]
    assert store._output_fields(False) == ["id"]  # pyright: ignore[reportPrivateUsage]
    assert store._output_fields(["id", "message_id"]) == [  # pyright: ignore[reportPrivateUsage]
        "message_id"
    ]


def test_cosine_distance_normalizes_real_milvus_distance_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def milvus_lite_version(_: str) -> str:
        return "3.0"

    store = _helper_store()

    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_URI", "./milvus.db")
    monkeypatch.setattr("src.vector_store.milvus.package_version", milvus_lite_version)
    assert store._hit_cosine_distance({"distance": 0.0}) == 0.0  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_URI", "http://localhost:19530")
    assert store._hit_cosine_distance({"distance": 1.0}) == 0.0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_milvus_lite_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        settings.VECTOR_STORE, "MILVUS_URI", str(tmp_path / "milvus.db")
    )
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_TOKEN", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_DB_NAME", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_CONSISTENCY_LEVEL", "Session")
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


def _create_reuse_collection(
    client: MilvusClient,
    collection_name: str,
    *,
    dimensions: int = 4,
    dynamic: bool = True,
    include_metadata: bool = True,
    metadata_type: DataType = DataType.JSON,
    omitted_field: str | None = None,
    metric_type: str = "COSINE",
    index_type: str = "AUTOINDEX",
) -> None:
    """Create a real Lite collection with a controlled schema defect."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=dynamic)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=65535,
    )
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimensions
    )
    if include_metadata:
        metadata_kwargs: dict[str, Any] = {
            "field_name": "metadata",
            "datatype": metadata_type,
            "nullable": True,
        }
        if metadata_type == DataType.VARCHAR:
            metadata_kwargs["max_length"] = 65535
        schema.add_field(**metadata_kwargs)
    for field_name in STANDARD_METADATA_FIELDS:
        if field_name == omitted_field:
            continue
        schema.add_field(
            field_name=field_name,
            datatype=DataType.VARCHAR,
            max_length=65535,
            nullable=True,
        )
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="vector", index_type=index_type, metric_type=metric_type
    )
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=indexes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema_kwargs", "message"),
    [
        ({"include_metadata": False}, "missing required JSON field 'metadata'"),
        ({"metadata_type": DataType.VARCHAR}, "field 'metadata' must be JSON"),
        ({"dynamic": False}, "must enable dynamic fields"),
        ({"omitted_field": "session_name"}, "missing required field 'session_name'"),
        ({"dimensions": 3}, r"vector dim \(3\).*VECTOR_DIMENSIONS \(4\)"),
        ({"metric_type": "L2"}, "vector index must use COSINE metric"),
        ({"index_type": "FLAT"}, "vector index must use AUTOINDEX"),
    ],
)
async def test_milvus_lite_reused_collection_schema_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    schema_kwargs: dict[str, Any],
    message: str,
) -> None:
    uri = str(tmp_path / "invalid-schema.db")
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_URI", uri)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_TOKEN", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_DB_NAME", None)
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 4)

    store = MilvusVectorStore()
    namespace = "honcho.msg.invalid-schema"
    collection_name = store._collection_name(  # pyright: ignore[reportPrivateUsage]
        namespace
    )
    _create_reuse_collection(store.client, collection_name, **schema_kwargs)
    try:
        with pytest.raises(VectorStoreError, match=message):
            await store.query(namespace, [1.0, 0.0, 0.0, 0.0])
    finally:
        await store.delete_namespace(namespace)
        assert not await store._has_collection(  # pyright: ignore[reportPrivateUsage]
            collection_name
        )
        await store.close()


@pytest.mark.asyncio
async def test_unreachable_client_error_allows_later_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "milvus")
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_URI", "http://127.0.0.1:1")
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_TOKEN", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_DB_NAME", None)
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 4)

    get_external_vector_store.cache_clear()
    with pytest.raises(VectorStoreError, match="initialize Milvus client") as error:
        get_external_vector_store()
    assert isinstance(error.value.__cause__, MilvusException)

    monkeypatch.setattr(
        settings.VECTOR_STORE, "MILVUS_URI", str(tmp_path / "recovered.db")
    )
    recovered = get_external_vector_store()
    assert isinstance(recovered, MilvusVectorStore)
    try:
        assert await recovered.query("recovered", [1.0, 0.0, 0.0, 0.0]) == []
    finally:
        await recovered.close()
        get_external_vector_store.cache_clear()


@pytest.mark.asyncio
async def test_client_call_preserves_milvus_cause_and_operation() -> None:
    store = _helper_store()

    def fail() -> None:
        raise MilvusException(message="unavailable")

    with pytest.raises(VectorStoreError, match="query namespace 'example'") as error:
        await store._run_client_call(  # pyright: ignore[reportPrivateUsage]
            "query namespace 'example'", fail
        )

    assert isinstance(error.value.__cause__, MilvusException)


@pytest.mark.asyncio
async def test_filter_value_error_is_not_wrapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        settings.VECTOR_STORE, "MILVUS_URI", str(tmp_path / "value-error.db")
    )
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_TOKEN", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_DB_NAME", None)
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 4)
    store = MilvusVectorStore()
    namespace = "value-error"
    try:
        await store.upsert_many(
            namespace,
            [VectorRecord(id="one", embedding=[1.0, 0.0, 0.0, 0.0])],
        )
        with pytest.raises(ValueError, match="Unsupported Milvus filter value"):
            await store.query(
                namespace,
                [1.0, 0.0, 0.0, 0.0],
                filters={"session_name": object()},
            )
    finally:
        await store.delete_namespace(namespace)
        await store.close()

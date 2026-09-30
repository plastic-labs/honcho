"""Per-store namespace dim probe tests.

LanceDB has an embedded driver we can spin up in a tmp dir, so we exercise
the real probe end-to-end. Turbopuffer needs a network + API key, so its
real probe runs against a faked namespace.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest

from src.config import settings
from src.exceptions import VectorStoreError
from src.vector_store.lancedb import LanceDBVectorStore
from src.vector_store.turbopuffer import TurbopufferVectorStore


@pytest.mark.asyncio
async def test_lancedb_probe_returns_declared_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Create a real LanceDB table at dim 768, confirm the probe recovers it."""
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.LANCEDB_PATH", str(tmp_path))

    store = LanceDBVectorStore()
    try:
        db = await store._get_db()  # pyright: ignore[reportPrivateUsage]
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 768)),
            ]
        )
        await db.create_table("probe_test", schema=schema)

        dim = await store.probe_namespace_dim("probe_test")
        assert dim == 768
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_lancedb_probe_returns_none_for_missing_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Lazy-create model: probing a nonexistent table is not an error."""
    monkeypatch.setattr("src.config.settings.VECTOR_STORE.LANCEDB_PATH", str(tmp_path))

    store = LanceDBVectorStore()
    try:
        dim = await store.probe_namespace_dim("does_not_exist")
        assert dim is None
    finally:
        await store.close()


def _turbopuffer_store(
    monkeypatch: pytest.MonkeyPatch, *, exists: bool, schema: dict[str, object]
) -> TurbopufferVectorStore:
    monkeypatch.setattr(settings.VECTOR_STORE, "TURBOPUFFER_API_KEY", "test-key")
    store = TurbopufferVectorStore()
    namespace = MagicMock()
    namespace.exists = AsyncMock(return_value=exists)
    namespace.schema = AsyncMock(return_value=schema)
    store._get_namespace = MagicMock(return_value=namespace)  # pyright: ignore[reportPrivateUsage]
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("type_str", "expected"),
    [("[768]f32", 768), ("[1536]f16", 1536), ("[256]i8", 256)],
)
async def test_turbopuffer_probe_parses_vector_dim(
    monkeypatch: pytest.MonkeyPatch, type_str: str, expected: int
) -> None:
    """Turbopuffer reports a vector column's type as ``[<dim>]<width>``. Lock
    the format through the real probe so an SDK change is loud."""
    store = _turbopuffer_store(
        monkeypatch, exists=True, schema={"vector": SimpleNamespace(type=type_str)}
    )
    assert await store.probe_namespace_dim("ns") == expected


@pytest.mark.asyncio
async def test_turbopuffer_probe_returns_none_for_missing_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _turbopuffer_store(monkeypatch, exists=False, schema={})
    assert await store.probe_namespace_dim("ns") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema",
    [{}, {"vector": SimpleNamespace(type="string")}],
    ids=["no-vector-attribute", "unparseable-type"],
)
async def test_turbopuffer_probe_rejects_invalid_schema(
    monkeypatch: pytest.MonkeyPatch, schema: dict[str, object]
) -> None:
    """An existing namespace with an unusable schema must fail loudly, not be
    bucketed as missing."""
    store = _turbopuffer_store(monkeypatch, exists=True, schema=schema)
    with pytest.raises(VectorStoreError):
        await store.probe_namespace_dim("ns")

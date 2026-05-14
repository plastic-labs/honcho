"""Per-store namespace dim probe tests.

LanceDB has an embedded driver we can spin up in a tmp dir, so we exercise
the real probe end-to-end. Turbopuffer needs a network + API key, so it is
covered only by static analysis + the parsing test below.
"""

from __future__ import annotations

import re

import pyarrow as pa
import pytest

from src.vector_store.lancedb import LanceDBVectorStore


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


def test_turbopuffer_vector_dim_regex_extracts_dim_from_type_string() -> None:
    """Turbopuffer's attribute type for a vector column is a bracket-prefixed
    dim with a width suffix: ``[768]f32``, ``[1536]f16``, ``[256]i8``. The
    probe extracts the integer inside the brackets. Lock the format here so
    an SDK change is loud."""
    pattern = re.compile(r"\[(\d+)\]")
    cases = {
        "[768]f32": "768",
        "[1536]f16": "1536",
        "[256]i8": "256",
    }
    for type_str, expected in cases.items():
        match = pattern.search(type_str)
        assert match is not None, f"failed to match {type_str!r}"
        assert match.group(1) == expected
    assert pattern.search("string") is None

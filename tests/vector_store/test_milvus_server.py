"""Opt-in lifecycle verification against a real Milvus Server."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.config import settings
from src.vector_store import VectorRecord
from src.vector_store.milvus import MilvusVectorStore

SERVER_URI = os.getenv("HONCHO_TEST_MILVUS_URI")

pytestmark = [
    pytest.mark.milvus_server,
    pytest.mark.skipif(
        not SERVER_URI,
        reason="HONCHO_TEST_MILVUS_URI is required for Milvus Server E2E",
    ),
]


@pytest.mark.asyncio
async def test_worker_write_api_read_and_client_restart_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Session visibility and persistence across Honcho client restarts."""
    assert SERVER_URI is not None
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_URI", SERVER_URI)
    monkeypatch.setattr(
        settings.VECTOR_STORE,
        "MILVUS_TOKEN",
        os.getenv("HONCHO_TEST_MILVUS_TOKEN"),
    )
    monkeypatch.setattr(
        settings.VECTOR_STORE,
        "MILVUS_DB_NAME",
        os.getenv("HONCHO_TEST_MILVUS_DB_NAME"),
    )
    monkeypatch.setattr(settings.VECTOR_STORE, "MILVUS_CONSISTENCY_LEVEL", "Session")
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 4)

    namespace = f"honcho.e2e.lifecycle.{uuid4().hex}"
    collection_name: str | None = None
    worker = MilvusVectorStore()
    api = MilvusVectorStore()
    try:
        collection_name = worker._collection_name(  # pyright: ignore[reportPrivateUsage]
            namespace
        )
        await worker.upsert_many(
            namespace,
            [
                VectorRecord(
                    id="worker-write",
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    metadata={"session_name": "session-visible", "source": "worker"},
                )
            ],
        )
        immediate = await api.query(namespace, [1.0, 0.0, 0.0, 0.0])
        assert [result.id for result in immediate] == ["worker-write"]
        assert immediate[0].metadata["session_name"] == "session-visible"
    finally:
        await worker.close()
        await api.close()

    restarted_worker = MilvusVectorStore()
    restarted_api = MilvusVectorStore()
    try:
        persisted = await restarted_api.query(namespace, [1.0, 0.0, 0.0, 0.0])
        assert [result.id for result in persisted] == ["worker-write"]
        assert persisted[0].metadata["source"] == "worker"
        await restarted_worker.delete_namespace(namespace)
        assert collection_name is not None
        assert not await restarted_api._has_collection(  # pyright: ignore[reportPrivateUsage]
            collection_name
        )
    finally:
        if collection_name and await restarted_worker._has_collection(  # pyright: ignore[reportPrivateUsage]
            collection_name
        ):
            await restarted_worker.delete_namespace(namespace)
        await restarted_worker.close()
        await restarted_api.close()

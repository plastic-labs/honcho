"""Tests for TurbopufferVectorStore error handling on 5xx responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from turbopuffer import InternalServerError

from src.config import settings
from src.exceptions import VectorStoreError
from src.vector_store import VectorRecord
from src.vector_store.turbopuffer import TurbopufferVectorStore


def _internal_server_error(status_code: int = 503) -> InternalServerError:
    request = httpx.Request(
        "POST", "https://api.turbopuffer.com/v2/namespaces/ns/write"
    )
    response = httpx.Response(status_code, request=request)
    return InternalServerError("turbopuffer unavailable", response=response, body=None)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> TurbopufferVectorStore:
    monkeypatch.setattr(settings.VECTOR_STORE, "TURBOPUFFER_API_KEY", "test-key")
    monkeypatch.setattr(settings.VECTOR_STORE, "TURBOPUFFER_REGION", "gcp-us-east4")
    return TurbopufferVectorStore()


@pytest.fixture
def record() -> VectorRecord:
    return VectorRecord(
        id="doc_1", embedding=[0.1, 0.2, 0.3, 0.4], metadata={"foo": "bar"}
    )


@pytest.mark.asyncio
async def test_upsert_many_raises_vector_store_error_on_5xx(
    store: TurbopufferVectorStore,
    record: VectorRecord,
) -> None:
    namespace_mock = MagicMock()
    namespace_mock.write = AsyncMock(side_effect=_internal_server_error(503))
    store._get_namespace = MagicMock(return_value=namespace_mock)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(VectorStoreError) as excinfo:
        await store.upsert_many("honcho.doc.test", [record])

    assert "honcho.doc.test" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, InternalServerError)
    namespace_mock.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_many_short_circuits_on_empty(
    store: TurbopufferVectorStore,
) -> None:
    namespace_mock = MagicMock()
    namespace_mock.write = AsyncMock()
    store._get_namespace = MagicMock(return_value=namespace_mock)  # pyright: ignore[reportPrivateUsage]

    await store.upsert_many("honcho.doc.test", [])

    namespace_mock.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_many_succeeds_without_raising(
    store: TurbopufferVectorStore,
    record: VectorRecord,
) -> None:
    namespace_mock = MagicMock()
    namespace_mock.write = AsyncMock()
    store._get_namespace = MagicMock(return_value=namespace_mock)  # pyright: ignore[reportPrivateUsage]

    result = await store.upsert_many("honcho.doc.test", [record])

    assert result is None
    namespace_mock.write.assert_awaited_once()

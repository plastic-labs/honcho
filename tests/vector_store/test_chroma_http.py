"""Exercise the real HTTP/Cloud SDK and error decoder over a mock HTTP transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from src.config import settings
from src.exceptions import VectorStoreError
from src.vector_store import VectorRecord

pytest.importorskip("chromadb")
from chromadb.errors import AuthorizationError, InvalidArgumentError  # noqa: E402

from src.vector_store.chroma import ChromaVectorStore  # noqa: E402


@dataclass
class ChromaHTTP:
    requests: list[httpx.Request] = field(default_factory=list)
    status: int = 200
    error: str | None = None
    fail_identity: bool = False
    fail_operations_only: bool = False
    disconnect: bool = False

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if self.disconnect:
            raise httpx.ConnectError("offline", request=request)
        if self.status != 200 and (
            self.fail_identity
            or (
                "/collections" in path
                and (
                    not self.fail_operations_only
                    or request.method == "DELETE"
                    or path.endswith(("/query", "/upsert", "/delete"))
                )
            )
        ):
            if self.error:
                return httpx.Response(
                    self.status,
                    json={"error": self.error, "message": "test server failure"},
                    request=request,
                )
            return httpx.Response(
                self.status, text="upstream unavailable", request=request
            )

        payload: Any
        if path.endswith("/auth/identity"):
            payload = {
                "user_id": "test",
                "tenant": "default_tenant",
                "databases": ["default_database"],
            }
        elif path.endswith("/tenants/default_tenant"):
            payload = {"name": "default_tenant"}
        elif path.endswith("/databases/default_database"):
            payload = {
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "default_database",
                "tenant": "default_tenant",
            }
        elif path.endswith("/pre-flight-checks"):
            payload = {"max_batch_size": 2}
        elif path.endswith("/query"):
            payload = {
                "ids": [["a"]],
                "distances": [[0.1]],
                "metadatas": [[{"level": "explicit"}]],
            }
        elif path.endswith(("/upsert", "/delete")) or request.method == "DELETE":
            payload = {}
        elif "/collections" in path:
            name = (
                json.loads(request.content)["name"]
                if request.method == "POST"
                else path.rsplit("/", 1)[1]
            )
            payload = {
                "id": "00000000-0000-0000-0000-000000000002",
                "name": name,
                "configuration_json": {
                    "hnsw": {"space": "cosine"},
                    "embedding_function": {"type": "legacy"},
                },
                "metadata": {},
                "dimension": 4,
                "tenant": "default_tenant",
                "database": "default_database",
            }
        else:
            raise AssertionError(f"Unexpected Chroma request: {request.method} {path}")
        return httpx.Response(200, json=payload, request=request)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> ChromaHTTP:
    server = ChromaHTTP()
    original_init = httpx.Client.__init__

    def init(client: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(server.respond)
        original_init(client, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", init)
    return server


@pytest.fixture(params=["http", "cloud"])
async def remote_store(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    transport: ChromaHTTP,
) -> AsyncIterator[ChromaVectorStore]:
    del transport  # Installs the transport before the real SDK constructs its client.
    monkeypatch.setattr(settings.VECTOR_STORE, "CHROMA_CLIENT_MODE", request.param)
    monkeypatch.setattr(settings.VECTOR_STORE, "CHROMA_API_KEY", "test-key")
    monkeypatch.setattr(settings.VECTOR_STORE, "CHROMA_TENANT", "default_tenant")
    monkeypatch.setattr(settings.VECTOR_STORE, "CHROMA_DATABASE", "default_database")
    store = ChromaVectorStore()
    try:
        yield store
    finally:
        await store.close()


async def test_remote_sdk_requests(
    remote_store: ChromaVectorStore, transport: ChromaHTTP
) -> None:
    records = [VectorRecord(id=str(i), embedding=[0.1] * 4) for i in range(3)]
    await remote_store.upsert_many("remote", records)
    assert [
        len(json.loads(r.content)["ids"])
        for r in transport.requests
        if r.url.path.endswith("/upsert")
    ] == [2, 1]
    results = await remote_store.query(
        "remote", [0.1] * 4, filters={"level": ["explicit"]}
    )
    assert [(r.id, r.score, r.metadata) for r in results] == [
        ("a", 0.1, {"level": "explicit"})
    ]
    query = next(r for r in transport.requests if r.url.path.endswith("/query"))
    assert json.loads(query.content)["where"] == {"level": {"$in": ["explicit"]}}
    assert await remote_store.probe_namespace_dim("remote") == 4
    await remote_store.delete_many("remote", ["a"])
    await remote_store.delete_namespace("remote")
    if settings.VECTOR_STORE.CHROMA_CLIENT_MODE == "cloud":
        assert query.headers["x-chroma-token"] == "test-key"


@pytest.mark.parametrize(
    "status,error", [(500, "InternalError"), (429, "RateLimitError"), (503, None)]
)
@pytest.mark.parametrize("stage", ["identity", "lookup", "operation"])
async def test_remote_transient_errors(
    remote_store: ChromaVectorStore,
    transport: ChromaHTTP,
    status: int,
    error: str | None,
    stage: str,
) -> None:
    transport.status = status
    transport.error = error
    transport.fail_identity = stage == "identity"
    transport.fail_operations_only = stage == "operation"
    assert await remote_store.query("remote", [0.1] * 4) == []
    with pytest.raises(VectorStoreError):
        await remote_store.upsert_many(
            "remote", [VectorRecord(id="a", embedding=[0.1] * 4)]
        )
    with pytest.raises(VectorStoreError):
        await remote_store.delete_many("remote", ["a"])
    with pytest.raises(VectorStoreError):
        await remote_store.delete_namespace("remote")


async def test_remote_connection_failure_recovers(
    remote_store: ChromaVectorStore, transport: ChromaHTTP
) -> None:
    transport.disconnect = True
    assert await remote_store.query("remote", [0.1] * 4) == []
    with pytest.raises(VectorStoreError):
        await remote_store.upsert_many(
            "remote", [VectorRecord(id="a", embedding=[0.1] * 4)]
        )
    transport.disconnect = False
    assert [r.id for r in await remote_store.query("remote", [0.1] * 4)] == ["a"]


@pytest.mark.parametrize(
    "status,error,expected",
    [
        (403, "AuthorizationError", AuthorizationError),
        (400, "InvalidArgumentError", InvalidArgumentError),
        (400, None, Exception),
    ],
)
async def test_remote_nontransient_errors_remain_visible(
    remote_store: ChromaVectorStore,
    transport: ChromaHTTP,
    status: int,
    error: str | None,
    expected: type[Exception],
) -> None:
    transport.status = status
    transport.error = error
    with pytest.raises(expected) as raised:
        await remote_store.query("remote", [0.1] * 4)
    assert not isinstance(raised.value, VectorStoreError)

"""Guards the default-node connection release in src.cache.client.

The startup PING is keyless, so redis-py routes it to the cluster's default
node instead of to a shard. That node is identical for every client in a
deployment -- `NodesManager.initialize` fills `nodes_cache` in CLUSTER SLOTS
order, which Redis returns sorted by slot since 6.2, then takes
`get_nodes_by_server_type(PRIMARY)[0]` -- so every process leaves one pooled,
permanently idle connection on whichever primary owns slot 0. redis-py's
cluster node pool does no idle reaping, so nothing closes it.

`test_default_node_is_still_the_first_primary` is the signal for when this can
be deleted: it fails once redis-py stops pinning the default node to the first
primary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, final

import pytest
from redis.asyncio import cluster as redis_cluster
from redis.asyncio.cluster import ClusterNode

from src.cache.client import (
    _release_default_node_connection,  # pyright: ignore[reportPrivateUsage]
    cache,
)
from src.config import settings


@final
class _FakeNode:
    disconnect_calls: int

    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect_free_connections(self) -> None:
        self.disconnect_calls += 1


@final
class _FakeBackend:
    """Mirrors the cashews backend -> redis-py client -> nodes_manager chain."""

    _client: Any

    def __init__(self, node: Any) -> None:
        self._client = type(
            "_Client", (), {"nodes_manager": type("_NM", (), {"default_node": node})()}
        )()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeNode:
    node = _FakeNode()
    monkeypatch.setattr(cache, "_backends", {"": _FakeBackend(node)})
    return node


@pytest.mark.asyncio
async def test_releases_the_idle_connection(
    fake_backend: _FakeNode, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.CACHE, "CLUSTER", True)
    await _release_default_node_connection()
    assert fake_backend.disconnect_calls == 1


@pytest.mark.asyncio
async def test_no_op_when_not_clustered(
    fake_backend: _FakeNode, monkeypatch: pytest.MonkeyPatch
):
    """A standalone client has no default node and nothing to release."""
    monkeypatch.setattr(settings.CACHE, "CLUSTER", False)
    await _release_default_node_connection()
    assert fake_backend.disconnect_calls == 0


@pytest.mark.asyncio
async def test_missing_internals_do_not_break_startup(monkeypatch: pytest.MonkeyPatch):
    """The chain is private in both libraries, so a rename must degrade quietly.

    The connection released here is an optimisation; failing startup over it
    would trade a connection-count problem for an outage.
    """
    monkeypatch.setattr(settings.CACHE, "CLUSTER", True)
    monkeypatch.setattr(cache, "_backends", {"": object()})
    await _release_default_node_connection()  # must not raise


def test_disconnect_free_connections_still_exists():
    """The redis-py API this relies on, which is what makes the release lazy."""
    assert hasattr(ClusterNode, "disconnect_free_connections")


def test_default_node_is_still_the_first_primary():
    """Fails once redis-py stops pinning every client to the same node.

    If upstream randomises or otherwise spreads the default-node choice, the
    connections this releases no longer pile onto one primary and the release
    can be dropped.
    """
    source = Path(redis_cluster.__file__).read_text(encoding="utf-8")
    assert "self.default_node = self.get_nodes_by_server_type(PRIMARY)[0]" in source, (
        "redis-py no longer pins the default node to the first primary. The "
        "_release_default_node_connection call in src/cache/client.py is "
        "likely obsolete -- verify against the installed redis-py and drop it."
    )

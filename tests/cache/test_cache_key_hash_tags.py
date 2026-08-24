"""Hash tags keep one workspace's cache keys on one Redis Cluster shard.

Without them a pod opens a connection to every shard, because its keys hash
all over the keyspace. The fleet was running ~4.45 Redis connections per pod
against a 3-shard cluster for that reason.
"""

import pytest
from redis.crc import key_slot

from src.crud.collection import COLLECTION_CACHE_KEY_TEMPLATE, collection_cache_key
from src.crud.peer import PEER_CACHE_KEY_TEMPLATE, peer_cache_key
from src.crud.session import SESSION_CACHE_KEY_TEMPLATE, session_cache_key
from src.crud.workspace import WORKSPACE_CACHE_KEY_TEMPLATE, workspace_cache_key

WS = "ws-test"


def _rendered() -> list[str]:
    return [
        workspace_cache_key(WS),
        session_cache_key(WS, "sess-1"),
        session_cache_key(WS, "sess-2"),
        peer_cache_key(WS, "peer-1"),
        collection_cache_key(WS, "obs-a", "obs-b"),
    ]


def test_every_template_tags_the_workspace() -> None:
    for tpl in (
        WORKSPACE_CACHE_KEY_TEMPLATE,
        SESSION_CACHE_KEY_TEMPLATE,
        PEER_CACHE_KEY_TEMPLATE,
        COLLECTION_CACHE_KEY_TEMPLATE,
    ):
        assert "{{{workspace_name}}}" in tpl, tpl


def test_one_workspace_is_one_slot() -> None:
    slots = {key_slot(k.encode()) for k in _rendered()}
    assert len(slots) == 1, f"expected one slot, got {slots} for {_rendered()}"


def test_only_the_workspace_is_tagged() -> None:
    # Exactly one brace pair per key. A second would make the tag the text
    # between the first { and the first }, which is not what we want.
    for k in _rendered():
        assert k.count("{") == 1 and k.count("}") == 1, k


@pytest.mark.parametrize("n", [300])
def test_workspaces_still_spread_across_shards(n: int) -> None:
    shards = {key_slot(workspace_cache_key(f"ws-{i}").encode()) % 3 for i in range(n)}
    assert shards == {0, 1, 2}, shards

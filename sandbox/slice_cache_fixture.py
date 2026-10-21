"""Populate four pre-delete cache entries using Honcho's actual fetch/serializer.

Executed on stdin inside the prototype API container by slice_runner.py. This is
an injected sizing fixture, not proof that a dream/summary producer ran. It must
never run after deletion: the comparison preserves these exact bytes throughout.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys

from src.cache.client import cache, close_cache, init_cache
from src.crud.peer import _fetch_peer, peer_cache_key
from src.crud.peer_card import construct_peer_card_label
from src.crud.session import _fetch_session, session_cache_key
from src.dependencies import tracked_db


async def main() -> None:
    workspace, session, canary = sys.argv[1:]
    entries = {}
    # Bypass the caching decorators, not the real fetch/serialization contract.
    # Otherwise earlier get-or-create entries may still contain empty metadata.
    async with tracked_db("slice-cache-fixture", read_only=True) as db:
        for observer in ("alice", "bob", "carol"):
            data = await inspect.unwrap(_fetch_peer)(db, workspace, observer)
            label = construct_peer_card_label(observer=observer, observed="alice")
            assert data and canary in json.dumps(data["internal_metadata"][label])
            entries[peer_cache_key(workspace, observer)] = data
        data = await inspect.unwrap(_fetch_session)(db, workspace, session)
        assert data
        for tier in ("short", "long"):
            summary = data["internal_metadata"]["summaries"][
                f"honcho_chat_summary_{tier}"
            ]
            assert canary in summary["content"]
        entries[session_cache_key(workspace, session)] = data

    await init_cache()
    try:
        for key, data in entries.items():
            await cache.set(key, data, expire=3600)
            assert await cache.get(key) == data
    finally:
        await close_cache()
    # The host independently reads Redis, removes only these fixture TTLs, and
    # requires exact payload hashes to survive AOF restart and both image reads.
    print("SLICE_CACHE_KEYS=" + json.dumps(sorted(entries)))


if __name__ == "__main__":
    asyncio.run(main())

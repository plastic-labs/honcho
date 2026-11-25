"""Regression tests for bounded growth of the Honcho local session cache.

Covers the fix for unbounded RSS growth in long-running gateways: prior to
this, ``HonchoSession.messages`` grew forever (never trimmed after a sync),
and ``HonchoSessionManager``'s ``_cache``/``_sessions_cache``
had no eviction path short of an explicit ``/new`` reset.
"""

import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace


from hermes_plugin_honcho.session import (
    HonchoSession,
    HonchoSessionManager,
    _PEERS_CACHE_MAX_SIZE,
    _SESSION_CACHE_MAX_SIZE,
    _SESSION_IDLE_TTL_SECONDS,
    _SESSION_MESSAGE_RETENTION,
)


def _session(key="k", honcho_session_id=None):
    return HonchoSession(
        key=key,
        user_peer_id="user",
        assistant_peer_id="assistant",
        honcho_session_id=honcho_session_id or f"hs-{key}",
    )


def _manager():
    return HonchoSessionManager(honcho=SimpleNamespace(), config=SimpleNamespace())


def test_trim_synced_messages_caps_total_length():
    session = _session()
    for i in range(250):
        session.add_message("user", f"msg{i}", _synced=True)
    for i in range(250, 255):
        session.add_message("user", f"msg{i}", _synced=False)
    assert len(session.messages) == 255

    HonchoSessionManager._trim_synced_messages(session)

    assert len(session.messages) == _SESSION_MESSAGE_RETENTION
    # oldest synced messages are the ones dropped; the unsynced tail survives intact
    assert not any(m.get("_synced") is False for m in session.messages[:-5])
    assert all(m.get("_synced") is False for m in session.messages[-5:])


def test_trim_synced_messages_never_drops_an_unsynced_message():
    session = _session()
    for i in range(300):
        session.add_message("user", f"m{i}", _synced=(i != 10))

    HonchoSessionManager._trim_synced_messages(session)

    contents = [m["content"] for m in session.messages]
    assert "m10" in contents
    idx = contents.index("m10")
    assert session.messages[idx].get("_synced") is False
    # trimming stops at the first unsynced message from the front — it does
    # not skip past it to keep reducing, so everything from there on survives
    assert contents[idx:] == [f"m{i}" for i in range(10, 300)]


def test_trim_synced_messages_is_a_noop_under_the_cap():
    session = _session()
    for i in range(10):
        session.add_message("user", f"m{i}", _synced=True)

    HonchoSessionManager._trim_synced_messages(session)

    assert len(session.messages) == 10


def test_sweep_idle_sessions_evicts_stale_entries_across_all_caches():
    mgr = _manager()
    stale = _session(key="stale", honcho_session_id="hs-stale")
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    fresh = _session(key="fresh", honcho_session_id="hs-fresh")

    mgr._cache = {"stale": stale, "fresh": fresh}
    mgr._sessions_cache = {"hs-stale": object(), "hs-fresh": object()}

    evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 1
    assert set(mgr._cache) == {"fresh"}
    assert set(mgr._sessions_cache) == {"hs-fresh"}


def test_sweep_idle_sessions_keeps_fresh_entries():
    mgr = _manager()
    fresh = _session(key="fresh")
    mgr._cache = {"fresh": fresh}
    mgr._sessions_cache = {fresh.honcho_session_id: object()}

    evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 0
    assert "fresh" in mgr._cache


def test_maybe_sweep_idle_sessions_is_rate_limited():
    mgr = _manager()
    stale = _session(key="stale")
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"stale": stale}

    mgr._last_idle_sweep_ts = time.time()  # just swept — this call should no-op
    mgr._maybe_sweep_idle_sessions()
    assert "stale" in mgr._cache

    mgr._last_idle_sweep_ts = 0.0  # force the interval to have elapsed
    mgr._maybe_sweep_idle_sessions()
    assert "stale" not in mgr._cache


def test_get_or_create_triggers_sweep_without_blocking_on_lock_reentrancy():
    """`_cache_lock` is an RLock specifically so a sweep triggered from inside
    `get_or_create` (which also takes the lock) can't deadlock the manager's
    own thread. Guard against that regressing silently.
    """
    mgr = _manager()
    stale = _session(key="stale")
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"stale": stale}
    mgr._last_idle_sweep_ts = 0.0

    done = threading.Event()

    def call_it():
        mgr._maybe_sweep_idle_sessions()
        done.set()

    t = threading.Thread(target=call_it)
    t.start()
    t.join(timeout=5)
    assert done.is_set(), "sweep did not complete — possible deadlock"
    assert "stale" not in mgr._cache


# ---------------------------------------------------------------------------
# hard caps, unsynced buffers, peers, and read activity (follows #71463)
# ---------------------------------------------------------------------------



def _fill_sessions(mgr, count, unsynced_keys=()):
    for i in range(count):
        key = f"k{i}"
        session = _session(key=key)
        if key in unsynced_keys:
            session.add_message("user", "pending", _synced=False)
        mgr._cache[key] = session
        mgr._sessions_cache[session.honcho_session_id] = object()


def test_size_cap_evicts_least_recently_used_sessions_with_their_entries():
    mgr = _manager()
    _fill_sessions(mgr, _SESSION_CACHE_MAX_SIZE + 2)

    with mgr._cache_lock:
        mgr._enforce_cache_caps_locked()

    assert len(mgr._cache) == _SESSION_CACHE_MAX_SIZE
    assert "k0" not in mgr._cache and "k1" not in mgr._cache
    assert "k2" in mgr._cache
    for gone in ("k0", "k1"):
        assert f"hs-{gone}" not in mgr._sessions_cache
    assert "hs-k2" in mgr._sessions_cache


def test_size_cap_never_evicts_a_session_with_unsynced_messages():
    mgr = _manager()
    _fill_sessions(mgr, _SESSION_CACHE_MAX_SIZE + 1, unsynced_keys={"k0"})

    with mgr._cache_lock:
        mgr._enforce_cache_caps_locked()

    assert "k0" in mgr._cache  # the only copy until its flush lands
    assert "k1" not in mgr._cache
    assert len(mgr._cache) == _SESSION_CACHE_MAX_SIZE


def test_idle_sweep_keeps_sessions_with_unsynced_messages():
    mgr = _manager()
    stale = _session(key="stale")
    stale.add_message("user", "pending", _synced=False)
    stale.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"stale": stale}

    with mgr._cache_lock:
        evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 0
    assert "stale" in mgr._cache


def test_peers_cap_evicts_unreferenced_peers_oldest_first():
    mgr = _manager()
    live = _session(key="live")
    mgr._cache = {"live": live}
    mgr._peers_cache[live.user_peer_id] = object()
    mgr._peers_cache[live.assistant_peer_id] = object()
    for i in range(_PEERS_CACHE_MAX_SIZE):
        mgr._peers_cache[f"guest{i}"] = object()

    with mgr._cache_lock:
        mgr._enforce_cache_caps_locked()

    assert len(mgr._peers_cache) == _PEERS_CACHE_MAX_SIZE
    assert live.user_peer_id in mgr._peers_cache and live.assistant_peer_id in mgr._peers_cache
    assert "guest0" not in mgr._peers_cache and "guest1" not in mgr._peers_cache
    assert f"guest{_PEERS_CACHE_MAX_SIZE - 1}" in mgr._peers_cache


def test_recall_read_counts_as_activity_for_the_idle_sweep():
    mgr = _manager()
    session = _session(key="read-only")
    session.updated_at = datetime.now() - timedelta(seconds=_SESSION_IDLE_TTL_SECONDS + 1)
    mgr._cache = {"read-only": session}

    assert mgr._cached_session("read-only") is session
    with mgr._cache_lock:
        evicted = mgr._sweep_idle_sessions_locked()

    assert evicted == 0
    assert "read-only" in mgr._cache


def test_sdk_object_hit_moves_the_key_to_the_recent_end():
    mgr = _manager()
    mgr._peers_cache = {"a": object(), "b": object()}

    mgr._cached_sdk_object(mgr._peers_cache, "a", lambda: None)

    assert list(mgr._peers_cache) == ["b", "a"]


def test_flush_does_not_resurrect_an_evicted_session():
    mgr = _manager()
    session = _session(key="gone")
    session.add_message("user", "late", _synced=False)
    peer = SimpleNamespace(message=lambda content: content)
    mgr._get_or_create_peer = lambda peer_id: peer
    mgr._sessions_cache[session.honcho_session_id] = SimpleNamespace(add_messages=lambda messages: None)

    assert mgr._flush_session(session) is True
    assert "gone" not in mgr._cache
    assert all(m["_synced"] for m in session.messages)


def _sdk_session():
    return SimpleNamespace(add_messages=lambda messages: None)


def test_cached_sdk_session_is_returned_without_setup():
    mgr = _manager()
    sdk = _sdk_session()
    mgr._sessions_cache["hs-x"] = sdk

    assert mgr._get_or_create_honcho_session("hs-x", None, None) is sdk


def _failing_then_recording_uploads(mgr, session):
    """The SDK session for ``session`` refuses the first batch; ``restore()`` swaps in one that records."""
    uploads = []

    def refuse(messages):
        raise ConnectionError("upload refused")

    mgr._get_or_create_peer = lambda peer_id: SimpleNamespace(message=lambda content: content)
    mgr._sessions_cache[session.honcho_session_id] = SimpleNamespace(add_messages=refuse)

    def restore():
        mgr._sessions_cache[session.honcho_session_id] = SimpleNamespace(add_messages=lambda ms: uploads.extend(ms))

    return uploads, restore


def test_failed_turn_save_after_an_eviction_is_retried_by_flush_all():
    """A clean session can be evicted while its caller still holds it. A save once shutdown began flushes inline,
    and a failed upload used to leave that object nowhere flush_all() could find it."""
    mgr = _manager()
    mgr._shutting_down = True
    session = _session(key="k")
    mgr._cache["k"] = session
    with mgr._cache_lock:
        mgr._evict_session_locked("k", session)
    session.add_message("user", "late")
    uploads, restore = _failing_then_recording_uploads(mgr, session)

    mgr.save(session)
    assert mgr._cache["k"] is session
    restore()
    mgr.flush_all()

    assert uploads == ["late"]
    assert session.messages[0]["_synced"] is True


def test_failed_collision_flush_waits_for_flush_all_without_displacing_the_newer_object():
    mgr = _manager()
    mgr._shutting_down = True
    newer = _session(key="k")
    mgr._cache["k"] = newer
    stale = _session(key="k")
    stale.add_message("user", "late")
    uploads, restore = _failing_then_recording_uploads(mgr, stale)

    mgr.save(stale)
    assert mgr._cache["k"] is newer
    assert mgr._retry_sessions == [stale]
    restore()
    mgr.flush_all()

    assert uploads == ["late"]
    assert mgr._retry_sessions == []
    assert mgr._cache["k"] is newer


def test_a_retained_session_is_listed_once_across_repeated_failures():
    mgr = _manager()
    mgr._shutting_down = True
    mgr._cache["k"] = _session(key="k")
    stale = _session(key="k")
    stale.add_message("user", "late")
    _failing_then_recording_uploads(mgr, stale)

    mgr.save(stale)
    mgr.save(stale)

    assert mgr._retry_sessions == [stale]

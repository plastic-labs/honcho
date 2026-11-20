"""Honcho-based session management for conversation history."""

from __future__ import annotations

import queue
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .client import spawn_context_thread
from .client import get_honcho_client
from .session_auth import HonchoAuthError, SessionAuthMixin
from .session_context import SessionContextMixin
from .session_migration import SessionMigrationMixin
from .session_peers import SessionPeersMixin, observe_self

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

# Sentinel to signal the async writer thread to shut down
_ASYNC_SHUTDOWN = object()

# Honcho persists every message and get_or_create() re-hydrates on a miss, so the local copies stay bounded.
_SESSION_MESSAGE_RETENTION = 200
_SESSION_IDLE_TTL_SECONDS = 3600
_SESSION_SWEEP_INTERVAL_SECONDS = 300
# Hard caps for a burst of distinct sessions inside one TTL window; the dicts evict least recently used,
# and _joined_author_peers drops its oldest session so its authors rejoin on their next write.
_SESSION_CACHE_MAX_SIZE = 128
_PEERS_CACHE_MAX_SIZE = 512


@dataclass
class HonchoSession:
    """A conversation session backed by Honcho: a local message cache that syncs to Honcho."""

    key: str  # channel:chat_id
    user_peer_id: str  # Honcho peer ID for the user
    assistant_peer_id: str  # Honcho peer ID for the assistant
    honcho_session_id: str  # Honcho session ID
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Held by _flush_session across select, send and the _synced flip. It lives with the message list it guards.
    _flush_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs})
        self.updated_at = datetime.now()


class HonchoSessionManager(SessionAuthMixin, SessionPeersMixin, SessionContextMixin, SessionMigrationMixin):
    """Conversation sessions backed by Honcho, alongside hermes' SQLite state and file memory.
    Auth retry, peer-ID resolution, recall and memory-file migration live in the mixins."""

    def __init__(
        self, honcho: Honcho | None = None, config: Any | None = None,
        runtime_user_peer_name: str | None = None, runtime_user_peer_name_alt: str | None = None,
        scopes: list[str] | None = None,
    ):
        """``honcho`` defaults to the per-identity cached client; the runtime peer names are the gateway
        user identity (and a stable alternate) for per-user memory scoping; ``scopes`` are the scopes a
        session this manager creates joins (its repository and machine)."""
        self._honcho = honcho
        self._config = config
        self._scopes = list(scopes or [])
        self._runtime_user_peer_name = runtime_user_peer_name
        self._runtime_user_peer_name_alt = runtime_user_peer_name_alt
        self._cache: dict[str, HonchoSession] = {}
        # Sessions whose flush failed after a newer object took their cache key; flush_all() retries them.
        self._retry_sessions: list[HonchoSession] = []
        self._cache_lock = threading.RLock()
        self._peers_cache: dict[str, Any] = {}
        # honcho_session_id -> author peer IDs already joined to that session.
        self._joined_author_peers: dict[str, set[str]] = {}
        self._sessions_cache: dict[str, Any] = {}
        self._last_idle_sweep_ts: float = 0.0
        # Bumped (under _cache_lock) whenever _force_reauth rebuilds the client, so an
        # in-flight resolver never stores an object bound to the discarded client.
        self._client_generation = 0

        # Set when a call still fails auth after a forced token refresh; cleared on the next success.
        self._auth_failure: str | None = None
        self._auth_notice_emitted = False

        # Async write queue — the writer thread starts lazily on first enqueue
        # (_ensure_async_writer_locked): constructing a manager must not spawn background
        # work or touch the network (unit tests build managers with mocked clients).
        self._async_queue: queue.Queue = queue.Queue()
        self._async_thread: threading.Thread | None = None
        self._async_thread_lock = threading.Lock()
        # Set by shutdown: no new threads after the join, late saves flush inline.
        self._shutting_down = False

    @property
    def honcho(self) -> Honcho:
        """The Honcho client, refreshing a near-expiry OAuth token in place. Always goes through
        ``get_honcho_client`` WITH this manager's bound config: a long session can't outlive its
        1h access token, and daemon threads can't see the ambient ContextVar profile, so a bare
        ``get_honcho_client()`` would migrate them onto the first-built profile.

        See #69123, #74065.
        """
        self._honcho = get_honcho_client(self._config)
        return self._honcho

    # ----- SDK object caches (generation-guarded against client rebuilds) -----

    def _cached_sdk_object(self, cache: dict[str, Any], key: str, fetch: Any) -> Any:
        """Get-or-fetch from ``cache``; a fetch that straddles a client rebuild is not cached."""
        while True:
            with self._cache_lock:
                if key in cache:
                    cache[key] = cache.pop(key)  # dict order is the LRU order the caps evict from
                    return cache[key]
                generation = self._client_generation
            obj = fetch()
            with self._cache_lock:
                if self._client_generation == generation:
                    return cache.setdefault(key, obj)
            # Client rebuilt mid-resolve: this object holds the discarded transport. Retry.

    def _cached_session(self, session_key: str) -> HonchoSession | None:
        """The locally cached session, stamped as used so recall alone keeps it out of the idle sweep."""
        with self._cache_lock:
            session = self._cache.get(session_key)
            if session is not None:
                session.updated_at = datetime.now()
                self._cache[session_key] = self._cache.pop(session_key)
        return session

    def _sdk_session(self, session_id: str) -> Any:
        """Get or create the SDK session (cached until a client rebuild clears the cache)."""
        return self._cached_sdk_object(self._sessions_cache, session_id, lambda: self.honcho.session(session_id))

    def _get_or_create_peer(self, peer_id: str) -> Any:
        """Get or create a Honcho peer (one get-or-create API call, then cached)."""
        return self._cached_sdk_object(
            self._peers_cache, peer_id, lambda: self._authed_call("peer setup", lambda: self.honcho.peer(peer_id)))

    def _peer_ref(self, peer_id: str) -> Any:
        """A peer handle for read-only calls. Built locally with no API call, so a read never
        creates the peer the way ``_get_or_create_peer`` would."""
        from honcho import Peer

        return Peer(peer_id, self.honcho)

    # ----- Session creation -----

    def _configure_session_peers(self, session_id: str, user_peer: Any, assistant_peer: Any, *, existing: bool) -> bool:
        """Join both peers to the session observing only themselves: each peer's memory is its own view, and
        nobody observes the other. A session that already had messages may hold older flags (the AI peer used
        to observe the user), and joining keeps stored flags, so those are set explicitly. False when auth died
        mid-way (already recorded by _authed_call)."""
        try:
            peers = (user_peer, assistant_peer)
            self._authed_call("session peer setup",
                              lambda: self._sdk_session(session_id).add_peers([(peer, observe_self()) for peer in peers]))
            if existing:
                self._authed_call("session peer configuration", lambda: [
                    self._sdk_session(session_id).set_peer_configuration(peer, observe_self()) for peer in peers])
        except HonchoAuthError:
            return False
        except Exception as e:
            logger.warning("Honcho session '%s' add_peers failed (non-fatal): %s", session_id, e)
        return True

    def _session_has_messages(self, session_id: str) -> bool:
        """Whether the session already holds a message (one page of one); True when the check fails, so a
        failure never sends scopes, which would copy an existing session's memory into them again."""
        try:
            page = self._authed_call("session message check", lambda: self._sdk_session(session_id).messages(size=1))
            return bool(list(getattr(page, "items", None) or []))
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Honcho session '%s' message check failed: %s", session_id, e)
            return True

    def _get_or_create_honcho_session(self, session_id: str, user_peer: Any, assistant_peer: Any) -> Any:
        """The SDK session with its peers configured. A session with no messages yet joins this manager's
        scopes: sending them for one that has messages would queue a copy of its memory into each scope again."""
        with self._cache_lock:
            if session_id in self._sessions_cache:
                logger.debug("Honcho session '%s' retrieved from cache", session_id)
                return self._sessions_cache[session_id]

        self._authed_call("session setup", lambda: self._sdk_session(session_id))
        existing = self._session_has_messages(session_id)
        if self._scopes and not existing:
            scopes = list(self._scopes)
            created = self._authed_call("session scopes", lambda: self.honcho.session(session_id, scopes=scopes))
            with self._cache_lock:
                self._sessions_cache[session_id] = created
        if self._configure_session_peers(session_id, user_peer, assistant_peer, existing=existing):
            logger.info("Honcho session '%s' %s", session_id, "retrieved" if existing else "created")

        with self._cache_lock:
            honcho_session = self._sessions_cache.get(session_id)
        if honcho_session is None:
            # A mid-init client rebuild dropped the cached session; resolve a fresh one.
            honcho_session = self._authed_call("session setup", lambda: self._sdk_session(session_id))
        return honcho_session

    @staticmethod
    def _has_unsynced(session: HonchoSession) -> bool:
        return any(not m.get("_synced") for m in list(session.messages))

    def _evict_session_locked(self, key: str, session: HonchoSession) -> None:
        """Drop one session and every entry keyed to it. Caller holds _cache_lock."""
        del self._cache[key]
        self._sessions_cache.pop(session.honcho_session_id, None)

    def _enforce_cache_caps_locked(self) -> None:
        """Evict least recently used entries above the hard caps. A session with unsynced messages
        is never evicted: it is the only copy until the flush lands. Caller holds _cache_lock."""
        for key, session in list(self._cache.items()):
            if len(self._cache) <= _SESSION_CACHE_MAX_SIZE:
                break
            if not self._has_unsynced(session):
                self._evict_session_locked(key, session)
        live_ids = {s.honcho_session_id for s in self._cache.values()}
        for session_id in list(self._sessions_cache):
            if len(self._sessions_cache) <= _SESSION_CACHE_MAX_SIZE:
                break
            if session_id not in live_ids:
                del self._sessions_cache[session_id]
        live_peers = {p for s in self._cache.values() for p in (s.user_peer_id, s.assistant_peer_id)}
        for peer_id in list(self._peers_cache):
            if len(self._peers_cache) <= _PEERS_CACHE_MAX_SIZE:
                break
            if peer_id not in live_peers:
                del self._peers_cache[peer_id]

    def _sweep_idle_sessions_locked(self) -> int:
        """Evict sessions idle beyond _SESSION_IDLE_TTL_SECONDS together with their SDK session, then enforce
        the caps. Caller holds _cache_lock."""
        cutoff = time.time() - _SESSION_IDLE_TTL_SECONDS
        evicted = 0
        for key, session in list(self._cache.items()):
            if session.updated_at.timestamp() >= cutoff or self._has_unsynced(session):
                continue
            self._evict_session_locked(key, session)
            evicted += 1
        self._enforce_cache_caps_locked()
        return evicted

    def _maybe_sweep_idle_sessions(self) -> None:
        """Rate-limited idle sweep, run from get_or_create so no watcher thread is needed."""
        now = time.time()
        with self._cache_lock:
            if now - self._last_idle_sweep_ts < _SESSION_SWEEP_INTERVAL_SECONDS:
                return
            self._last_idle_sweep_ts = now
            evicted = self._sweep_idle_sessions_locked()
        if evicted:
            logger.info("Honcho session cache idle sweep evicted %d session(s)", evicted)

    def get_or_create(self, key: str, *, user_peer_id: str | None = None) -> HonchoSession:
        """Get an existing session or create a new one for ``key`` (usually channel:chat_id).
        ``user_peer_id`` replaces the resolved user peer when the session's participant is not the
        runtime user, e.g. the sender bot of an a2a session."""
        self._maybe_sweep_idle_sessions()
        if (cached := self._cached_session(key)) is not None:
            logger.debug("Local session cache hit: %s", key)
            return cached

        # Gateway sessions normally use the platform-native runtime identity so multi-user
        # bots scope memory per user; config can alias/prefix it, or pinPeerName pins all
        # identities to peerName for single-user deployments (see _resolve_user_peer_id).
        # Determine peer IDs — no lock needed (read-only, no shared state mutation). See #14984.
        user_peer_id = user_peer_id or self._resolve_user_peer_id(key)
        assistant_peer_id = self.assistant_peer_id()

        # All expensive I/O outside the lock — Honcho's persistence is source of truth.
        honcho_session_id = self._sanitize_id(key)
        user_peer = self._get_or_create_peer(user_peer_id)
        assistant_peer = self._get_or_create_peer(assistant_peer_id)
        self._get_or_create_honcho_session(honcho_session_id, user_peer, assistant_peer)

        session = HonchoSession(key=key, user_peer_id=user_peer_id, assistant_peer_id=assistant_peer_id,
                                honcho_session_id=honcho_session_id)
        with self._cache_lock:
            self._cache[key] = session
            self._enforce_cache_caps_locked()
        return session

    # ----- Writes -----

    def _author_peer_for_session(self, honcho_session: Any, honcho_session_id: str, author_peer_id: str) -> Any:
        """The author's peer, joined to the session the first time it writes.

        Joins are remembered per session, so this costs one API call per author."""
        peer = self._get_or_create_peer(author_peer_id)
        with self._cache_lock:
            if author_peer_id in self._joined_author_peers.get(honcho_session_id, ()):
                return peer
        try:
            honcho_session.add_peers([(peer, observe_self())])
        except Exception as e:
            # The write still lands under the right peer. Only the membership (observe config) is missing.
            logger.debug("Honcho author peer join failed for %s: %s", author_peer_id, e)
            return peer
        with self._cache_lock:
            self._joined_author_peers.setdefault(honcho_session_id, set()).add(author_peer_id)
            while len(self._joined_author_peers) > _SESSION_CACHE_MAX_SIZE:
                self._joined_author_peers.pop(next(iter(self._joined_author_peers)))
        return peer
    @staticmethod
    def _trim_synced_messages(session: HonchoSession) -> None:
        """Drop the oldest synced messages beyond _SESSION_MESSAGE_RETENTION. Messages are appended in
        order, so trimming from the front stops at the first unsynced one and never drops it."""
        excess = len(session.messages) - _SESSION_MESSAGE_RETENTION
        while excess > 0 and session.messages and session.messages[0].get("_synced"):
            session.messages.pop(0)
            excess -= 1

    def _flush_session(self, session: HonchoSession) -> bool:
        """Write unsynced messages to Honcho synchronously. The session's lock keeps the async writer and an
        exit-time flush_all() from posting the same batch."""
        with session._flush_lock:
            return self._flush_session_locked(session)

    def _flush_session_before(self, session: HonchoSession, deadline: float | None) -> bool:
        """_flush_session that starts only while ``deadline`` has time left and waits for the session's lock no
        longer than that; False, with nothing sent, otherwise. An upload that has started is not interrupted: the
        SDK has no per-call timeout, so it runs to the client's HTTP timeout."""
        if deadline is None:
            self._flush_session(session)
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not session._flush_lock.acquire(timeout=remaining):
            return False
        try:
            self._flush_session(session)  # re-enters the RLock already held above
        finally:
            session._flush_lock.release()
        return True

    def _flush_session_locked(self, session: HonchoSession) -> bool:
        new_messages = [m for m in session.messages if not m.get("_synced")]
        if not new_messages:
            return True

        # Resolved inside the operation so a retry after a client rebuild gets fresh objects.
        def _sync_messages() -> int:
            user_peer = self._get_or_create_peer(session.user_peer_id)
            assistant_peer = self._get_or_create_peer(session.assistant_peer_id)
            honcho_session = self._sessions_cache.get(session.honcho_session_id)
            if honcho_session is None:
                honcho_session = self._get_or_create_honcho_session(session.honcho_session_id, user_peer, assistant_peer)
            honcho_messages = []
            for m in new_messages:
                if m["role"] != "user":
                    honcho_messages.append(assistant_peer.message(m["content"]))
                    continue
                author_peer_id = m.get("author_peer_id")
                peer = (self._author_peer_for_session(honcho_session, session.honcho_session_id, author_peer_id)
                        if author_peer_id else user_peer)
                honcho_messages.append(peer.message(m["content"]))
            honcho_session.add_messages(honcho_messages)
            return len(honcho_messages)

        try:
            logger.debug("Synced %d messages to Honcho for %s", self._authed_call("message sync", _sync_messages), session.key)
            ok = True
        except Exception as e:
            logger.error("Failed to sync messages to Honcho: %s", e)
            ok = False
        for msg in new_messages:
            msg["_synced"] = ok
        if ok:
            # Under the cache lock so the eviction check never iterates a list being trimmed.
            with self._cache_lock:
                self._trim_synced_messages(session)
        return ok

    def _try_flush(self, session: HonchoSession, level: int, msg: str) -> bool:
        """_flush_session that logs (never raises) a failure; False when the batch didn't land."""
        try:
            if self._flush_session(session):
                return True
            logger.log(level, msg)
        except Exception as e:
            logger.log(level, msg + ": %s", e)
        return False

    def _async_writer_loop(self) -> None:
        """Background daemon thread: drains the async write queue, retrying each batch once."""
        while True:
            try:
                item = self._async_queue.get(timeout=5)
                if item is _ASYNC_SHUTDOWN:
                    break
                if not self._try_flush(item, logging.WARNING, "Honcho async write failed, retrying once"):
                    if self._shutting_down:
                        # The shutdown flush is already attempting this session within its own budget.
                        logger.error("Honcho async write failed while shutting down, so the batch stays unsynced")
                        continue
                    time.sleep(2)
                    self._try_flush(item, logging.ERROR, "Honcho async write retry failed, dropping batch")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Honcho async writer error: %s", e)

    def _reclaim_key_locked(self, session: HonchoSession) -> HonchoSession | None:
        """Put a session with unsynced messages back where flush_all() looks. Returns the newer object that
        owns the key when there is one (this session's batch still needs a home), None otherwise."""
        if not self._has_unsynced(session):
            return None
        current = self._cache.get(session.key)
        if current is None:
            self._cache[session.key] = session
        return None if current is None or current is session else current

    def _retain_for_retry(self, session: HonchoSession) -> None:
        """An evicted key takes the session back; a key a newer object owns keeps that object, and this one
        waits in the retry list."""
        with self._cache_lock:
            if self._reclaim_key_locked(session) is not None and not any(s is session for s in self._retry_sessions):
                self._retry_sessions.append(session)

    def _flush_now(self, session: HonchoSession) -> None:
        """A save-time flush whose failed batch stays reachable for flush_all(), even after an eviction."""
        if not self._flush_session(session):
            self._retain_for_retry(session)

    def save(self, session: HonchoSession) -> None:
        """Queue the session's new messages for the background writer; once shutdown began, write them now."""
        # Under the writer lock, so a put cannot slip in after shutdown drained the queue.
        with self._async_thread_lock:
            if not self._shutting_down:
                self._ensure_async_writer_locked()
                self._async_queue.put(session)
                return
        self._flush_now(session)

    def flush_all(self, timeout: float | None = None) -> None:
        """Flush unsynced messages for all cached sessions, then drain the async queue inline. ``timeout`` bounds
        when a flush may start and how long it waits for a session's lock, not an upload already in flight, which
        runs to the client's HTTP timeout. A session skipped keeps its messages and is counted in one warning."""
        deadline = None if timeout is None else time.monotonic() + timeout
        skipped = self._flush_cached_before(deadline)
        skipped.extend(self._drain_async_queue(deadline))
        self._warn_unsynced(skipped, timeout)

    def _flush_cached_before(self, deadline: float | None) -> list[HonchoSession]:
        """Flush every cached and retry-listed session that ``deadline`` allows; returns the ones it did not."""
        with self._cache_lock:
            sessions = list(self._cache.values())
            sessions += [s for s in self._retry_sessions if not any(s is c for c in sessions)]
        skipped: list[HonchoSession] = []
        for session in sessions:
            try:
                if not self._flush_session_before(session, deadline):
                    skipped.append(session)
            except Exception as e:
                logger.error("Honcho flush_all error for %s: %s", session.key, e)
        with self._cache_lock:
            self._retry_sessions = [s for s in self._retry_sessions if self._has_unsynced(s)]
        return skipped

    def _warn_unsynced(self, skipped: list[HonchoSession], timeout: float | None) -> None:
        left = [s for s in {id(s): s for s in skipped}.values() if self._has_unsynced(s)]
        if left:
            unsynced = sum(1 for s in left for m in list(s.messages) if not m.get("_synced"))
            logger.warning("Honcho flush ran out of time after %.1fs with %d message(s) in %d session(s) still unsynced",
                           timeout or 0.0, unsynced, len(left))

    def _drain_async_queue(self, deadline: float | None = None) -> list[HonchoSession]:
        """Flush every queued session inline. Returns the sessions ``deadline`` left unflushed."""
        skipped: list[HonchoSession] = []
        while not self._async_queue.empty():
            try:
                item = self._async_queue.get_nowait()
            except queue.Empty:
                break
            if item is not _ASYNC_SHUTDOWN and not self._flush_session_before(item, deadline):
                skipped.append(item)
        return skipped

    def _ensure_async_writer_locked(self) -> None:
        if self._async_thread is None or not self._async_thread.is_alive():
            self._async_thread = spawn_context_thread(self._async_writer_loop, name="honcho-async-writer", owner=self)
            self._async_thread.start()

    def _stop_async_writer_before(self, deadline: float) -> list[HonchoSession]:
        """Join the writer for what is left of ``deadline``, then drain the queue under the same deadline. A writer
        still inside an upload keeps running to the client's HTTP timeout; it holds the shared client, so it is
        joined, never abandoned."""
        with self._async_thread_lock:
            self._shutting_down = True
        if self._async_thread is not None and self._async_thread.is_alive():
            self._async_queue.put(_ASYNC_SHUTDOWN)
            self._async_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return self._drain_async_queue(deadline)

    def shutdown(self, timeout: float = 10.0) -> None:
        """Flush everything, then stop the async writer thread, within ``timeout``. The budget stops new uploads
        from starting and bounds the lock waits and the join; an upload already in flight runs to the client's
        HTTP timeout. Whatever stayed unsynced is counted in one warning."""
        with self._async_thread_lock:
            self._shutting_down = True
        deadline = time.monotonic() + timeout
        skipped = self._flush_cached_before(deadline)
        skipped.extend(self._drain_async_queue(deadline))
        skipped.extend(self._stop_async_writer_before(deadline))
        self._warn_unsynced(skipped, timeout)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Callable  # noqa: F401,E402
from pathlib import Path  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

"""The first-turn dialectic join must be skipped on api_server (chat) platforms.

Every api_server POST builds a fresh agent, so ``_turn_count == 1`` on every request and the
bounded first-turn dialectic join (``firstTurnDialecticWait``) is paid on every single turn while
the caller waits for the reply — and the agent that waited is discarded with the response, so the
join buys latency and nothing else. The dialectic itself is an LLM call on the memory host that
continues in the background; only the blocking join is skipped, and only for the api_server
platform. Interactive platforms keep the wait.

``firstTurnDialecticWait`` cannot express this: it is read from the per-host config block
(``client.py``: ``look.parsed("firstTurnDialecticWait", ...)``), so a host that serves both the
CLI and api_server — the normal self-hosted shape — cannot zero it for the chat path without also
removing the wait for a human at a prompt. The gate therefore keys on the platform the provider
was built for, not on config.

Every timing assertion here is expressed relative to the cap this suite configures itself, so each
one describes the behaviour (a bounded join vs no join) rather than how long anything took on the
machine or deployment the suite happened to run on.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from hermes_honcho import HonchoMemoryProvider

_DIALECTIC_WAIT = 0.5  # the join cap this suite configures; every timing assertion is relative to it


def _config(*, dialectic_wait: float = _DIALECTIC_WAIT) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        recall_sync=False,
        init_on_session_start=False,
        injection_frequency="every-turn",
        context_cadence=1,
        dialectic_cadence=1,
        query_rewrite=False,
        first_turn_base_wait=0.2,
        first_turn_dialectic_wait=dialectic_wait,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-session",
        timeout=None,
    )


class _Manager:
    """Minimal manager: the recall HTTP calls return immediately, as they do in production."""

    def get_prefetch_context(self, session_key, user_message=None):
        return {"representation": "cached representation"}

    def set_context_result(self, session_key, result):
        pass

    def pop_context_result(self, session_key):
        return {}


def _ready_provider(monkeypatch, platform: str, *, dialectic_wait: float = _DIALECTIC_WAIT,
                    dialectic_release: threading.Event | None = None) -> HonchoMemoryProvider:
    """A provider on turn 1 of a fresh api_server/CLI agent whose dialectic never returns early."""
    provider = HonchoMemoryProvider()
    monkeypatch.setattr(
        "hermes_honcho.client.HonchoClientConfig.from_global_config",
        lambda: SimpleNamespace(enabled=False, api_key=None, base_url=None),
    )
    # initialize() only records the platform here (not configured -> no network work).
    provider.initialize("session-1", platform=platform)
    provider._config = _config(dialectic_wait=dialectic_wait)
    provider._manager = _Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True
    provider._recall_mode = "hybrid"
    provider._recall_sync = False
    provider._turn_count = 1
    provider._FIRST_TURN_DIALECTIC_CAP = dialectic_wait
    provider._FIRST_TURN_BASE_TIMEOUT = 0.2
    if dialectic_release is not None:
        # A dialectic that behaves like the real one: an LLM call far longer than the cap.
        provider._run_dialectic_depth = (
            lambda query, use_query_rewrite=True: (dialectic_release.wait(timeout=10), "late result")[1]
        )
    return provider


def test_platform_is_recorded_from_initialize(monkeypatch):
    """The gate keys on the request's platform, so both values must survive initialize()."""
    from hermes_honcho import _NO_FIRST_TURN_DIALECTIC_WAIT_PLATFORMS

    api = HonchoMemoryProvider()
    cli = HonchoMemoryProvider()
    monkeypatch.setattr(
        "hermes_honcho.client.HonchoClientConfig.from_global_config",
        lambda: SimpleNamespace(enabled=False, api_key=None, base_url=None),
    )

    api.initialize("session-1", platform="api_server")
    cli.initialize("session-1", platform="cli")

    assert api._platform == "api_server"
    assert cli._platform == "cli"
    assert "api_server" in _NO_FIRST_TURN_DIALECTIC_WAIT_PLATFORMS
    assert "cli" not in _NO_FIRST_TURN_DIALECTIC_WAIT_PLATFORMS


@pytest.mark.parametrize("platform", ["cli", "tui"])
def test_interactive_first_turn_still_joins_the_dialectic(monkeypatch, platform):
    """Interaction is unchanged where a human waits at a prompt: the bounded join still happens."""
    release = threading.Event()
    provider = _ready_provider(monkeypatch, platform, dialectic_release=release)
    try:
        started = time.perf_counter()
        provider.prefetch("what did we decide about the rollback plan?")
        elapsed = time.perf_counter() - started
        assert elapsed >= _DIALECTIC_WAIT * 0.8, f"{platform} stopped waiting on the dialectic"
    finally:
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5)


def test_api_server_first_turn_does_not_join_the_dialectic(monkeypatch):
    """The chat path returns to the model call immediately; the dialectic keeps running."""
    release = threading.Event()
    provider = _ready_provider(monkeypatch, "api_server", dialectic_release=release)
    try:
        started = time.perf_counter()
        provider.prefetch("what did we decide about the rollback plan?")
        elapsed = time.perf_counter() - started
        assert elapsed < _DIALECTIC_WAIT * 0.4, f"api_server still burned {elapsed:.3f}s on the join"
        # The memory work is untouched: the dialectic was spawned and is still in flight.
        assert provider._prefetch_thread is not None
        assert provider._prefetch_thread.is_alive()
    finally:
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5)


def test_api_server_still_spawns_the_dialectic_nobody_waits_for(monkeypatch):
    """Skipping the join must not skip the dialectic: the background run still publishes."""
    release = threading.Event()
    provider = _ready_provider(monkeypatch, "api_server", dialectic_release=release)
    try:
        assert provider._last_dialectic_turn == -999
        provider._first_turn_dialectic_wait("what did we decide about the rollback plan?")
        thread = provider._prefetch_thread
        assert thread is not None and thread.is_alive()
    finally:
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5)


def test_api_server_consumes_a_ready_dialectic_without_waiting(monkeypatch):
    """A dialectic that already landed is still injected — only the wait is dropped."""
    provider = _ready_provider(monkeypatch, "api_server")
    provider._prefetch_result = "the user prefers terse answers"
    provider._prefetch_result_fired_at = 1

    started = time.perf_counter()
    out = provider.prefetch("what did we decide about the rollback plan?")
    elapsed = time.perf_counter() - started

    assert elapsed < _DIALECTIC_WAIT * 0.6, f"a landed dialectic still cost {elapsed:.3f}s of waiting"
    assert "the user prefers terse answers" in out

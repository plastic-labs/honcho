"""A process's first turn waits for its dialectic the same way whether it opens a session or resumes one.

In one long-lived process, turn N's injection comes from the dialectic queued at the end of
turn N-1. ``hermes chat --resume <id> --query ...`` starts a new process at turn N, where that
dialectic never ran and later turns do not wait, so the resumed turn injected nothing.

Driven through Hermes' real MemoryManager: it abandons a provider prefetch after a few seconds,
so first-turn waits that ran inside prefetch() could never be configured longer than that.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from agent.memory_manager import MemoryManager
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig

PREVIOUS = "Add the ShapeIndex encoder."
CURRENT = "Now decode it back."
DIALECTIC_ANSWER = "The user wants every encoder round-trip tested."
LATENCY = 0.3


def _provider(*, prewarm_latency: float = LATENCY, prewarm_answer: str = "Generic prewarm summary.",
              init_kwargs: dict | None = None) -> HonchoMemoryProvider:
    """A provider over a Honcho that answers slowly."""
    manager = MagicMock()
    manager.get_peer_card.return_value = []
    manager.dialectic_threads = []

    def _dialectic(*_args, **_kwargs):
        manager.dialectic_threads.append(threading.current_thread().name)
        if threading.current_thread().name == "honcho-prewarm-dialectic":
            time.sleep(prewarm_latency)
            return prewarm_answer
        time.sleep(LATENCY)
        return DIALECTIC_ANSWER

    manager.dialectic_query.side_effect = _dialectic

    cfg = HonchoClientConfig(api_key="test-key", enabled=True)
    provider = HonchoMemoryProvider()
    with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
         patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
         patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=manager), \
         patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
        provider.initialize(session_id="resumed-session", **(init_kwargs or {}))
    if provider._init_thread is not None:
        provider._init_thread.join(timeout=5.0)
    return provider


def _memory(provider: HonchoMemoryProvider) -> MemoryManager:
    # Shorter than every Honcho call above, so any wait left inside prefetch() is abandoned.
    memory = MemoryManager(external_prefetch_timeout=0.2)
    memory.add_provider(provider)
    return memory


def test_resumed_turn_injects_a_dialectic_it_waited_for():
    provider = _provider()
    memory = _memory(provider)

    memory.on_turn_start(2, CURRENT, previous_message=PREVIOUS)
    injected = memory.prefetch_all(CURRENT)

    assert DIALECTIC_ANSWER in injected
    # The catch-up stands in for one missed refresh; the next turn is an ordinary later turn that does not wait.
    memory.on_turn_start(3, "Cover the edge cases.", previous_message=CURRENT)
    assert provider._manager.dialectic_threads.count("honcho-resume-dialectic") == 1


def test_first_turn_waits_outlast_the_memory_managers_prefetch_timeout():
    memory = _memory(_provider())

    memory.on_turn_start(1, CURRENT)
    injected = memory.prefetch_all(CURRENT)

    # Turn 1 reuses the in-flight process-start prewarm rather than starting its own dialectic.
    assert "Generic prewarm summary." in injected


@pytest.mark.parametrize(
    "turn, turn_start_kwargs",
    [(1, {}), (2, {"previous_message": PREVIOUS})],
    ids=["new-session", "resumed"],
)
def test_first_turn_waits_are_bounded(turn, turn_start_kwargs):
    provider = _provider()
    provider._FIRST_TURN_BASE_TIMEOUT = provider._FIRST_TURN_DIALECTIC_CAP = 0.05
    memory = _memory(provider)

    started = time.monotonic()
    memory.on_turn_start(turn, CURRENT, **turn_start_kwargs)
    elapsed = time.monotonic() - started

    assert elapsed < LATENCY
    injected = memory.prefetch_all(CURRENT)
    assert DIALECTIC_ANSWER not in injected and "Generic prewarm summary." not in injected


def test_late_prewarm_does_not_replace_the_resumed_dialectic():
    provider = _provider(prewarm_latency=4 * LATENCY)
    memory = _memory(provider)

    memory.on_turn_start(2, CURRENT, previous_message=PREVIOUS)
    time.sleep(3 * LATENCY)  # the prewarm finishes after the resumed dialectic

    injected = memory.prefetch_all(CURRENT)
    assert DIALECTIC_ANSWER in injected
    assert "Generic prewarm summary." not in injected


def test_trivial_first_turn_leaves_the_dialectic_wait_to_the_next_turn():
    # The prewarm comes back empty, so no dialectic has landed when the first real question arrives.
    provider = _provider(prewarm_latency=0.0, prewarm_answer="")
    memory = _memory(provider)

    memory.on_turn_start(1, "hi")
    memory.prefetch_all("hi")
    provider.on_turn_start(2, CURRENT)

    # Called directly: the turn-2 dialectic wait still runs inside prefetch(), as it always has.
    assert DIALECTIC_ANSWER in provider.prefetch(CURRENT)


def test_a_session_hermes_says_it_is_continuing_skips_the_paid_prewarm_and_first_turn_join():
    """The API server builds a fresh agent per request, so its turn counter restarts at 1 mid-conversation."""
    provider = _provider(init_kwargs={"resumed": True, "message_count": 4})
    memory = _memory(provider)

    memory.on_turn_start(1, CURRENT)
    memory.prefetch_all(CURRENT)

    assert "honcho-prewarm-dialectic" not in provider._manager.dialectic_threads
    assert "honcho-prefetch-first" not in provider._manager.dialectic_threads


def test_agent_model_header_follows_the_model_hermes_reports():
    from hermes_plugin_honcho.client import AGENT_MODEL_HEADER

    provider = _provider()
    client = MagicMock(_async_http=None, _http_config={"default_headers": {}})
    client._http.default_headers = {}
    provider._manager.honcho = client

    provider.on_turn_start(1, CURRENT, model="openai/gpt-6-astra")
    assert client._http.default_headers[AGENT_MODEL_HEADER] == "openai/gpt-6-astra"

    # A turn Hermes reports no model for removes the header instead of keeping the last one.
    provider.on_turn_start(2, CURRENT, model="")
    assert AGENT_MODEL_HEADER not in client._http.default_headers
    assert AGENT_MODEL_HEADER not in client._http_config["default_headers"]

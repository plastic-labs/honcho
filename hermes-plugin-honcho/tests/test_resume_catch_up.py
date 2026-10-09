"""A process that resumes a session must inject what a long-lived process would have.

In one long-lived process, turn N's injection comes from the refresh queued at the end of
turn N-1. ``hermes chat --resume <id> --query ...`` starts a new process at turn N, where that
refresh never ran and later turns do not wait, so the resumed turn injected nothing.

Driven through Hermes' real MemoryManager: it abandons a provider prefetch after a few seconds,
and a catch-up that waited inside prefetch() was abandoned with it.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from agent.memory_manager import MemoryManager

from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig

PREVIOUS = "Add the ShapeIndex encoder."


def _resumed_provider(context_queries: list[str], *, save_messages: bool) -> HonchoMemoryProvider:
    """A resumed provider, with a Honcho that answers slowly and holds the earlier turn only if it saved it."""
    manager = MagicMock()
    manager.get_or_create.return_value.messages = (
        [{"role": "user", "content": PREVIOUS}, {"role": "assistant", "content": "Done."}] if save_messages else []
    )
    manager.pop_context_result.return_value = {}

    def _context(_key, query=None, **_kwargs):
        context_queries.append(query)
        time.sleep(0.3)
        return {"representation": f"conclusions recalled for: {query}"}

    def _dialectic(*_args, **_kwargs):
        time.sleep(0.3)
        return "The user wants every encoder round-trip tested."

    manager.get_prefetch_context.side_effect = _context
    manager.dialectic_query.side_effect = _dialectic

    cfg = HonchoClientConfig(api_key="test-key", enabled=True, recall_mode="hybrid", dialectic_depth=1,
                             save_messages=save_messages)
    provider = HonchoMemoryProvider()
    with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
         patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
         patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=manager), \
         patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
        provider.initialize(session_id="resumed-session")
    if provider._init_thread is not None:
        provider._init_thread.join(timeout=5.0)
    return provider


@pytest.mark.parametrize(
    "save_messages, turn_start_kwargs",
    [(False, {"previous_message": PREVIOUS}), (True, {})],
    ids=["from-hermes-transcript", "from-honcho-history"],
)
def test_resumed_turn_injects_the_previous_turns_refresh(save_messages, turn_start_kwargs):
    context_queries: list[str] = []
    provider = _resumed_provider(context_queries, save_messages=save_messages)
    memory = MemoryManager(external_prefetch_timeout=0.2)
    memory.add_provider(provider)

    memory.on_turn_start(2, "Now decode it back.", **turn_start_kwargs)
    injected = memory.prefetch_all("Now decode it back.")

    assert f"conclusions recalled for: {PREVIOUS}" in injected
    assert "The user wants every encoder round-trip tested." in injected
    assert context_queries[0] == PREVIOUS

    # The catch-up stands in for one missed refresh; the next turn is an ordinary later turn.
    memory.on_turn_start(3, "Cover the edge cases.", previous_message="Now decode it back.")
    memory.prefetch_all("Cover the edge cases.")
    assert context_queries.count(PREVIOUS) == 1

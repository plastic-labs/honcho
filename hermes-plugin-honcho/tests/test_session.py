"""Tests for plugins/memory/honcho/session.py — HonchoSession and helpers."""

import threading
import time

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from hermes_plugin_honcho.session import (
    HonchoSession,
    HonchoSessionManager,
)
from hermes_plugin_honcho import HonchoMemoryProvider


# ---------------------------------------------------------------------------
# HonchoSession dataclass
# ---------------------------------------------------------------------------


class TestHonchoSession:
    def _make_session(self):
        return HonchoSession(
            key="telegram:12345",
            user_peer_id="user-telegram-12345",
            assistant_peer_id="hermes-assistant",
            honcho_session_id="telegram-12345",
        )

    def test_initial_state(self):
        session = self._make_session()
        assert session.key == "telegram:12345"
        assert session.messages == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)

    def test_add_message(self):
        session = self._make_session()
        session.add_message("user", "Hello!")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello!"
        assert "timestamp" in session.messages[0]


# ---------------------------------------------------------------------------
# HonchoSessionManager._sanitize_id
# ---------------------------------------------------------------------------


class TestSanitizeId:
    def test_clean_id_unchanged(self):
        mgr = HonchoSessionManager()
        assert mgr._sanitize_id("telegram-12345") == "telegram-12345"


    def test_special_chars_replaced(self):
        mgr = HonchoSessionManager()
        result = mgr._sanitize_id("user@chat#room!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result


class TestPeerLookupHelpers:
    def _make_cached_manager(self):
        mgr = HonchoSessionManager()
        session = HonchoSession(
            key="telegram:123",
            user_peer_id="robert",
            assistant_peer_id="hermes",
            honcho_session_id="telegram-123",
        )
        mgr._cache[session.key] = session
        return mgr, session


    def test_set_peer_card_writes_the_users_own_card(self):
        # Every peer observes only itself, so writes go to the user's own card, where reads look.
        mgr, session = self._make_cached_manager()
        user_peer = MagicMock()
        user_peer.set_card.return_value = ["Role: user"]
        mgr._get_or_create_peer = MagicMock(return_value=user_peer)

        result = mgr.set_peer_card(session.key, ["Role: user"])

        assert result == ["Role: user"]
        mgr._get_or_create_peer.assert_called_with(session.user_peer_id)
        user_peer.set_card.assert_called_once_with(["Role: user"])

    def test_search_context_uses_peer_perspective_message_search(self):
        """Search spans the target peer's sessions instead of its representation."""
        mgr, session = self._make_cached_manager()
        honcho_client = MagicMock()
        honcho_client.search.return_value = [
            SimpleNamespace(content="Robert runs neuralancer", peer_id="hermes", session_id="s-old", id="m1"),
            SimpleNamespace(content="I founded neuralancer in 2019", peer_id="robert", session_id="s-old", id="m2"),
        ]
        with patch.object(HonchoSessionManager, "honcho", new_callable=lambda: property(lambda s: honcho_client)):
            result = mgr.search_context(session.key, "neuralancer")

        # Returns the actual message content, ranked.
        assert "Robert runs neuralancer" in result
        assert "neuralancer in 2019" in result
        # Scoped to the target (user) peer's sessions, all authors.
        honcho_client.search.assert_called_once()
        _args, kwargs = honcho_client.search.call_args
        assert kwargs["filters"] == {"peer_perspective": session.user_peer_id}
        # Assistant-authored messages are labeled so the model can tell
        # user-stated facts from assistant-derived ones.
        assert "[assistant" in result

    def _search_with_fallback(self, perspective_result):
        """Run search_context with the perspective search returning (or raising) ``perspective_result``."""
        mgr, session = self._make_cached_manager()
        honcho_client = MagicMock()
        if isinstance(perspective_result, Exception):
            honcho_client.search.side_effect = perspective_result
        else:
            honcho_client.search.return_value = perspective_result
        fallback_peer = MagicMock()
        fallback_peer.search.return_value = [
            SimpleNamespace(content="I founded neuralancer in 2019", peer_id="robert", session_id="s-old", id="m2"),
        ]
        mgr._peer_ref = MagicMock(return_value=fallback_peer)
        mgr._get_or_create_peer = MagicMock(side_effect=AssertionError("search must not create a peer"))
        with patch.object(HonchoSessionManager, "honcho", new_callable=lambda: property(lambda s: honcho_client)):
            result = mgr.search_context(session.key, "neuralancer")
        return result, session, mgr, fallback_peer

    def test_search_context_falls_back_to_peer_search_on_empty_perspective(self):
        """A pre-v3.1.0 server returns [] for history outside a moved joined_at window (#1376)."""
        result, session, mgr, fallback_peer = self._search_with_fallback([])

        assert "neuralancer in 2019" in result
        mgr._peer_ref.assert_called_once_with(session.user_peer_id)
        fallback_peer.search.assert_called_once()

    def test_search_context_falls_back_to_peer_search_when_perspective_raises(self):
        """Servers without the perspective filter raise; peer-authored search still answers."""
        result, session, mgr, fallback_peer = self._search_with_fallback(RuntimeError("unknown filter"))

        assert "neuralancer in 2019" in result
        mgr._peer_ref.assert_called_once_with(session.user_peer_id)

    def test_search_context_skips_fallback_when_perspective_has_results(self):
        result, _session, mgr, fallback_peer = self._search_with_fallback(
            [SimpleNamespace(content="Robert runs neuralancer", peer_id="hermes", session_id="s-old", id="m1")]
        )

        assert "Robert runs neuralancer" in result
        mgr._peer_ref.assert_not_called()
        fallback_peer.search.assert_not_called()

    def test_peer_ref_makes_no_api_call(self):
        """The fallback's peer handle is local: building it must not get-or-create the peer."""
        from honcho import Peer

        mgr, _session = self._make_cached_manager()
        honcho_client = MagicMock(workspace_id="ws")
        with patch.object(HonchoSessionManager, "honcho", new_callable=lambda: property(lambda s: honcho_client)):
            peer = mgr._peer_ref("robert")

        assert isinstance(peer, Peer)
        assert peer.id == "robert"
        assert honcho_client.method_calls == []


    def test_create_conclusion_defaults_to_user_target(self):
        mgr, session = self._make_cached_manager()
        assistant_peer = MagicMock()
        scope = MagicMock()
        assistant_peer.conclusions_of.return_value = scope
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)

        ok = mgr.create_conclusion(session.key, "User prefers dark mode")

        assert ok is True
        assistant_peer.conclusions_of.assert_called_once_with(session.user_peer_id)
        scope.create.assert_called_once_with([{
            "content": "User prefers dark mode",
            "session_id": session.honcho_session_id,
        }])


class TestConcludeToolDispatch:
    def test_conclude_schema_has_no_anyof(self):
        """anyOf/oneOf/allOf breaks Anthropic and Fireworks APIs — schema must be plain object."""
        from hermes_plugin_honcho.tool_schemas import CONCLUDE_SCHEMA
        params = CONCLUDE_SCHEMA["parameters"]
        assert params["type"] == "object"
        assert "conclusion" in params["properties"]
        assert "delete_id" in params["properties"]
        assert "list" in params["properties"]
        assert "query" in params["properties"]
        assert "anyOf" not in params
        assert "oneOf" not in params
        assert "allOf" not in params

    def test_honcho_conclude_defaults_to_user_peer(self):
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._manager.create_conclusion.return_value = True

        result = provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "User prefers dark mode"},
        )

        assert "Conclusion saved for user" in result
        provider._manager.create_conclusion.assert_called_once_with(
            "telegram:123",
            "User prefers dark mode",
            peer="user",
        )


    def test_sync_turn_strips_leaked_memory_context_before_honcho_ingest(self):
        provider = HonchoMemoryProvider()
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._cron_skipped = False
        provider._config = SimpleNamespace(message_max_chars=25000)

        session = MagicMock()
        provider._manager.get_or_create.return_value = session

        provider.sync_turn(
            (
                "hello\n\n"
                "<memory-context>\n"
                "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n"
                "## Honcho Context\n"
                "stale memory\n"
                "</memory-context>"
            ),
            (
                "<memory-context>\n"
                "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n"
                "## Honcho Context\n"
                "stale memory\n"
                "</memory-context>\n\n"
                "Visible answer"
            ),
        )
        provider._sync_thread.join(timeout=1.0)

        assert session.add_message.call_args_list[0].args == ("user", "hello")
        assert session.add_message.call_args_list[1].args == ("assistant", "Visible answer")


# ---------------------------------------------------------------------------
# Message chunking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider init behavior: lazy vs eager in tools mode
# ---------------------------------------------------------------------------


class TestInitPassesIdentity:
    """Session init hands the gateway's user ids to the session manager."""

    def _make_provider_with_config(self, peer_name=None, user_id=None, user_id_alt=None):
        """Create a HonchoMemoryProvider with mocked config and dependencies."""
        from hermes_plugin_honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(api_key="test-key", enabled=True, peer_name=peer_name)

        provider = HonchoMemoryProvider()

        # Patch the config loading and session init to avoid real Honcho calls
        from unittest.mock import patch, MagicMock

        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        init_kwargs = {}
        if user_id:
            init_kwargs["user_id"] = user_id
        if user_id_alt:
            init_kwargs["user_id_alt"] = user_id_alt

        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager) as mock_manager_cls, \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-001", **init_kwargs)
            provider._init_thread.join(timeout=5)
            _settle_prewarm(provider)

        return provider, cfg, mock_manager_cls

    def test_explicit_peer_name_not_overridden_by_user_id(self):
        """Explicit peerName in config must not be replaced by gateway user_id."""
        _, cfg, _ = self._make_provider_with_config(peer_name="Kathie", user_id="8439114563")
        assert cfg.peer_name == "Kathie"


    def test_user_id_alt_is_passed_to_session_manager(self):
        """Gateway alternate user IDs are available for Honcho alias matching."""
        _, _, mock_manager_cls = self._make_provider_with_config(user_id="open-id", user_id_alt="union-id")
        assert mock_manager_cls.call_args.kwargs["runtime_user_peer_name"] == "open-id"
        assert mock_manager_cls.call_args.kwargs["runtime_user_peer_name_alt"] == "union-id"


class TestChunkMessage:
    def test_short_message_single_chunk(self):
        result = HonchoMemoryProvider._chunk_message("hello world", 100)
        assert result == ["hello world"]


    def test_splits_at_paragraph_boundary(self):
        msg = "first paragraph.\n\nsecond paragraph."
        # limit=30: total is 35, forces split; second chunk with prefix is 29, fits
        result = HonchoMemoryProvider._chunk_message(msg, 30)
        assert len(result) == 2
        assert result[0] == "first paragraph."
        assert result[1] == "[continued] second paragraph."


    def test_continuation_prefix(self):
        msg = "a" * 200
        result = HonchoMemoryProvider._chunk_message(msg, 50)
        assert len(result) >= 2
        assert not result[0].startswith("[continued]")
        for chunk in result[1:]:
            assert chunk.startswith("[continued] ")


# ---------------------------------------------------------------------------
# Context token budget enforcement
# ---------------------------------------------------------------------------


class TestDialecticInputGuard:
    def test_long_query_truncated(self):
        """Queries over 10000 chars are truncated."""
        mgr = HonchoSessionManager()

        # Create a cached session so dialectic_query doesn't bail early
        session = HonchoSession(
            key="test", user_peer_id="u", assistant_peer_id="a",
            honcho_session_id="s",
        )
        mgr._cache["test"] = session

        # Mock the peer to capture the query
        mock_peer = MagicMock()
        mock_peer.chat.return_value = "answer"
        mgr._get_or_create_peer = MagicMock(return_value=mock_peer)

        long_query = "word " * 2500  # 12500 chars, over the 10000 limit
        mgr.dialectic_query("test", long_query)

        # The query passed to chat() should be truncated
        actual_query = mock_peer.chat.call_args[0][0]
        assert len(actual_query) <= 10000


class TestDialecticInjectionCap:
    """The 600-char cap applies to injection, not explicit chat calls."""

    def _manager_with_long_answer(self, answer):
        mgr = HonchoSessionManager()

        session = HonchoSession(
            key="test", user_peer_id="u", assistant_peer_id="a",
            honcho_session_id="s",
        )
        mgr._cache["test"] = session

        mock_peer = MagicMock()
        mock_peer.chat.return_value = answer
        mgr._get_or_create_peer = MagicMock(return_value=mock_peer)
        return mgr

    def test_injection_path_truncates(self):
        """Default (auto-injection) path clips to 600 chars with an ellipsis."""
        answer = "fact " * 200  # 1000 chars, well over the 600-char cap
        mgr = self._manager_with_long_answer(answer)

        result = mgr.dialectic_query("test", "summarize")

        assert len(result) <= 610  # cap + word-boundary slack + ellipsis
        assert result.endswith(" …")

    def test_tool_path_returns_full_answer(self):
        """Explicit tool call (apply_injection_cap=False) returns the full answer."""
        answer = "fact " * 200  # 1000 chars, well over the 600-char cap
        mgr = self._manager_with_long_answer(answer)

        result = mgr.dialectic_query("test", "summarize", apply_injection_cap=False)

        assert result == answer
        assert not result.endswith(" …")


# ---------------------------------------------------------------------------


def _settle_prewarm(provider):
    """Wait for the session-start prewarm dialectic thread, then return the
    provider to a clean 'nothing fired yet' state so cadence/first-turn/
    trivial-prompt tests can assert from a known baseline."""
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=3.0)
    with provider._prefetch_lock:
        provider._prefetch_result = ""
        provider._prefetch_result_fired_at = -999
    provider._prefetch_thread = None
    provider._prefetch_thread_started_at = 0.0
    provider._last_dialectic_turn = -999
    provider._dialectic_empty_streak = 0
    if getattr(provider, "_manager", None) is not None:
        try:
            provider._manager.dialectic_query.reset_mock()
            provider._manager.prefetch_context.reset_mock()
        except AttributeError:
            pass


class TestTrivialPromptHeuristic:
    """Trivial prompts ('ok', 'y', slash commands) must short-circuit injection."""

    @staticmethod
    def _make_provider():
        from unittest.mock import patch, MagicMock
        from hermes_plugin_honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(api_key="test-key", enabled=True)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-trivial")
        _settle_prewarm(provider)
        return provider

    def test_classifier_catches_common_trivial_forms(self):
        for t in ("ok", "OK", " ok ", "y", "yes", "sure", "thanks", "lgtm", "/help", "", "   "):
            assert HonchoMemoryProvider._is_trivial_prompt(t), f"expected trivial: {t!r}"

    def test_classifier_catches_greetings(self):
        """Greeting words must register as trivial so context injection is skipped."""
        for t in ("hi", "HI", "hey", "hello", "yo", "sup", " hi ", "hey!", "hello."):
            assert HonchoMemoryProvider._is_trivial_prompt(t), f"expected trivial: {t!r}"

    def test_prefetch_skips_on_trivial_prompt(self):
        provider = self._make_provider()
        provider._session_key = "test"
        provider._last_dialectic_turn = 0
        provider._turn_count = 5

        assert provider.prefetch("ok") == ""
        assert provider.prefetch("/help") == ""
        # Dialectic should not have fired
        assert provider._manager.dialectic_query.call_count == 0


    def test_trivial_prompt_injects_ready_pending_dialectic(self):
        """A trivial turn consumes a ready result without starting new work."""
        provider = self._make_provider()
        provider._session_key = "test"
        provider._turn_count = 2
        # Simulate: queue_prefetch fired the dialectic at end of turn 1.
        provider._last_dialectic_turn = 1
        with provider._prefetch_lock:
            provider._prefetch_result = "PENDING_DIALECTIC"
            provider._prefetch_result_fired_at = 1

        injected = provider.prefetch("ok")

        assert "PENDING_DIALECTIC" in injected
        # And it was consumed, not left to go stale.
        with provider._prefetch_lock:
            assert provider._prefetch_result == ""

    def test_trivial_prompt_discards_stale_pending_dialectic(self):
        """A pending result older than cadence × multiplier must still be
        discarded on a trivial turn — the fix must not resurrect stale content."""
        provider = self._make_provider()
        provider._session_key = "test"
        provider._last_dialectic_turn = 1
        provider._turn_count = 1 + 4 * provider._STALE_RESULT_MULTIPLIER + 1  # 10 → stale
        with provider._prefetch_lock:
            provider._prefetch_result = "STALE_DIALECTIC"
            provider._prefetch_result_fired_at = 1

        injected = provider.prefetch("ok")

        assert injected == ""
        with provider._prefetch_lock:
            assert provider._prefetch_result == ""


class TestDialecticCadenceAdvancesOnSuccess:
    """Cadence tracker advances only when the dialectic call returns a
    non-empty result. Empty results (transient API error, sparse representation)
    must retry on the next eligible turn instead of waiting the full cadence."""

    @staticmethod
    def _make_provider():
        from unittest.mock import patch, MagicMock
        from hermes_plugin_honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(api_key="test-key", enabled=True)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-retry")
        _settle_prewarm(provider)
        return provider


    def test_non_empty_dialectic_result_advances_cadence(self):
        provider = self._make_provider()
        provider._session_key = "test"
        provider._manager.dialectic_query.return_value = "real synthesis output"
        provider._turn_count = 5
        provider._last_dialectic_turn = 0

        provider.queue_prefetch("what changed in the repo today")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)

        assert provider._last_dialectic_turn == 5

    def test_in_flight_thread_is_not_stacked(self):
        import threading as _threading
        import time as _time
        provider = self._make_provider()
        provider._session_key = "test"
        provider._turn_count = 10
        provider._last_dialectic_turn = 0

        # Simulate a prior thread still running (fresh, not stale)
        hold = _threading.Event()

        def _block():
            hold.wait(timeout=5.0)

        fresh = _threading.Thread(target=_block, daemon=True)
        fresh.start()
        provider._prefetch_thread = fresh
        provider._prefetch_thread_started_at = _time.monotonic()  # fresh start

        provider.queue_prefetch("what changed in the repo today")
        # Should have short-circuited — no new dialectic call
        assert provider._manager.dialectic_query.call_count == 0
        hold.set()
        fresh.join(timeout=2.0)


class TestSessionStartDialecticPrewarm:
    """Session-start prewarm fires a depth-aware dialectic whose result is
    consumed by turn 1 — no duplicate .chat() and no dead-cache orphaning."""

    @staticmethod
    def _make_provider(cfg_extra=None, dialectic_result="prewarm synthesis"):
        from unittest.mock import patch, MagicMock
        from hermes_plugin_honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True)
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None
        mock_manager.dialectic_query.return_value = dialectic_result

        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-prewarm")
        return provider

    def test_prewarm_populates_prefetch_result(self):
        p = self._make_provider()
        # Wait for prewarm thread to land
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=3.0)
        with p._prefetch_lock:
            assert p._prefetch_result == "prewarm synthesis"
        assert p._last_dialectic_turn == 0


    def test_turn1_consumes_prewarm_without_duplicate_dialectic(self):
        """With prewarm result already in _prefetch_result, turn 1 prefetch
        should NOT fire another dialectic."""
        p = self._make_provider()
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=3.0)
        p._manager.dialectic_query.reset_mock()
        p._session_key = "test-prewarm"
        p._turn_count = 1

        result = p.prefetch("hello world")
        assert "prewarm synthesis" in result
        # The sync first-turn path must NOT have fired another .chat()
        assert p._manager.dialectic_query.call_count == 0


class TestDialecticLiveness:
    """Liveness + observability: stale-thread recovery, stale-result discard,
    empty-streak backoff, and the snapshot method used for diagnostics."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from hermes_plugin_honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True, timeout=2.0)
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None
        mock_manager.dialectic_query.return_value = ""  # default: silent

        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-liveness")
        _settle_prewarm(provider)
        return provider

    def test_stale_thread_is_treated_as_dead(self):
        """A thread older than timeout × multiplier no longer blocks new fires."""
        import threading as _threading
        p = self._make_provider()
        p._session_key = "test"
        p._turn_count = 10
        p._last_dialectic_turn = 0
        p._manager.dialectic_query.return_value = "fresh synthesis"

        # Plant an alive thread with an old timestamp (stale)
        hold = _threading.Event()
        stuck = _threading.Thread(target=lambda: hold.wait(timeout=10.0), daemon=True)
        stuck.start()
        p._prefetch_thread = stuck
        # timeout=2.0, multiplier=2.0, so anything older than 4s is stale
        p._prefetch_thread_started_at = 0.0  # very old (1970 monotonic baseline)

        p.queue_prefetch("what changed in the repo today")
        # New thread should have been spawned since stuck one is stale
        assert p._prefetch_thread is not stuck, "stale thread must be recycled"
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=2.0)
        assert p._manager.dialectic_query.call_count == 1
        hold.set()
        stuck.join(timeout=2.0)


    def test_empty_streak_widens_effective_cadence(self):
        """After N empty returns, the gate waits cadence + N turns."""
        p = self._make_provider()
        p._dialectic_empty_streak = 3
        # cadence=1, streak=3 → effective = 4
        assert p._effective_cadence() == 4


class TestDialecticLifecycleSmoke:
    """End-to-end smoke walking a multi-turn session through prewarm,
    turn 1 consume, trivial skip, cadence fire, empty-result retry,
    heuristic bump, and session-end flush."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from hermes_plugin_honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True)
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None

        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            return provider, mock_manager, cfg

    def _await_thread(self, provider):
        """Wait up to 30 seconds and fail clearly if background work hangs."""
        thread = provider._prefetch_thread
        if thread is None:
            return
        deadline = time.monotonic() + 30.0
        while thread.is_alive() and time.monotonic() < deadline:
            thread.join(timeout=1.0)
        assert not thread.is_alive(), (
            "prefetch/prewarm thread did not finish within 30s — "
            "this is a real hang, not a timing flake"
        )

    def test_full_multi_turn_session(self):
        """Walks init → turns 1..6 → session end. Asserts at every step that
        the plugin did exactly what it should and nothing more: a dialectic
        on every non-trivial turn, the empty-result backoff, and the retry."""
        from unittest.mock import patch, MagicMock
        provider, mgr, cfg = self._make_provider()

        # Program the dialectic responses in the exact order they'll be requested.
        # An extra or missing call fails the test — strong smoke signal.
        responses = iter([
            "prewarm: user is eri, works on hermes",      # session-start prewarm
            "turn 4: long query synthesis",               # turn 4 queue_prefetch
            "",                                           # turn 5 fire: silent failure
            "retry success: fresh synthesis",             # turn 6 queue_prefetch retry
        ])
        mgr.dialectic_query.side_effect = lambda *a, **kw: next(responses)

        # ---- init: prewarm fires ----
        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mgr), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="smoke-test")
            provider._init_thread.join(timeout=5)

        self._await_thread(provider)
        with provider._prefetch_lock:
            assert provider._prefetch_result.startswith("prewarm"), \
                "session-start prewarm must land in _prefetch_result"
        assert provider._last_dialectic_turn == 0, "prewarm marks turn 0"
        assert mgr.dialectic_query.call_count == 1

        # ---- turn 1: consume prewarm, no duplicate dialectic ----
        provider.on_turn_start(1, "hey")
        inject1 = provider.prefetch("hey")
        assert "prewarm" in inject1, "turn 1 must surface prewarm"
        provider.sync_turn("hey", "hi there")
        provider.queue_prefetch("hey")  # trivial: no fire
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 1, "turn 1 must not fire — prewarm covered it"

        # ---- turn 2: trivial 'ok' → skip everything ----
        provider.on_turn_start(2, "ok")
        assert provider.prefetch("ok") == "", "trivial prompt must short-circuit injection"
        provider.sync_turn("ok", "cool")
        provider.queue_prefetch("ok")
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 1, "trivial must not fire dialectic"

        # ---- turn 3: slash '/help' → also skip ----
        provider.on_turn_start(3, "/help")
        assert provider.prefetch("/help") == ""
        provider.queue_prefetch("/help")
        assert mgr.dialectic_query.call_count == 1

        # ---- turn 4: long query → fires + heuristic bumps ----
        long_q = "walk me through " + ("x " * 100)  # ~200 chars → heuristic +1
        provider.on_turn_start(4, long_q)
        provider.prefetch(long_q)
        provider.sync_turn(long_q, "sure")
        provider.queue_prefetch(long_q)
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 2, "turn 4 fires"
        _, kwargs = mgr.dialectic_query.call_args
        assert kwargs.get("reasoning_level") in {"medium", "high"}, \
            f"long query must bump reasoning level above 'low'; got {kwargs.get('reasoning_level')}"
        assert provider._last_dialectic_turn == 4, "cadence tracker advances on success"

        # ---- turn 5: fires but silent failure (empty dialectic) ----
        provider.on_turn_start(5, "and then what")
        provider.queue_prefetch("and then what")
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 3, "turn 5 fires"
        assert provider._last_dialectic_turn == 4, "silent failure must NOT burn the cadence window"
        assert provider._effective_cadence() == 2, "an empty answer widens the cadence by one turn"

        # ---- turn 6: retries once the widened cadence allows ----
        provider.on_turn_start(6, "try again")
        provider.queue_prefetch("try again")  # (6-4) >= 2 → fires again
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 4, "turn 6 retries"
        assert provider._last_dialectic_turn == 6, "retry success advances"

        # ---- session end: flush messages ----
        provider.on_session_end([])
        mgr.flush_all.assert_called()


class TestReasoningHeuristic:
    """Char-count heuristic that scales the auto-injected reasoning level by
    query length, clamped at reasoning_level_cap."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from hermes_plugin_honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True)
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-heuristic")
        _settle_prewarm(provider)
        return provider



# ---------------------------------------------------------------------------
# set_peer_card None guard
# ---------------------------------------------------------------------------


class TestSetPeerCardNoneGuard:
    """set_peer_card must return None (not raise) when peer ID cannot be resolved."""

    def _make_manager(self):
        from hermes_plugin_honcho.client import HonchoClientConfig
        from hermes_plugin_honcho.session import HonchoSessionManager

        cfg = HonchoClientConfig(api_key="test-key", enabled=True)
        mgr = HonchoSessionManager.__new__(HonchoSessionManager)
        mgr._cache = {}
        mgr._cache_lock = threading.RLock()
        mgr._sessions_cache = {}
        mgr._config = cfg
        mgr._session_observation = {}
        return mgr

    def test_returns_none_when_peer_resolves_to_none(self):
        """set_peer_card returns None when _resolve_peer_id returns None."""
        from unittest.mock import patch
        mgr = self._make_manager()

        session = HonchoSession(
            key="test",
            honcho_session_id="sid",
            user_peer_id="user-peer",
            assistant_peer_id="ai-peer",
        )
        mgr._cache["test"] = session

        with patch.object(mgr, "_resolve_peer_id", return_value=None):
            result = mgr.set_peer_card("test", ["fact 1", "fact 2"], peer="ghost")

        assert result is None


# ---------------------------------------------------------------------------
# get_session_context cache-miss fallback respects peer param

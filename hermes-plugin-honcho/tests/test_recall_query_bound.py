"""The auto-recall ``search_query`` is bounded at the caller, not by the embedder.

The provider passes the user's whole turn to Honcho's ``/context`` as ``search_query``. Honcho
embeds that query with the deployment's embedding model, and on a self-hosted stack that endpoint
is frequently narrower than the turn: measured 2026-09-15 with ``all-minilm:l6-v2`` behind an
ollama OpenAI-compatible endpoint, a 1,400-char prose query and a 6,000-char one returned
byte-identical vectors (the endpoint reads 256 tokens; 1,200 chars is 253, 1,400 is 256) while the
call still billed a 9,778-token embed. The recalled query is a retrieval key, not a transcript, so
it is bounded once in ``get_prefetch_context`` — which covers both the shared path and
``recallSync``'s ``current_query_only`` path that calls ``peer.context()`` directly.
"""

from types import SimpleNamespace

from hermes_honcho.session import HonchoSession, HonchoSessionManager


class _FakeSummary:
    content = "summary"


class _FakeContext:
    summary = _FakeSummary()
    peer_representation = "representation"
    peer_card = ["fact"]
    messages = []


class _RecordingHonchoSession:
    def __init__(self):
        self.calls = []

    def context(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeContext()


class _RecordingPeer:
    """Records the kwargs of every ``peer.context()`` call instead of hitting Honcho."""

    def __init__(self):
        self.calls = []

    def context(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeContext()

    def representation(self, **kwargs):  # pragma: no cover - only used when the context is empty
        return "representation"

    def get_card(self, **kwargs):  # pragma: no cover - only used when the context card is empty
        return ["fact"]


def _manager_with_cached_session(*, ai_observe_others=True, recall_max_query_chars=1000):
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=ai_observe_others,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
        recall_max_query_chars=recall_max_query_chars,
    )
    mgr = HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)
    session = HonchoSession(
        key="test-session",
        user_peer_id="chris",
        assistant_peer_id="hermes",
        honcho_session_id="test-session",
    )
    fake_honcho_session = _RecordingHonchoSession()
    mgr._cache[session.key] = session
    mgr._sessions_cache[session.honcho_session_id] = fake_honcho_session
    return mgr, fake_honcho_session


def _manager_with_recording_peer(*, recall_max_query_chars=1000):
    mgr, _session = _manager_with_cached_session(recall_max_query_chars=recall_max_query_chars)
    peer = _RecordingPeer()
    mgr._get_or_create_peer = lambda peer_id: peer
    return mgr, peer


def _recall_query_emitted(mgr, peer, message, *, current_query_only=False):
    """The ``search_query`` the provider actually puts on the wire for one prefetch."""
    mgr.get_prefetch_context("test-session", message, current_query_only=current_query_only)
    queries = [c["search_query"] for c in peer.calls if "search_query" in c]
    assert len(queries) == 1, peer.calls
    return queries[0]


def _huge_turn(words=7000):
    """A pasted-turn-sized message: 35,000 chars of whole words."""
    return "word " * words


# ---------------------------------------------------------------------------
# The whole turn never reaches Honcho as the recall query
# ---------------------------------------------------------------------------


def test_recall_query_is_bounded_to_the_embedders_window():
    """A whole pasted turn must not reach Honcho as the recall query: the stack's embedding
    endpoint reads at most 256 tokens (~1,400 chars of prose) and silently drops the rest, so a
    33 KB query bought a vector of its head at the price of a 9,778-token embed call."""
    from hermes_honcho.session_context import _MAX_RECALL_QUERY_CHARS

    mgr, peer = _manager_with_recording_peer()

    message = _huge_turn()
    query = _recall_query_emitted(mgr, peer, message)

    assert len(message) == 35000
    assert len(query) <= _MAX_RECALL_QUERY_CHARS < len(message)
    # The head is kept verbatim, cut on a word boundary (the dialectic cap's convention).
    assert message.startswith(query) and query.endswith("word")
    assert message[len(query)] == " "


def test_recall_query_is_bounded_on_the_current_query_only_path():
    """recallSync's bounded current-query recall calls peer.context() directly, bypassing
    _fetch_peer_context — it needs the same bound."""
    mgr, peer = _manager_with_recording_peer()

    message = _huge_turn()
    query = _recall_query_emitted(mgr, peer, message, current_query_only=True)

    assert len(query) <= 1000 < len(message)
    assert message.startswith(query)


def test_short_recall_query_is_sent_verbatim():
    mgr, peer = _manager_with_recording_peer()

    assert _recall_query_emitted(mgr, peer, "work kanban task t_c67dbc89") == "work kanban task t_c67dbc89"


def test_empty_recall_query_is_none_not_empty_string():
    """An empty turn carries no recall key: the bounded query reads None (semantic search off),
    never an empty string that a backend could treat as "match everything"."""
    mgr, peer = _manager_with_recording_peer()

    query = _recall_query_emitted(mgr, peer, "", current_query_only=True)

    assert query is None

    # The default branch omits the parameter entirely rather than sending a blank query.
    mgr.get_prefetch_context("test-session", "")
    assert all("search_query" not in call for call in peer.calls[1:])


# ---------------------------------------------------------------------------
# The bound itself
# ---------------------------------------------------------------------------


def test_recall_query_cap_comes_from_config():
    """``recallMaxQueryChars`` sets the bound, and 0 disables it — the same convention the
    dialectic cap uses. The unset default must stay the module constant's value."""
    from hermes_honcho.client import HonchoClientConfig
    from hermes_honcho.session_context import _MAX_RECALL_QUERY_CHARS

    message = _huge_turn()

    mgr, peer = _manager_with_recording_peer(recall_max_query_chars=200)
    query = _recall_query_emitted(mgr, peer, message)
    assert len(query) <= 200 < len(message)
    assert message.startswith(query)

    uncapped, uncapped_peer = _manager_with_recording_peer(recall_max_query_chars=0)
    assert _recall_query_emitted(uncapped, uncapped_peer, message) == message.strip()

    assert HonchoClientConfig().recall_max_query_chars == _MAX_RECALL_QUERY_CHARS


def test_bound_recall_query_handles_blank_and_disabled_cap():
    from hermes_honcho.session_context import bound_recall_query

    assert bound_recall_query(None) is None
    assert bound_recall_query("   ") is None
    assert bound_recall_query("short query") == "short query"
    assert bound_recall_query("  padded  ") == "padded"
    # max_chars <= 0 disables the cap rather than truncating everything away.
    assert bound_recall_query("x" * 5000, 0) == "x" * 5000


def test_bound_recall_query_keeps_an_unbroken_token_whole(caplog):
    """With no space inside the window the trim keeps the slice (never an empty query)."""
    import logging

    from hermes_honcho.session_context import bound_recall_query

    with caplog.at_level(logging.DEBUG, logger="plugins.memory.honcho.session"):
        assert bound_recall_query("a" * 4000) == "a" * 1000

    assert "Honcho recall query truncated for embedding: 4000 -> 1000 chars" in caplog.text

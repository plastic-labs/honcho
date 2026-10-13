"""The dialectic caller consumes honcho's tool-loop signal (``iterations`` / ``capped_out``).

A deployment that reports the tool loop answers the chat route with ``{"content", "iterations",
"capped_out"}``. Before this change the plugin could not see either key: it called the SDK's
``Peer.chat()``, which returns ``data.get("content")`` and drops the rest, so a capped-out
(budget-exhausted, tool-less synthesis) answer was indistinguishable from a complete one.

The fakes below are shaped like the real seam: ``_FakePeer.chat()`` returns content only, while
``_FakePeer._honcho._http.post()`` (the route the SDK itself walks) carries the diagnostics. A
test that observes the new behaviour therefore proves the plugin went through the HTTP layer, not
through the lossy return type.

Absent keys mean *unknown*, never "not capped": a server that reports neither keeps the previous
behaviour exactly, which the control cases below pin down.
"""

import json
import logging
from unittest.mock import MagicMock

from hermes_honcho.client import HonchoClientConfig
from hermes_honcho.session import HonchoSession, HonchoSessionManager

_LOGGER_NAME = "plugins.memory.honcho.session"
_CHAT_PATH = "/v3/workspaces/hermes/peers/chris/chat"


class _FakeHTTP:
    """Stands in for honcho's ``HonchoHTTPClient``: records bodies, replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, path, *, body=None, **kwargs):
        self.calls.append({"path": path, "body": body, **kwargs})
        return self.responses.pop(0) if self.responses else None


class _FakeHonchoClient:
    def __init__(self, responses):
        self.workspace_id = "hermes"
        self.workspace_ensured = False
        self._http = _FakeHTTP(responses)

    def _ensure_workspace(self):  # pragma: no cover - asserted through the recorded calls
        self.workspace_ensured = True


class _FakePeer:
    """A peer as the SDK exposes it: ``chat()`` is lossy, ``_honcho._http`` is not."""

    def __init__(self, responses, *, content="dialectic answer"):
        self.id = "chris"
        self.workspace_id = "hermes"
        self._honcho = _FakeHonchoClient(responses)
        self._content = content
        self.chat_calls = []

    def chat(self, query, **kwargs):
        self.chat_calls.append({"query": query, **kwargs})
        return self._content


def _manager(*responses, retry_on_capped=False, max_chars=600, level="low", content="dialectic answer"):
    cfg = HonchoClientConfig(dialectic_max_chars=max_chars, dialectic_retry_on_capped=retry_on_capped)
    mgr = HonchoSessionManager(config=cfg)
    mgr._dialectic_reasoning_level = level
    mgr._dialectic_retry_on_capped = retry_on_capped
    mgr._dialectic_max_chars = max_chars
    session = HonchoSession(key="test", user_peer_id="chris", assistant_peer_id="hermes", honcho_session_id="s")
    mgr._cache["test"] = session
    peer = _FakePeer(list(responses), content=content)
    mgr._get_or_create_peer = lambda peer_id: peer
    return mgr, peer


def _warnings(caplog):
    return [r.message for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# The signal is read, and a truncated answer is distinguishable from a complete one
# ---------------------------------------------------------------------------


def test_capped_out_answer_is_read_off_the_response_and_logged(caplog):
    mgr, peer = _manager({"content": "a synthesis", "iterations": 3, "capped_out": True})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "what do you know?")

    assert result == "a synthesis"
    said = " | ".join(_warnings(caplog))
    assert "truncated" in said and "iterations=3" in said and "level=low" in said
    # The diagnostics came from the route the SDK drops them on, not from chat()'s return.
    assert [c["path"] for c in peer._honcho._http.calls] == [_CHAT_PATH]
    assert peer.chat_calls == []


def test_complete_answer_is_not_flagged_and_is_only_reported_when_unknown(caplog):
    mgr, peer = _manager({"content": "a synthesis", "iterations": 1, "capped_out": False})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q")

    assert result == "a synthesis"
    assert _warnings(caplog) == []


def test_control_arm_without_the_field_is_silent_and_unmarked(caplog):
    """A body with content only must not be guessed at: absent means unknown, not capped."""
    mgr, peer = _manager({"content": "a synthesis"})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q", mark_truncated=True)

    assert result == "a synthesis"
    assert _warnings(caplog) == []
    assert len(peer._honcho._http.calls) == 1


def test_explicit_tool_call_marks_a_truncated_answer_only_when_capped(caplog):
    mgr, peer = _manager({"content": "a synthesis", "iterations": 3, "capped_out": True})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        capped = mgr.dialectic_query("test", "q", apply_injection_cap=False, mark_truncated=True)

    assert capped.startswith("a synthesis")
    assert "recall truncated" in capped and "iterations=3" in capped

    clean, _ = _manager({"content": "a synthesis", "iterations": 1, "capped_out": False})
    assert clean.dialectic_query("test", "q", apply_injection_cap=False, mark_truncated=True) == "a synthesis"


def test_auto_injected_cap_still_trims_and_the_marker_survives_it(caplog):
    """The injection cap clips the answer, never the truncation marker (which is appended after)."""
    mgr, peer = _manager({"content": "fact " * 100, "iterations": 3, "capped_out": True}, max_chars=50)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q", mark_truncated=True)

    assert result.startswith("fact fact")
    assert result.splitlines()[0].endswith(" …")
    assert "recall truncated" in result


# ---------------------------------------------------------------------------
# The opt-in single retry: one escalation, never a loop
# ---------------------------------------------------------------------------


def test_retry_escalates_one_level_and_uses_the_better_answer(caplog):
    mgr, peer = _manager(
        {"content": "capped synthesis", "iterations": 3, "capped_out": True},
        {"content": "fuller synthesis", "iterations": 1, "capped_out": False},
        retry_on_capped=True,
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q")

    assert result == "fuller synthesis"
    bodies = [c["body"] for c in peer._honcho._http.calls]
    assert [b["reasoning_level"] for b in bodies] == ["low", "medium"]
    assert all(b["stream"] is False for b in bodies)
    assert any("re-asking once at level=medium" in m for m in _warnings(caplog))


def test_retry_is_bounded_to_one_extra_request_when_it_also_caps_out(caplog):
    mgr, peer = _manager(
        {"content": "capped synthesis", "iterations": 3, "capped_out": True},
        {"content": "still capped", "iterations": 4, "capped_out": True},
        retry_on_capped=True,
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q")

    assert result == "still capped"
    assert len(peer._honcho._http.calls) == 2  # never a third request
    assert any("not retrying again" in m for m in _warnings(caplog))


def test_retry_at_the_ceiling_spends_nothing(caplog):
    mgr, peer = _manager(
        {"content": "capped synthesis", "iterations": 6, "capped_out": True}, retry_on_capped=True, level="max"
    )

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q")

    assert result == "capped synthesis"
    assert len(peer._honcho._http.calls) == 1
    assert any("no higher reasoning level" in m for m in _warnings(caplog))


def test_retry_off_by_default_keeps_the_single_pass(caplog):
    """Default config: the truncation is reported, and no second request is spent."""
    mgr, peer = _manager({"content": "capped synthesis", "iterations": 3, "capped_out": True})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = mgr.dialectic_query("test", "q")

    assert result == "capped synthesis"
    assert len(peer._honcho._http.calls) == 1
    assert any("truncated" in m for m in _warnings(caplog))


def test_config_field_reaches_the_manager():
    """dialecticRetryOnCapped is a real config key, not just a call-site argument."""
    from hermes_honcho.config_schema import CONFIG_SCHEMA

    field = next(f for f in CONFIG_SCHEMA.fields if f.key == "dialecticRetryOnCapped")
    assert field.default == "false"

    cfg = HonchoClientConfig(dialectic_retry_on_capped=True)
    assert cfg.dialectic_retry_on_capped is True
    assert HonchoSessionManager(config=cfg)._dialectic_retry_on_capped is True
    assert HonchoSessionManager(config=HonchoClientConfig())._dialectic_retry_on_capped is False


def test_per_call_argument_overrides_the_config():
    mgr, peer = _manager({"content": "capped synthesis", "iterations": 3, "capped_out": True}, retry_on_capped=True)

    result = mgr.dialectic_query("test", "q", retry_on_capped=False)

    assert result == "capped synthesis"
    assert len(peer._honcho._http.calls) == 1


# ---------------------------------------------------------------------------
# The seam: no SDK internals, no signal — and recall still works
# ---------------------------------------------------------------------------


def test_missing_sdk_internals_fall_back_to_peer_chat():
    """A peer whose HTTP layer is unreachable degrades to the pre-change call, not to an error."""

    class _BarePeer:
        def __init__(self):
            self.calls = []

        def chat(self, query, **kwargs):
            self.calls.append({"query": query, **kwargs})
            return "fallback answer"

    cfg = HonchoClientConfig()
    mgr = HonchoSessionManager(config=cfg)
    mgr._dialectic_reasoning_level = "low"
    session = HonchoSession(key="test", user_peer_id="chris", assistant_peer_id="hermes", honcho_session_id="s")
    mgr._cache["test"] = session
    peer = _BarePeer()
    mgr._get_or_create_peer = lambda peer_id: peer

    assert mgr.dialectic_query("test", "q") == "fallback answer"
    assert peer.calls == [{"query": "q", "target": "chris", "reasoning_level": "low"}]


# ---------------------------------------------------------------------------
# The agent-visible caller: the honcho_reasoning tool
# ---------------------------------------------------------------------------


def test_explicit_tool_call_passes_the_marker_flag_to_the_manager():
    """The honcho_reasoning tool is the caller that asks for the agent-visible marker."""
    from hermes_honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    provider._manager = MagicMock()
    provider._manager.dialectic_query.return_value = "answer"
    provider._session_key = "test"

    out = json.loads(provider._tool_reasoning({"query": "who is this?"}))

    assert out["result"] == "answer"
    assert provider._manager.dialectic_query.call_args.kwargs["mark_truncated"] is True


def test_truncated_answer_reaches_the_agent_marked_end_to_end(caplog):
    """provider tool -> real manager -> stub SDK: the truncated answer arrives marked."""
    from hermes_honcho import HonchoMemoryProvider

    mgr, peer = _manager({"content": "capped synthesis", "iterations": 3, "capped_out": True})
    provider = HonchoMemoryProvider()
    provider._manager = mgr
    provider._session_key = "test"

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        out = json.loads(provider._tool_reasoning({"query": "who is this?"}))

    assert out["result"].startswith("capped synthesis")
    assert "recall truncated" in out["result"]
    assert any("truncated" in m for m in _warnings(caplog))
    assert peer.chat_calls == []


def test_diagnostics_never_break_recall_when_the_http_layer_raises():
    class _BoomHTTP:
        def post(self, path, **kwargs):
            raise RuntimeError("connection reset")

    class _FakeClient:
        workspace_id = "hermes"
        _http = _BoomHTTP()

        def _ensure_workspace(self):
            return None

    class _Peer:
        id = "chris"
        workspace_id = "hermes"
        _honcho = _FakeClient()

        def chat(self, query, **kwargs):
            return "fallback answer"

    mgr = HonchoSessionManager(config=HonchoClientConfig())
    mgr._dialectic_reasoning_level = "low"
    mgr._cache["test"] = HonchoSession(
        key="test", user_peer_id="chris", assistant_peer_id="hermes", honcho_session_id="s"
    )
    mgr._get_or_create_peer = lambda peer_id: _Peer()

    assert mgr.dialectic_query("test", "q") == "fallback answer"

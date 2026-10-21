"""Blocking dialectic mode: one dialectic per turn over the current request and recent exchanges.

Hermes abandons a provider prefetch after a few seconds, so the dialectic runs in on_turn_start
and prefetch() only hands back its result. A resumed Hermes process starts each turn fresh, so the
recent exchanges live in a file under HERMES_HOME that sync_turn writes and the next turn reads.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from agent.memory_manager import MemoryManager

from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_plugin_honcho.dialectic import DEFAULT_BLOCKING_RESULT_TEMPLATE
from hermes_plugin_honcho.recent_turns import extract_exchange, load_exchanges, ring_path, summarize_tool_calls

ANSWER = "They want every change covered by a test before review."


class BlockingManager:
    """Records every Honcho call a provider makes."""

    def __init__(self, answer: str = ANSWER):
        self.answer = answer
        self.dialectic_calls: list[tuple[str, str, dict]] = []
        self.context_calls: list[str] = []

    def dialectic_query(self, session_key, query, **kwargs):
        self.dialectic_calls.append((session_key, query, kwargs))
        return self.answer

    def get_prefetch_context(self, session_key, query=None, **_kwargs):
        self.context_calls.append(f"get:{query}")
        return {"representation": "BASE LAYER"}

    def prefetch_context(self, session_key, query):
        self.context_calls.append(f"queue:{query}")

    def pop_context_result(self, _session_key):
        return {}

    def resolve_author_peer_id(self, *_args, **_kwargs):
        return None

    def get_or_create(self, *_args, **_kwargs):
        return MagicMock(messages=[])

    def save(self, _session):
        pass


def make_provider(tmp_path, *, mode="blocking", session_key="session-a", manager=None, **options):
    """A ready provider whose recent-turns file lives in ``tmp_path``, like a fresh process each call."""
    cfg = HonchoClientConfig(api_key="k", enabled=True, recall_mode="context", hermes_home=tmp_path,
                             dialectic_injection_mode=mode, dialectic_reasoning_level="high", **options)
    provider = HonchoMemoryProvider()
    provider._config = cfg
    provider._manager = manager or BlockingManager()
    provider._session_key = session_key
    provider._session_initialized = True
    provider._recall_mode = cfg.recall_mode
    provider._dialectic_injection_mode = cfg.dialectic_injection_mode
    provider._injection_log_path = str(tmp_path / "injection.log")
    return provider


def tool_turn(user: str, reply: str) -> list[dict]:
    """A transcript ending in one turn where the agent ran two tools."""
    return [
        {"role": "user", "content": "an older request"},
        {"role": "assistant", "content": "older reply"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "pytest -q"}'}},
            {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "3 passed"},
        {"role": "tool", "tool_call_id": "c2", "content": "def a():\n    return 1"},
        {"role": "assistant", "content": reply},
    ]


def log_records(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "injection.log").read_text().splitlines()]


def test_recent_exchanges_survive_a_process_restart(tmp_path):
    first = make_provider(tmp_path)
    first.on_turn_start(1, "Write the parser.")
    assert first.prefetch("Write the parser.") == DEFAULT_BLOCKING_RESULT_TEMPLATE.replace("{dialectic_output}", ANSWER)
    first.sync_turn("Write the parser.", "Parser written.", messages=tool_turn("Write the parser.", "Parser written."))

    second = make_provider(tmp_path)  # the next turn runs in a new process
    second.on_turn_start(2, "Now add error handling.")
    injected = second.prefetch("Now add error handling.")

    (_key, query, kwargs) = second._manager.dialectic_calls[0]
    assert "User: Write the parser." in query
    assert '- terminal({"command": "pytest -q"}) → 3 passed' in query
    assert '- read_file({"path": "a.py"}) → def a(): return 1' in query
    assert "Assistant: Parser written." in query
    assert "an older request" not in query
    assert query.rstrip().endswith("Now add error handling.")
    assert kwargs == {"reasoning_level": "high", "peer": "user", "apply_injection_cap": False, "raise_errors": True,
                      "stream": True}
    assert ANSWER in injected

    record = log_records(tmp_path)[-1]
    assert record["reason"] == "injected" and record["mode"] == "blocking"
    assert record["query"] == query and isinstance(record["dialectic_ms"], int)
    assert record["dialectic_output"] == ANSWER
    assert record["recent_turns_count"] == 1 and record["dialectic_reason"] == "answered"
    assert record["payload"] == injected


def test_ring_keeps_the_last_three_exchanges_per_session(tmp_path):
    for n in range(5):
        make_provider(tmp_path).sync_turn(f"request {n}", f"reply {n}")
    make_provider(tmp_path, session_key="session-b").sync_turn("other session", "other reply")

    path = ring_path(tmp_path)
    assert [t["user"] for t in load_exchanges(path, "session-a")] == ["request 2", "request 3", "request 4"]
    assert [t["user"] for t in load_exchanges(path, "session-b")] == ["other session"]


def test_sync_turn_without_messages_records_the_exchange_without_tools(tmp_path):
    make_provider(tmp_path).sync_turn("Plan the release.", "Here is the plan.")

    (exchange,) = load_exchanges(ring_path(tmp_path), "session-a")
    assert (exchange["user"], exchange["assistant"], exchange["tools"]) == ("Plan the release.", "Here is the plan.", [])


def test_hermes_passes_the_transcript_to_sync_turn(tmp_path):
    provider = make_provider(tmp_path)
    memory = MemoryManager()
    memory.add_provider(provider)
    try:
        memory.sync_all("Write the parser.", "Parser written.", messages=tool_turn("Write the parser.", "Parser written."))
        assert memory.flush_pending(timeout=5.0)
    finally:
        memory.shutdown_all()

    (exchange,) = load_exchanges(ring_path(tmp_path), "session-a")
    assert exchange["tools"][0].startswith("terminal(")


def test_tool_summary_truncates_and_caps():
    calls = [{"id": f"c{i}", "function": {"name": "terminal", "arguments": "x" * 300}} for i in range(30)]
    messages = [{"role": "assistant", "tool_calls": calls},
                {"role": "tool", "tool_call_id": "c0", "content": "y" * 500}]

    lines = summarize_tool_calls(messages)

    assert len(lines) == 26 and lines[-1] == "… 5 more"
    assert lines[0] == f"terminal({'x' * 119}…) → {'y' * 199}…"
    assert lines[1].endswith("→ (no result)")


def test_exchange_falls_back_to_the_transcript_for_the_reply():
    exchange = extract_exchange("Write the parser.", "", tool_turn("Write the parser.", "Parser written."))
    assert exchange["assistant"] == "Parser written."


def test_templates_render_literally(tmp_path):
    template = 'Q for {peer_name}: {"json": true}\n{recent_turns}\nNOW: {current_prompt}'
    provider = make_provider(tmp_path, peer_name="ada", dialectic_query_template=template,
                             dialectic_result_template="<<{dialectic_output}>>")
    provider.sync_turn("Earlier {current_prompt} request", "ok")

    provider.on_turn_start(2, "Ship {recent_turns} it.")
    assert provider.prefetch("Ship it.") == f"<<{ANSWER}>>"

    query = provider._manager.dialectic_calls[0][1]
    assert query.startswith('Q for ada: {"json": true}\n### Exchange 1 of 1\nUser: Earlier {current_prompt} request')
    assert query.endswith("NOW: Ship {recent_turns} it.")


def _three_exchanges(tmp_path, limit: int) -> HonchoMemoryProvider:
    """A provider holding three exchanges of ten 60-char tool lines, a 200-char reply and a 100-char request."""
    provider = make_provider(tmp_path, dialectic_max_input_chars=limit)
    for n in range(3):
        provider.sync_turn(f"req{n} " + "u" * 95, f"rep{n} " + "a" * 195)
        provider._turn_count = n + 1
    path = ring_path(tmp_path)
    data = json.loads(path.read_text())
    for n, turn in enumerate(data["sessions"]["session-a"]):
        turn["tools"] = [f"t{n}.{i}(" + "x" * 50 + ")" for i in range(10)]
    path.write_text(json.dumps(data))
    return provider


def _query_for(provider, prompt="CURRENT PROMPT") -> tuple[str, dict]:
    provider.on_turn_start(9, prompt)
    provider.prefetch(prompt)
    return provider._manager.dialectic_calls[-1][1], provider._blocking_injection


def test_input_cap_trims_the_oldest_exchanges_tool_lines_first(tmp_path):
    full, _ = _query_for(_three_exchanges(tmp_path / "full", limit=100_000))
    provider = _three_exchanges(tmp_path / "cut", limit=len(full) - 300)

    query, _ = _query_for(provider)

    (note,) = log_records(tmp_path / "cut")[-1]["query_trimmed"]
    dropped = int(note.removeprefix("exchange 1: ").split()[0])
    assert len(query) <= len(full) - 300 and query.endswith("CURRENT PROMPT")
    # Only the oldest exchange's last tool lines went; its text and the newer exchanges are whole.
    assert f"t0.{9 - dropped}(" in query and f"t0.{10 - dropped}(" not in query
    assert all(f"t{n}.9(" in query for n in (1, 2)) and "rep0 " in query


def test_input_cap_cuts_text_then_drops_but_keeps_the_newest_exchange(tmp_path):
    provider = _three_exchanges(tmp_path, limit=900)

    query, _ = _query_for(provider)

    assert len(query) <= 900 and query.endswith("CURRENT PROMPT")
    assert "req0 " not in query and "req1 " not in query  # older exchanges dropped after trimming
    assert "Exchange 1 of 1" in query and "req2 " in query
    trimmed = log_records(tmp_path)[-1]["query_trimmed"]
    assert trimmed[:4] == ["exchange 1: 10 tool line(s)", "exchange 1: assistant text cut by 200 chars",
                           "exchange 1: user text cut by 100 chars", "exchange 1: dropped"]
    assert "exchange 2: dropped" in trimmed and "exchange 3: dropped" not in trimmed


def test_an_oversized_prompt_loses_its_middle_not_its_end(tmp_path):
    provider = _three_exchanges(tmp_path, limit=1000)
    prompt = "HEAD " + "m" * 3000 + " TAIL"

    query, _ = _query_for(provider, prompt=prompt)

    assert len(query) <= 1000  # so the manager's own cut, which drops the end, never runs
    assert "HEAD " in query and query.endswith(" TAIL") and "chars omitted …]" in query
    record = log_records(tmp_path)[-1]
    assert record["recent_turns_count"] == 1  # the newest exchange is emptied, never dropped
    assert record["query_trimmed"][-1].startswith("current prompt: ")


def test_a_template_over_the_cap_is_reported(tmp_path, caplog):
    provider = _three_exchanges(tmp_path, limit=150)

    _query_for(provider, prompt="P" * 250)

    assert log_records(tmp_path)[-1]["query_trimmed"][-1].endswith("chars will be cut from the end")
    assert "dialecticMaxInputChars=150" in caplog.text


def test_a_template_without_the_prompt_warns_once(tmp_path, caplog):
    provider = make_provider(tmp_path, dialectic_query_template="What matters? {recent_turns}")
    for turn in (1, 2):
        _query_for(provider, prompt=f"request {turn}")
    assert caplog.text.count("has no {current_prompt}") == 1


def test_a_notice_does_not_hide_a_failed_dialectic(tmp_path):
    manager = BlockingManager()
    manager.dialectic_query = MagicMock(side_effect=RuntimeError("boom"))
    provider = make_provider(tmp_path, manager=manager)
    provider._init_auth_failure = "token expired"

    provider.on_turn_start(1, "Write the parser.")
    assert "[Honcho memory status]" in provider.prefetch("Write the parser.")

    record = log_records(tmp_path)[-1]
    assert record["reason"] == "injected" and record["dialectic_reason"] == "dialectic-failed"


def test_an_earlier_outcome_never_reaches_a_later_turn(tmp_path):
    provider = make_provider(tmp_path)
    provider._blocking_injection = {"payload": "STALE"}
    with patch.object(provider, "_query_blocking_dialectic", side_effect=RuntimeError("boom")), \
         pytest.raises(RuntimeError):
        provider.on_turn_start(2, "Write the parser.")
    assert provider.prefetch("Write the parser.") == ""


def test_context_tokens_does_not_cut_the_answer(tmp_path):
    provider = make_provider(tmp_path, context_tokens=5)
    provider.on_turn_start(1, "Write the parser.")
    assert ANSWER in provider.prefetch("Write the parser.")


def test_blocking_mode_makes_no_other_honcho_calls(tmp_path):
    provider = make_provider(tmp_path)
    for turn, prompt in enumerate(("Write the parser.", "Now add error handling."), 1):
        provider.on_turn_start(turn, prompt)
        provider.prefetch(prompt)
        provider.queue_prefetch(prompt)

    assert provider._manager.context_calls == []
    assert len(provider._manager.dialectic_calls) == 2
    assert provider._prefetch_thread is None


def test_blocking_mode_skips_the_session_init_prewarm(tmp_path):
    manager = BlockingManager()
    cfg = HonchoClientConfig(api_key="k", enabled=True, recall_mode="hybrid", hermes_home=tmp_path,
                             dialectic_injection_mode="blocking")
    provider = HonchoMemoryProvider()
    with patch("hermes_plugin_honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
         patch("hermes_plugin_honcho.client.get_honcho_client", return_value=MagicMock()), \
         patch("hermes_plugin_honcho.session.HonchoSessionManager", return_value=manager):
        provider.initialize(session_id="s")
        provider._init_thread.join(timeout=5.0)

    assert provider._session_initialized
    assert provider._prefetch_thread is None and manager.dialectic_calls == []


@pytest.mark.parametrize("answer, reason", [("", "dialectic-empty"), (None, "dialectic-failed")])
def test_no_answer_injects_nothing(tmp_path, answer, reason):
    manager = BlockingManager(answer=answer or "")
    if answer is None:
        manager.dialectic_query = MagicMock(side_effect=RuntimeError("boom"))
    provider = make_provider(tmp_path, manager=manager)

    provider.on_turn_start(1, "Write the parser.")
    assert provider.prefetch("Write the parser.") == ""

    record = log_records(tmp_path)[-1]
    assert record["reason"] == reason and "Write the parser." in record["query"]
    assert record.get("error") == (None if answer == "" else "RuntimeError: boom")


def test_trivial_prompt_skips_the_dialectic(tmp_path):
    provider = make_provider(tmp_path)
    provider.on_turn_start(1, "ok")
    assert provider.prefetch("ok") == ""
    assert provider._manager.dialectic_calls == []
    assert log_records(tmp_path)[-1]["reason"] == "trivial-prompt"


def test_default_mode_is_unchanged(tmp_path):
    assert HonchoClientConfig().dialectic_injection_mode == "async"
    provider = make_provider(tmp_path, mode="async")
    provider.on_turn_start(1, "Write the parser.")
    provider.sync_turn("Write the parser.", "Parser written.", messages=tool_turn("Write the parser.", "Parser written."))

    assert provider._manager.dialectic_calls == []
    assert not ring_path(tmp_path).exists()
    assert "BASE LAYER" in provider.prefetch("Write the parser.")
    assert "mode" not in log_records(tmp_path)[-1]


@pytest.mark.parametrize("raw, expected", [
    ({}, ("async", None, None)),
    ({"dialecticInjectionMode": "bogus"}, ("async", None, None)),
    ({"dialecticInjectionMode": "blocking", "dialecticQueryTemplate": "Q {current_prompt}",
      "dialecticResultTemplate": "R {dialectic_output}"}, ("blocking", "Q {current_prompt}", "R {dialectic_output}")),
    ({"hosts": {"hermes": {"dialecticInjectionMode": "blocking", "dialecticQueryTemplate": "  "}}},
     ("blocking", None, None)),
])
def test_config_keys(tmp_path, raw, expected):
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps({"apiKey": "k", **raw}))
    cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
    assert (cfg.dialectic_injection_mode, cfg.dialectic_query_template, cfg.dialectic_result_template) == expected


def test_a_dialectic_slower_than_the_prefetch_timeout_still_injects(tmp_path):
    manager = BlockingManager()
    answer = manager.dialectic_query

    def _slow(*args, **kwargs):
        time.sleep(0.4)
        return answer(*args, **kwargs)

    manager.dialectic_query = _slow
    memory = MemoryManager(external_prefetch_timeout=0.1)
    memory.add_provider(make_provider(tmp_path, manager=manager))

    memory.on_turn_start(1, "Write the parser.")
    assert ANSWER in memory.prefetch_all("Write the parser.")


class StreamingPeer:
    """An SDK peer whose streaming chat yields ``chunks``, then raises ``error`` if one is given."""

    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.calls = chunks, error, []

    def chat(self, *_args, **_kwargs):
        raise AssertionError("blocking mode must stream")

    def chat_stream(self, query, **kwargs):
        self.calls.append((query, kwargs))
        yield from self.chunks
        if self.error:
            raise self.error


def streaming_manager(peer):
    from hermes_plugin_honcho.session import HonchoSession, HonchoSessionManager

    mgr = HonchoSessionManager(config=HonchoClientConfig(host="hermes", api_key="k", enabled=True,
                                                         dialectic_dynamic=False, dialectic_reasoning_level="high"))
    mgr._cache["k"] = HonchoSession(key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s")
    mgr._get_or_create_peer = lambda _peer_id: peer
    return mgr


def test_streamed_dialectic_joins_the_chunks_and_asks_as_the_ai_peer():
    peer = StreamingPeer(["They want ", "tests ", "first."])
    answer = streaming_manager(peer).dialectic_query("k", "q", reasoning_level="high", apply_injection_cap=False,
                                                     raise_errors=True, stream=True)
    assert answer == "They want tests first."
    assert peer.calls == [("q", {"reasoning_level": "high", "target": "u"})]


def test_a_stream_that_breaks_midway_is_an_error_not_a_short_answer():
    import httpx

    peer = StreamingPeer(["They want "], error=httpx.RemoteProtocolError("peer closed connection"))
    with pytest.raises(httpx.RemoteProtocolError):
        streaming_manager(peer).dialectic_query("k", "q", raise_errors=True, stream=True)


def test_a_turn_hermes_never_synced_is_recorded_from_its_request(tmp_path):
    first = make_provider(tmp_path)
    first.on_turn_start(1, "Write the parser.")
    first.prefetch("Write the parser.")
    first.sync_turn("Write the parser.", "Parser written.")
    # Turn 2 ends without a reply, so Hermes never calls sync_turn for it.
    make_provider(tmp_path).on_turn_start(2, "Now add retry.", previous_message="Write the parser.")

    third = make_provider(tmp_path)
    third.on_turn_start(3, "How does the VM dispatch a panic?", previous_message="Now add retry.")
    third.prefetch("How does the VM dispatch a panic?")

    query = third._manager.dialectic_calls[0][1]
    assert query.index("User: Write the parser.") < query.index("User: Now add retry.")
    assert "Assistant: (that turn ended without a reply)" in query
    record = log_records(tmp_path)[-1]
    assert record["previous_message_stubbed"] is True and record["recent_turns_count"] == 2
    # Persisted, so the next turn quotes it as well, and it is not stubbed a second time.
    fourth = make_provider(tmp_path)
    fourth.on_turn_start(4, "Add docs.", previous_message="How does the VM dispatch a panic?")
    fourth.prefetch("Add docs.")
    users = [t["user"] for t in load_exchanges(ring_path(tmp_path), "session-a")]
    assert users == ["Write the parser.", "Now add retry.", "How does the VM dispatch a panic?"]


def test_a_synced_previous_turn_is_not_stubbed_again(tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("Write   the parser.", "Parser written.")
    provider.on_turn_start(2, "Now add retry.", previous_message="Write the parser.")
    provider.prefetch("Now add retry.")

    assert len(load_exchanges(ring_path(tmp_path), "session-a")) == 1
    record = log_records(tmp_path)[-1]
    assert "previous_message_stubbed" not in record and record["recent_turns_count"] == 1


def test_no_stub_on_the_first_turn(tmp_path):
    provider = make_provider(tmp_path)
    provider.on_turn_start(1, "Write the parser.", previous_message="something older")
    provider.prefetch("Write the parser.")
    assert not ring_path(tmp_path).exists()
    assert "previous_message_stubbed" not in log_records(tmp_path)[-1]


# The request Hermes restated after a compaction in the run that showed the false stub.
REQUEST = ("The core window helpers module (ranking/offset/aggregate helpers, frame constructors, validation) "
           "is in place now. Next piece is wiring this into the query builders.")


def _replayed(text: str) -> str:
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER

    return f"{_INFLIGHT_TASK_REPLAY_HEADER}\n{text}"


def _skill_turn(text: str) -> str:
    from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION, _SINGLE_SKILL_MARKER, _SKILL_INVOCATION_PREFIX

    return (f'{_SKILL_INVOCATION_PREFIX}"review" skill. {_SINGLE_SKILL_MARKER}\n\n# Review\nSkill body.\n\n'
            f"{_SINGLE_SKILL_INSTRUCTION}{text}")


def _stub_for(tmp_path, *, synced_turn: int, turn: int, previous: str) -> HonchoMemoryProvider:
    """A provider at ``turn`` whose ring holds REQUEST synced at ``synced_turn``, with Hermes' ``previous``."""
    writer = make_provider(tmp_path)
    writer._turn_count = synced_turn
    writer.sync_turn(REQUEST, "Wired into the query builders.")
    provider = make_provider(tmp_path)
    provider.on_turn_start(turn, "Now add a named window.", previous_message=previous)
    provider.prefetch("Now add a named window.")
    return provider


@pytest.mark.parametrize("previous", [
    _replayed(REQUEST),
    # Merged onto the compaction summary: the request follows the last header.
    "[CONTEXT SUMMARY]: earlier work ...\n--- END OF CONTEXT SUMMARY ---\n\n" + _replayed(_replayed(REQUEST)),
    _skill_turn(REQUEST),
    REQUEST[:200] + "\n\n(and keep the tests green)",
], ids=["compaction-replay", "replay-on-summary", "skill-scaffold", "longer-restatement"])
def test_a_rewritten_previous_request_is_not_stubbed(tmp_path, previous):
    # Turn numbers do not settle it: the ring entry is two turns back.
    _stub_for(tmp_path, synced_turn=2, turn=4, previous=previous)

    assert [t["user"] for t in load_exchanges(ring_path(tmp_path), "session-a")] == [REQUEST]
    assert "previous_message_stubbed" not in log_records(tmp_path)[-1]


def test_an_exchange_from_the_previous_turn_counts_as_synced(tmp_path):
    # After a compaction Hermes' turn number can fall back (turn 4 reported as 3).
    _stub_for(tmp_path, synced_turn=3, turn=3, previous="something else entirely")
    assert "previous_message_stubbed" not in log_records(tmp_path)[-1]


def test_an_unsynced_turn_after_a_compaction_is_stubbed_without_the_header(tmp_path):
    _stub_for(tmp_path, synced_turn=2, turn=4, previous=_replayed("Add the PARTITION BY clause."))

    users = [t["user"] for t in load_exchanges(ring_path(tmp_path), "session-a")]
    assert users == [REQUEST, "Add the PARTITION BY clause."]
    assert log_records(tmp_path)[-1]["previous_message_stubbed"] is True

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
    assert kwargs == {"reasoning_level": "high", "peer": "user", "apply_injection_cap": False}
    assert ANSWER in injected

    record = log_records(tmp_path)[-1]
    assert record["reason"] == "injected" and record["mode"] == "blocking"
    assert record["query"] == query and isinstance(record["dialectic_ms"], int)
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


def test_templates_render_literally_and_history_yields_to_the_input_cap(tmp_path, caplog):
    template = 'Q for {peer_name}: {"json": true}\n{recent_turns}\nNOW: {current_prompt}'
    provider = make_provider(tmp_path, peer_name="ada", dialectic_query_template=template,
                             dialectic_result_template="<<{dialectic_output}>>", dialectic_max_input_chars=300)
    for n in range(3):
        provider.sync_turn(f"request {n} " + "z" * 60, "ok")

    provider.on_turn_start(4, "Ship {recent_turns} it.")
    assert provider.prefetch("Ship it.") == f"<<{ANSWER}>>"

    query = provider._manager.dialectic_calls[0][1]
    assert query.startswith('Q for ada: {"json": true}')
    assert query.endswith("NOW: Ship {recent_turns} it.")
    assert "request 2" in query and "request 0" not in query and len(query) <= 300
    assert "dialecticMaxInputChars=300" in caplog.text


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

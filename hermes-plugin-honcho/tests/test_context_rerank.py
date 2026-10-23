"""Context-rerank mode: one sized peer.context() fetch per turn, narrowed by one LLM call to what the turn needs.

It shares the blocking mode's seam: the work runs in on_turn_start, prefetch() injects only its result,
and the recent exchanges come from the same file sync_turn writes.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_plugin_honcho.context_rerank import (
    DEFAULT_RERANK_PROMPT_TEMPLATE,
    call_rerank,
    count_conclusions,
    render_rerank_prompt,
)
from hermes_plugin_honcho.dialectic import DEFAULT_BLOCKING_RESULT_TEMPLATE
from hermes_plugin_honcho.recent_turns import load_exchanges, ring_path
from hermes_plugin_honcho.session import HonchoSession, HonchoSessionManager

ANSWER = "- Cover every change with a test before review."
REPRESENTATION = """## Explicit Observations

[2026-10-01 10:00:00] User writes Go.
[2026-10-02 10:00:00] User wants tests first.

## Deductive Observations

[2026-10-03 10:00:00] User values regression safety.
    - [2026-10-02 10:00:00] User wants tests first.

## Inductive Observations

 **Pattern** [high]: User reviews diffs before merging.
   **Sources**:
   - [2026-10-01 10:00:00] User writes Go.

## Contradictions

 **CONTRADICTION**: User both wants and avoids mocks.
"""
USAGE = SimpleNamespace(input_tokens=41000, output_tokens=900, cache_creation_input_tokens=0, cache_read_input_tokens=0)


class RerankManager:
    """Records every Honcho call a provider makes."""

    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.rerank_calls: list[tuple[str, str, dict]] = []
        self.dialectic_calls: list[str] = []
        self.context_calls: list[str] = []

    def get_rerank_context(self, session_key, search_query, **kwargs):
        self.rerank_calls.append((session_key, search_query, kwargs))
        if self.fail:
            raise self.fail
        return {"representation": REPRESENTATION, "card": ["Name: Ada", "Prefers Go"]}

    def dialectic_query(self, session_key, query, **_kwargs):
        self.dialectic_calls.append(query)
        return "dialectic answer"

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


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: records the client and stream arguments, returns a fixed message."""

    instances: list["FakeAnthropic"] = []

    def __init__(self, *, text=ANSWER, error: Exception | None = None, **client_kwargs):
        self.client_kwargs, self.stream_kwargs, self.error = client_kwargs, None, error
        self.text = text
        self.messages = self
        FakeAnthropic.instances.append(self)

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="private"), SimpleNamespace(type="text", text=self.text)],
            usage=USAGE, stop_reason="end_turn")
        stream = MagicMock()
        stream.__enter__.return_value.get_final_message.return_value = message
        return stream


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeAnthropic.instances.clear()


def make_provider(tmp_path, *, manager=None, factory=FakeAnthropic, **options):
    cfg = HonchoClientConfig(api_key="k", enabled=True, recall_mode="context", hermes_home=tmp_path, timeout=900.0,
                             dialectic_injection_mode="context-rerank", **options)
    provider = HonchoMemoryProvider()
    provider._config = cfg
    provider._manager = manager or RerankManager()
    provider._session_key = "session-a"
    provider._session_initialized = True
    provider._recall_mode = cfg.recall_mode
    provider._dialectic_injection_mode = cfg.dialectic_injection_mode
    provider._injection_log_path = str(tmp_path / "injection.log")
    provider._rerank_client_factory = factory
    return provider


def log_records(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "injection.log").read_text().splitlines()]


def test_render_substitutes_placeholders_literally():
    template = '{"json": true} {peer_representation} | {history_of_trial} | {user_message} {other}'
    rendered = render_rerank_prompt(template, peer_representation="R {user_message}", history_of_trial="H",
                                    user_message="U {history_of_trial}")
    assert rendered == '{"json": true} R {user_message} | H | U {history_of_trial} {other}'


def test_default_prompt_has_every_placeholder_once():
    for name in ("peer_representation", "history_of_trial", "user_message"):
        assert DEFAULT_RERANK_PROMPT_TEMPLATE.count("{" + name + "}") == 1


def test_count_conclusions_per_heading():
    assert count_conclusions(REPRESENTATION) == {"explicit": 2, "deductive": 1, "inductive": 1, "contradictions": 1}
    assert count_conclusions("") == {"explicit": 0, "deductive": 0, "inductive": 0, "contradictions": 0}


def test_call_rerank_streams_and_keeps_only_text():
    text, usage = call_rerank("PROMPT", api_key="key", base_url="https://example.test", model="m", max_tokens=123,
                              timeout=9.0, client_factory=FakeAnthropic)

    (client,) = FakeAnthropic.instances
    assert client.client_kwargs == {"api_key": "key", "base_url": "https://example.test", "timeout": 9.0}
    assert client.stream_kwargs == {"model": "m", "max_tokens": 123,
                                    "messages": [{"role": "user", "content": "PROMPT"}]}
    assert text == ANSWER
    assert usage == {"input_tokens": 41000, "output_tokens": 900, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0, "stop_reason": "end_turn"}


def test_turn_injects_the_rerank_answer_and_logs_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HONCHO_RERANK_API_KEY", "rk")
    make_provider(tmp_path).sync_turn("Write the parser.", "Parser written.")

    provider = make_provider(tmp_path, context_max_conclusions=400, context_search_top_k=100,
                             context_search_max_distance=0.8, context_include_most_frequent=False,
                             dialectic_result_template="<<{dialectic_output}>>")
    provider.on_turn_start(2, "Now add error handling.")
    injected = provider.prefetch("Now add error handling.")

    assert injected == f"<<{ANSWER}>>"
    (key, query, kwargs) = provider._manager.rerank_calls[0]
    assert (key, query) == ("session-a", "Now add error handling.")
    assert kwargs == {"max_conclusions": 400, "search_top_k": 100, "search_max_distance": 0.8,
                      "include_most_frequent": False}
    assert provider._manager.dialectic_calls == []

    (client,) = FakeAnthropic.instances
    assert client.client_kwargs == {"api_key": "rk", "base_url": "https://api.anthropic.com", "timeout": 900.0}
    assert client.stream_kwargs["model"] == "claude-sonnet-5-5" and client.stream_kwargs["max_tokens"] == 16000
    prompt = client.stream_kwargs["messages"][0]["content"]
    assert "<conclusions>\n## Peer Card\nName: Ada\nPrefers Go\n\n## Explicit Observations" in prompt
    assert "### Exchange 1 of 1\nUser: Write the parser.\nAssistant: Parser written." in prompt
    assert "<latest_message>\nNow add error handling.\n</latest_message>" in prompt

    record = log_records(tmp_path)[-1]
    assert record["reason"] == "injected" and record["mode"] == "context-rerank"
    assert record["dialectic_reason"] == "answered" and record["payload"] == injected
    assert record["rerank_prompt"] == prompt and record["rerank_output"] == ANSWER
    assert record["context_counts"] == {"explicit": 2, "deductive": 1, "inductive": 1, "contradictions": 1}
    assert record["context_bytes"] > len(REPRESENTATION)
    assert isinstance(record["context_ms"], int) and isinstance(record["rerank_ms"], int)
    assert record["usage"]["input_tokens"] == 41000 and record["model"] == "claude-sonnet-5-5"


def test_base_url_and_prompt_template_are_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("HONCHO_RERANK_API_KEY", "rk")
    monkeypatch.setenv("HONCHO_RERANK_BASE_URL", "http://proxy.test")
    provider = make_provider(tmp_path, rerank_model="other-model", rerank_max_tokens=50,
                             rerank_prompt_template="ASK {user_message}")
    provider.on_turn_start(1, "Ship it.")
    assert provider.prefetch("Ship it.") == DEFAULT_BLOCKING_RESULT_TEMPLATE.replace("{dialectic_output}", ANSWER)

    (client,) = FakeAnthropic.instances
    assert client.client_kwargs["base_url"] == "http://proxy.test"
    assert client.stream_kwargs["model"] == "other-model" and client.stream_kwargs["max_tokens"] == 50
    assert client.stream_kwargs["messages"][0]["content"] == "ASK Ship it."


def test_mode_skips_base_layer_and_queue_and_writes_the_ring(tmp_path, monkeypatch):
    monkeypatch.setenv("HONCHO_RERANK_API_KEY", "rk")
    provider = make_provider(tmp_path)
    provider.on_turn_start(1, "Write the parser.")
    provider.prefetch("Write the parser.")
    provider.queue_prefetch("Write the parser.")
    provider.sync_turn("Write the parser.", "Parser written.")

    assert provider._manager.context_calls == []
    assert [t["user"] for t in load_exchanges(ring_path(tmp_path), "session-a")] == ["Write the parser."]


@pytest.mark.parametrize(("env", "manager", "factory", "reason", "error"), [
    ({}, None, FakeAnthropic, "rerank-failed", "HONCHO_RERANK_API_KEY is not set"),
    ({"HONCHO_RERANK_API_KEY": "rk"}, RerankManager(fail=RuntimeError("boom")), FakeAnthropic,
     "context-failed", "RuntimeError: boom"),
    ({"HONCHO_RERANK_API_KEY": "rk"}, None, lambda **kw: FakeAnthropic(error=TimeoutError("slow"), **kw),
     "rerank-failed", "TimeoutError: slow"),
    ({"HONCHO_RERANK_API_KEY": "rk"}, None, lambda **kw: FakeAnthropic(text="  ", **kw), "rerank-empty", None),
])
def test_failures_inject_nothing_and_log_why(tmp_path, monkeypatch, env, manager, factory, reason, error):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    provider = make_provider(tmp_path, manager=manager, factory=factory)
    provider.on_turn_start(1, "Write the parser.")

    assert provider.prefetch("Write the parser.") == ""
    record = log_records(tmp_path)[-1]
    assert record["reason"] == reason and record["mode"] == "context-rerank"
    assert record.get("error") == error


def test_get_rerank_context_passes_knobs_and_omits_none():
    manager = HonchoSessionManager.__new__(HonchoSessionManager)
    session = HonchoSession(key="k", user_peer_id="user-1", assistant_peer_id="ai", honcho_session_id="s")
    peer = MagicMock()
    peer.context.return_value = SimpleNamespace(representation=REPRESENTATION, peer_card=["Name: Ada"])
    manager._cached_session = lambda key: session
    manager._resolve_observer_target = lambda s, p: ("ai", "user-1")
    manager._get_or_create_peer = lambda peer_id: peer
    manager._authed_call = lambda label, fn: fn()

    ctx = manager.get_rerank_context("k", "query", max_conclusions=400, search_top_k=100,
                                     search_max_distance=None, include_most_frequent=True)

    peer.context.assert_called_once_with(target="user-1", search_query="query", search_top_k=100,
                                         include_most_frequent=True, max_conclusions=400)
    assert ctx == {"representation": REPRESENTATION, "card": ["Name: Ada"]}


def test_config_parses_the_rerank_keys(tmp_path):
    from hermes_plugin_honcho.client import _behavior_fields, _HostLookup

    raw = {"dialecticInjectionMode": "context-rerank", "contextMaxConclusions": 300, "contextSearchTopK": 50,
           "contextSearchMaxDistance": 0.7, "contextIncludeMostFrequent": False, "rerankModel": "m",
           "rerankMaxTokens": 99, "rerankPromptTemplate": "T {user_message}"}
    fields = _behavior_fields(_HostLookup({}, raw), explicitly_configured=True)
    assert {k: fields[k] for k in ("dialectic_injection_mode", "context_max_conclusions", "context_search_top_k",
                                   "context_search_max_distance", "context_include_most_frequent", "rerank_model",
                                   "rerank_max_tokens", "rerank_prompt_template")} == {
        "dialectic_injection_mode": "context-rerank", "context_max_conclusions": 300, "context_search_top_k": 50,
        "context_search_max_distance": 0.7, "context_include_most_frequent": False, "rerank_model": "m",
        "rerank_max_tokens": 99, "rerank_prompt_template": "T {user_message}"}

    defaults = _behavior_fields(_HostLookup({}, {}), explicitly_configured=True)
    assert (defaults["context_max_conclusions"], defaults["context_search_top_k"],
            defaults["context_search_max_distance"], defaults["context_include_most_frequent"],
            defaults["rerank_model"], defaults["rerank_max_tokens"], defaults["rerank_prompt_template"]) == (
        400, 100, None, True, "claude-sonnet-5-5", 16000, None)

# pyright: reportPrivateUsage=false, reportUnusedParameter=false
"""Tests for the Langfuse session/trace wiring in `src/llm/runtime.py`.

One agentic run = one trace; `session_id = run_id` (globally unique, so it's
conflict-free across tenants — unlike the Honcho session name). The run root
(`langfuse_agent_run`) owns the trace attrs; step spans and generations nest
under it. A nested generation stays silent (the run owns the attrs); one that
escaped its run root (the streamed final answer) re-joins the session by run_id.
Single-call agents (deriver, summarizer) get no run root and no session.
Disabled (no LANGFUSE_PUBLIC_KEY) → no Langfuse calls at all.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from src.config import settings
from src.llm import runtime
from src.llm.types import LLMTelemetryContext


@pytest.fixture
def capture_propagate(monkeypatch: pytest.MonkeyPatch):
    """Stub `langfuse.propagate_attributes`, capturing the kwargs it's called with.

    Returns a dict that's empty until propagate_attributes is invoked.
    """
    captured: dict[str, Any] = {}

    @contextlib.contextmanager
    def fake_propagate(**kwargs: Any):
        captured.clear()
        captured.update(kwargs)
        yield

    import langfuse

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    return captured


@pytest.fixture
def langfuse_client(monkeypatch: pytest.MonkeyPatch):
    """Stub `langfuse.get_client()`, capturing observation/generation/span calls.

    Returns ``{"observation", "generation", "span", "run_span"}`` — each sub-dict
    stays empty until the corresponding call is made:
    - ``observation``: the span opened via `start_as_current_observation`
      (run root or step span).
    - ``generation``: the rename via `update_current_generation`.
    - ``span``: the step I/O via `update_current_span`.
    - ``run_span``: I/O written onto the yielded run-root span object's
      ``.update`` (merged, mirroring Langfuse's update semantics).
    """
    captured: dict[str, dict[str, Any]] = {
        "observation": {},
        "generation": {},
        "span": {},
        "run_span": {},
    }

    class FakeSpan:
        def update(self, **kwargs: Any) -> None:
            # Merge (don't clear): Langfuse's span.update accumulates, so input
            # set at run start and output set at run end coexist.
            captured["run_span"].update(kwargs)

    @contextlib.contextmanager
    def fake_observation(**kwargs: Any):
        captured["observation"].clear()
        captured["observation"].update(kwargs)
        yield FakeSpan()

    class FakeClient:
        def start_as_current_observation(self, **kwargs: Any):
            return fake_observation(**kwargs)

        def update_current_generation(self, **kwargs: Any) -> None:
            captured["generation"].clear()
            captured["generation"].update(kwargs)

        def update_current_span(self, **kwargs: Any) -> None:
            captured["span"].clear()
            captured["span"].update(kwargs)

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", lambda: FakeClient())
    return captured


@pytest.fixture
def langfuse_enabled(monkeypatch: pytest.MonkeyPatch):
    """Turn the integration on with a known NAMESPACE (the tenant / user_id)."""
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "NAMESPACE", "acme-tenant")


@contextlib.contextmanager
def _inside_agent_run():
    """Set the `_in_agent_run` ContextVar for the body, resetting it after.

    Simulates execution nested inside a run root without opening one (which
    would itself call propagate_attributes and pollute the capture).
    """
    token = runtime._in_agent_run.set(True)
    try:
        yield
    finally:
        runtime._in_agent_run.reset(token)


class TestAnnotateDisabled:
    def test_noop_when_key_unset(
        self, monkeypatch: pytest.MonkeyPatch, capture_propagate: dict[str, Any]
    ):
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)

        runtime.annotate_current_langfuse_trace(
            "anthropic",
            "claude-x",
            telemetry=LLMTelemetryContext(run_id="run-abc"),
        )

        assert capture_propagate == {}


class TestAnnotateInsideRun:
    """A generation nested inside a run root: the run owns the trace attrs, so
    this call must NOT propagate — it only renames the generation."""

    def test_nested_generation_does_not_propagate(
        self,
        langfuse_enabled: None,
        capture_propagate: dict[str, Any],
        langfuse_client: dict[str, dict[str, Any]],
    ):
        telemetry = LLMTelemetryContext(
            workspace_name="ws1",
            call_purpose="dialectic.answer",
            agent_type="dialectic",
            run_id="run-abc",
            iteration=2,
            peer_name="alice",
            track_name="Dialectic Agent",
        )

        with _inside_agent_run():
            runtime.annotate_current_langfuse_trace(
                "anthropic", "claude-x", telemetry=telemetry
            )

        # Run root owns user_id/session_id/trace_name — re-propagating here would
        # clobber the run's session, so we don't propagate at all.
        assert capture_propagate == {}
        # The generation is still named per agent+action for by-name aggregation.
        assert langfuse_client["generation"]["name"] == "Dialectic Agent LLM call"


class TestAnnotateOwnTraceRoot:
    """A generation that IS its own trace root stamps the trace attributes:
    single calls (no run_id → no session) and multi-turn generations that
    escaped their run root (run_id present, no active run root → rejoin the
    run's session by run_id)."""

    def test_single_call_has_no_session_and_names_trace(
        self,
        langfuse_enabled: None,
        capture_propagate: dict[str, Any],
        langfuse_client: dict[str, dict[str, Any]],
    ):
        telemetry = LLMTelemetryContext(
            workspace_name="ws1",
            call_purpose="deriver.representation",
            observed="bob",
            track_name="Minimal Deriver",
        )

        runtime.annotate_current_langfuse_trace(
            "gemini", "gemini-x", telemetry=telemetry
        )

        assert capture_propagate["session_id"] is None
        assert capture_propagate["user_id"] == "acme-tenant"
        # Single-call: this generation IS the trace root, so it names the trace.
        assert capture_propagate["trace_name"] == "Minimal Deriver"
        assert capture_propagate["metadata"]["observed"] == "bob"
        # The generation observation is still named per agent+action.
        assert langfuse_client["generation"]["name"] == "Minimal Deriver LLM call"

    def test_no_telemetry_still_stamps_user_id(
        self,
        langfuse_enabled: None,
        capture_propagate: dict[str, Any],
        langfuse_client: dict[str, dict[str, Any]],
    ):
        runtime.annotate_current_langfuse_trace("openai", "gpt-x", telemetry=None)

        assert capture_propagate["user_id"] == "acme-tenant"
        assert capture_propagate["session_id"] is None
        assert capture_propagate["trace_name"] is None
        assert capture_propagate["metadata"]["provider"] == "openai"
        # No telemetry → no per-agent generation name.
        assert langfuse_client["generation"] == {}

    def test_escaped_multiturn_generation_rejoins_session(
        self,
        langfuse_enabled: None,
        capture_propagate: dict[str, Any],
        langfuse_client: dict[str, dict[str, Any]],
    ):
        # run_id present but NOT inside a run root (the streamed final answer
        # drains after the tool loop closed its run trace) → this generation is
        # its own trace root and rejoins the run's session by run_id.
        telemetry = LLMTelemetryContext(
            run_id="run-abc",
            iteration=2,
            workspace_name="ws1",
            track_name="Dialectic Agent Stream",
        )

        runtime.annotate_current_langfuse_trace(
            "anthropic", "claude-x", telemetry=telemetry
        )

        # The Langfuse session is the agentic run, not a Honcho Session.
        assert capture_propagate["session_id"] == "run-abc"
        assert capture_propagate["trace_name"] == "Dialectic Agent Stream"
        metadata = capture_propagate["metadata"]
        assert "honcho_session_id" not in metadata
        assert metadata["workspace_name"] == "ws1"
        # propagate_attributes requires str values.
        assert metadata["iteration"] == "2"
        assert all(isinstance(v, str) for v in metadata.values())
        assert (
            langfuse_client["generation"]["name"] == "Dialectic Agent Stream LLM call"
        )


class TestAgentRun:
    """`langfuse_agent_run` opens the one run-level trace per run: an
    ``as_type="span"`` root with ``session_id = run_id``, opened only for
    multi-turn runs (run_id present)."""

    def test_noop_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)

        with runtime.langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="r1")
        ) as span:
            assert span is None

        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_run_id(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        # Single-call agents (deriver/summarizer) have no run_id → no run root,
        # so their LLM calls stay standalone, sessionless traces.
        with runtime.langfuse_agent_run(
            "Minimal Deriver", LLMTelemetryContext(workspace_name="ws1")
        ) as span:
            assert span is None

        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_telemetry(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        with runtime.langfuse_agent_run("anything", None) as span:
            assert span is None

        assert langfuse_client["observation"] == {}

    def test_opens_run_root_and_owns_trace_attrs(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        tele = LLMTelemetryContext(
            workspace_name="ws1",
            call_purpose="dialectic.answer",
            agent_type="dialectic",
            run_id="run-abc",
            observed="bob",
            track_name="Dialectic Agent",
        )

        with runtime.langfuse_agent_run("Dialectic Agent", tele) as span:
            assert span is not None

        # The run IS the trace root: an as_type="span" observation whose name is
        # STABLE (no step number) so Langfuse aggregates by name.
        observation = langfuse_client["observation"]
        assert observation["as_type"] == "span"
        assert observation["name"] == "Dialectic Agent"
        # Trace grouping: one Langfuse session per run, drillable per tenant.
        assert capture_propagate["session_id"] == "run-abc"
        assert capture_propagate["user_id"] == "acme-tenant"
        assert capture_propagate["trace_name"] == "Dialectic Agent"
        md = capture_propagate["metadata"]
        assert md["workspace_name"] == "ws1"
        assert md["agent_type"] == "dialectic"
        assert md["observed"] == "bob"
        # Honcho's Session is deliberately NOT the grouping key.
        assert "honcho_session_id" not in md

    def test_marks_in_agent_run_for_nested_calls(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        # Inside the run root, nested generations must see _in_agent_run set so
        # they stay silent; it resets on exit so a later streamed final answer
        # stamps itself as its own trace.
        assert runtime._in_agent_run.get() is False
        with runtime.langfuse_agent_run(
            "Dialectic Agent",
            LLMTelemetryContext(run_id="r1", track_name="Dialectic Agent"),
        ):
            assert runtime._in_agent_run.get() is True
        assert runtime._in_agent_run.get() is False


class TestAgentStep:
    """`langfuse_agent_step` opens a per-iteration child span under the run
    root (one reasoning turn). Unlike the run root, it does NOT touch trace
    attributes."""

    def test_noop_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)

        with runtime.langfuse_agent_step(
            "Dialectic Agent step", LLMTelemetryContext(run_id="r1")
        ):
            pass

        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_run_id(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        with runtime.langfuse_agent_step(
            "Minimal Deriver step", LLMTelemetryContext(workspace_name="ws1")
        ):
            pass

        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_telemetry(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        with runtime.langfuse_agent_step("anything", None):
            pass

        assert langfuse_client["observation"] == {}

    def test_opens_child_span_without_propagating(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        tele = LLMTelemetryContext(
            workspace_name="ws1",
            agent_type="dialectic",
            run_id="run-abc",
            iteration=2,
            observed="bob",
            track_name="Dialectic Agent",
        )

        with runtime.langfuse_agent_step("Dialectic Agent step", tele):
            pass

        observation = langfuse_client["observation"]
        assert observation["as_type"] == "span"
        assert observation["name"] == "Dialectic Agent step"
        # The per-step index rides on the span's metadata (str-coerced).
        assert observation["metadata"]["iteration"] == "2"
        assert observation["metadata"]["observed"] == "bob"
        # The run root owns the trace attrs — the step must NOT propagate.
        assert capture_propagate == {}


class TestRunIO:
    """`annotate_langfuse_run_io` stamps the run-root span (the trace root) so
    the trace list shows the run's query (input) and final answer (output)."""

    def test_noop_when_run_span_none(self, langfuse_client: dict[str, dict[str, Any]]):
        runtime.annotate_langfuse_run_io(
            None, input=[{"role": "user", "content": "hi"}], output="answer"
        )

        assert langfuse_client["run_span"] == {}

    def test_sets_input_and_output(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        messages = [{"role": "user", "content": "How many coffees?"}]

        with runtime.langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="run-abc")
        ) as span:
            # input stamped at run start, output at run end — two calls, merged.
            runtime.annotate_langfuse_run_io(span, input=messages)
            runtime.annotate_langfuse_run_io(span, output="You bought 4 coffees.")

        assert langfuse_client["run_span"]["input"] == messages
        assert langfuse_client["run_span"]["output"] == "You bought 4 coffees."

    def test_writes_only_provided_kwargs(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        with runtime.langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="run-abc")
        ) as span:
            runtime.annotate_langfuse_run_io(span, input=[{"role": "user"}])

        # Only input was passed → output is not written (so it isn't blanked).
        assert "input" in langfuse_client["run_span"]
        assert "output" not in langfuse_client["run_span"]


class TestStepIO:
    """`annotate_langfuse_step_io` stamps the step span so the per-turn span
    isn't blank — only the nested generation used to carry I/O."""

    def test_noop_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)

        runtime.annotate_langfuse_step_io(
            LLMTelemetryContext(run_id="r1"),
            [{"role": "user", "content": "hi"}],
            "hi",
            [],
        )

        assert langfuse_client["span"] == {}

    def test_noop_without_run_id(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        # Single-call agents open no step span, so there's no span to annotate.
        runtime.annotate_langfuse_step_io(
            LLMTelemetryContext(workspace_name="ws1"),
            [{"role": "user", "content": "hi"}],
            "hi",
            [],
        )

        assert langfuse_client["span"] == {}

    def test_text_answer_sets_input_and_output(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        messages = [{"role": "user", "content": "What is the user's name?"}]

        runtime.annotate_langfuse_step_io(
            LLMTelemetryContext(run_id="run-abc"),
            messages,
            "The user's name is Jordan.",
            [],
        )

        span = langfuse_client["span"]
        assert span["input"] == messages
        assert span["output"] == "The user's name is Jordan."

    def test_tool_calling_turn_summarizes_tools_as_output(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        # A tool-calling turn has no text yet — the step's "output" is the set of
        # tools it chose, by name (the per-tool I/O lives on the tool children).
        runtime.annotate_langfuse_step_io(
            LLMTelemetryContext(run_id="run-abc"),
            [{"role": "user", "content": "how many coffees?"}],
            "",
            [{"name": "grep_messages"}, {"name": "search_memory"}],
        )

        assert langfuse_client["span"]["output"] == {
            "tool_calls": ["grep_messages", "search_memory"]
        }

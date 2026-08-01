# pyright: reportPrivateUsage=false, reportUnusedParameter=false
"""Tests for the Langfuse session/trace wiring in `src/llm/runtime.py`.

One agentic run = one trace; `session_id = run_id` (globally unique, so it's
conflict-free across tenants — unlike the Honcho session name). The run handle
(`start_langfuse_agent_run`) opens an `as_type="span"` root and keeps it
current via an ``ExitStack`` until `.end()`. Step spans + nested generations
nest under the run while it's open. A single-call agent (deriver, summarizer)
gets no run handle and self-stamps its lone generation as the trace root.
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
    - ``run_span``: I/O written onto the span object handed back by
      `start_as_current_observation` (merged, mirroring Langfuse's update
      semantics).
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
    """Turn the integration on with a known NAMESPACE (the tenant / user_id).

    Pins LANGFUSE_EXPORTER_MODE='inline' — this module tests the legacy inline
    span machinery, which is gated to inline mode (the default is now 'exporter',
    where these functions no-op in favor of the LangfuseExporter)."""
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "LANGFUSE_EXPORTER_MODE", "inline")
    monkeypatch.setattr(settings, "NAMESPACE", "acme-tenant")


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
    """A generation nested under a run (it carries a `parent_span_id`): the run
    owns the trace attrs, so this call must NOT propagate. It still stamps model
    + per-step metadata + name on the generation (the multi-turn regression fix —
    provider/model used to be dropped on every iteration after the first).
    Nesting is derived from the explicit `parent_span_id`, not a contextvar."""

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
            span_id="run-abc",
            # A non-null parent_span_id is what marks this generation as nested
            # under the run span (replaces the old `_in_agent_run` contextvar).
            parent_span_id="run-abc",
            iteration=2,
            peer_name="alice",
            track_name="Dialectic Agent",
        )

        runtime.annotate_current_langfuse_trace(
            "anthropic", "claude-x", telemetry=telemetry
        )

        # Run handle owns user_id/session_id/trace_name — re-propagating here
        # would clobber the run's session, so we don't propagate at all.
        assert capture_propagate == {}
        # Per-call generation: name + model + step metadata stamped every
        # iteration (formerly only name was stamped, dropping provider/model).
        gen = langfuse_client["generation"]
        assert gen["name"] == "Dialectic Agent LLM call"
        assert gen["model"] == "claude-x"
        assert gen["metadata"]["provider"] == "anthropic"
        assert gen["metadata"]["model"] == "claude-x"
        assert gen["metadata"]["iteration"] == "2"
        assert gen["metadata"]["agent_type"] == "dialectic"


class TestAnnotateOwnTraceRoot:
    """A generation that IS its own trace root stamps the trace attributes:
    single calls (no run_id → no session) and the no-telemetry case."""

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
        # No telemetry → no per-agent generation name, but provider/model still set.
        gen = langfuse_client["generation"]
        assert gen["name"] is None
        assert gen["model"] == "gpt-x"


class TestAgentRun:
    """`start_langfuse_agent_run` returns an imperative handle: opens an
    ``as_type="span"`` root, stamps ``session_id = run_id`` via
    ``propagate_attributes``, and keeps the span open until ``.end()``.
    Only fires for multi-turn runs (run_id present)."""

    def test_noop_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)

        handle = runtime.start_langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="r1")
        )

        assert handle is None
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
        handle = runtime.start_langfuse_agent_run(
            "Minimal Deriver", LLMTelemetryContext(workspace_name="ws1")
        )

        assert handle is None
        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_telemetry(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        handle = runtime.start_langfuse_agent_run("anything", None)
        assert handle is None
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

        handle = runtime.start_langfuse_agent_run("Dialectic Agent", tele)
        assert handle is not None
        try:
            # The run IS the trace root: an as_type="span" observation whose
            # name is STABLE (no step number) so Langfuse aggregates by name.
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
        finally:
            handle.end()

    def test_run_keyed_on_span_id_and_nesting_via_parent_span_id(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        # The `_in_agent_run` contextvar is retired — nesting is now derived
        # from the explicit `parent_span_id` field on the telemetry context.
        assert not hasattr(runtime, "_in_agent_run")

        # The run handle opens keyed on span_id (falling back to run_id).
        handle = runtime.start_langfuse_agent_run(
            "Dialectic Agent",
            LLMTelemetryContext(
                run_id="r1", span_id="r1", track_name="Dialectic Agent"
            ),
        )
        assert handle is not None
        handle.end()

        # A root call (no parent_span_id) propagates trace attrs; a nested call
        # (parent_span_id set) stays silent.
        capture_propagate.clear()
        runtime.annotate_current_langfuse_trace(
            "anthropic",
            "claude-x",
            telemetry=LLMTelemetryContext(run_id="r2", span_id="r2"),
        )
        assert capture_propagate.get("session_id") == "r2"

        capture_propagate.clear()
        runtime.annotate_current_langfuse_trace(
            "anthropic",
            "claude-x",
            telemetry=LLMTelemetryContext(
                run_id="r2", span_id="r2", parent_span_id="r2"
            ),
        )
        assert capture_propagate == {}

    def test_end_is_idempotent(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        # The streaming wrapper may call .end() after the api.py finally already
        # called it (or vice-versa); the handle has to tolerate that.
        handle = runtime.start_langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="r1")
        )
        assert handle is not None
        handle.end()
        handle.end()  # must not raise


class TestAgentRunIO:
    """The run handle exposes `.update(input=..., output=...)` for stamping
    the run-root span — the trace's input/output preview in the Langfuse UI.
    A second call merges into the first (Langfuse's update semantics)."""

    def test_sets_input_then_output_on_handle(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        messages = [{"role": "user", "content": "How many coffees?"}]
        handle = runtime.start_langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="run-abc")
        )
        assert handle is not None
        try:
            handle.update(input=messages)
        finally:
            handle.end(output="You bought 4 coffees.")

        assert langfuse_client["run_span"]["input"] == messages
        assert langfuse_client["run_span"]["output"] == "You bought 4 coffees."

    def test_end_without_output_leaves_output_unset(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        handle = runtime.start_langfuse_agent_run(
            "Dialectic Agent", LLMTelemetryContext(run_id="run-abc")
        )
        assert handle is not None
        handle.update(input=[{"role": "user"}])
        handle.end()

        assert "input" in langfuse_client["run_span"]
        # Only input was passed → output is not written (so it isn't blanked).
        assert "output" not in langfuse_client["run_span"]


class TestAgentStep:
    """`start_langfuse_agent_step` opens a per-iteration child span under the
    run root (one reasoning turn). Unlike the run handle, it does NOT touch
    trace attributes."""

    def test_noop_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)

        step = runtime.start_langfuse_agent_step(
            "Dialectic Agent step", LLMTelemetryContext(run_id="r1")
        )

        assert step is None
        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_run_id(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
        capture_propagate: dict[str, Any],
    ):
        step = runtime.start_langfuse_agent_step(
            "Minimal Deriver step", LLMTelemetryContext(workspace_name="ws1")
        )

        assert step is None
        assert langfuse_client["observation"] == {}
        assert capture_propagate == {}

    def test_noop_without_telemetry(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        step = runtime.start_langfuse_agent_step("anything", None)
        assert step is None
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

        step = runtime.start_langfuse_agent_step("Dialectic Agent step", tele)
        assert step is not None
        try:
            observation = langfuse_client["observation"]
            assert observation["as_type"] == "span"
            assert observation["name"] == "Dialectic Agent step"
            # The per-step index rides on the span's metadata (str-coerced).
            assert observation["metadata"]["iteration"] == "2"
            assert observation["metadata"]["observed"] == "bob"
            # The run root owns the trace attrs — the step must NOT propagate.
            assert capture_propagate == {}
        finally:
            step.end()


class TestStepIO:
    """`step.annotate_io` stamps this turn's I/O on the step span — without it,
    only the nested generation would carry I/O and the step would show blank."""

    def test_text_answer_sets_input_and_output(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        messages = [{"role": "user", "content": "What is the user's name?"}]
        step = runtime.start_langfuse_agent_step(
            "Dialectic Agent step", LLMTelemetryContext(run_id="run-abc")
        )
        assert step is not None
        try:
            step.annotate_io(messages, "The user's name is Jordan.", [])
        finally:
            step.end()

        # The step span's input/output is stamped via the handle's underlying
        # span.update() — captured on the run_span fixture key.
        assert langfuse_client["run_span"]["input"] == messages
        assert langfuse_client["run_span"]["output"] == "The user's name is Jordan."

    def test_tool_calling_turn_summarizes_tools_as_output(
        self,
        langfuse_enabled: None,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        # A tool-calling turn has no text yet — the step's "output" is the set
        # of tools it chose, by name (per-tool I/O lives on the tool children).
        step = runtime.start_langfuse_agent_step(
            "Dialectic Agent step", LLMTelemetryContext(run_id="run-abc")
        )
        assert step is not None
        try:
            step.annotate_io(
                [{"role": "user", "content": "how many coffees?"}],
                "",
                [{"name": "grep_messages"}, {"name": "search_memory"}],
            )
        finally:
            step.end()

        assert langfuse_client["run_span"]["output"] == {
            "tool_calls": ["grep_messages", "search_memory"]
        }


class TestAnnotateGenerationIOGating:
    """`annotate_current_generation_io` writes to the ACTIVE @observe generation
    span (the `conditional_observe` wrapper), which only exists in inline mode.
    In exporter mode — the default — there is no active span, so calling
    `update_current_generation()` would make the Langfuse SDK log "No active span
    in current context" on every LLM call. The helper must therefore no-op in
    exporter mode (the LangfuseExporter projects I/O from the captured stream).
    Regression guard for the gate that was on LANGFUSE_PUBLIC_KEY instead of
    langfuse_inline_enabled."""

    def test_noops_in_exporter_mode_even_with_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        langfuse_client: dict[str, dict[str, Any]],
    ):
        # Key present but exporter mode (the production default).
        monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setattr(settings, "LANGFUSE_EXPORTER_MODE", "exporter")

        runtime.annotate_current_generation_io(
            input=[{"role": "user", "content": "hi"}],
            output="hello",
            usage_details={"input": 1, "output": 1},
        )

        # No active generation span in exporter mode → must not touch it.
        assert langfuse_client["generation"] == {}

    def test_writes_in_inline_mode(
        self,
        langfuse_enabled: None,  # pins inline mode + a key
        langfuse_client: dict[str, dict[str, Any]],
    ):
        messages = [{"role": "user", "content": "hi"}]
        runtime.annotate_current_generation_io(
            input=messages,
            output="hello",
            usage_details={"input": 1, "output": 1},
        )

        gen = langfuse_client["generation"]
        assert gen["input"] == messages
        assert gen["output"] == "hello"
        assert gen["usage_details"] == {"input": 1, "output": 1}

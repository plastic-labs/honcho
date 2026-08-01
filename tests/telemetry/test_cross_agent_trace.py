# pyright: reportPrivateUsage=false, reportUnannotatedClassAttribute=false, reportUnusedFunction=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Cross-agent trace-metadata contract.

Verifies that each agent's telemetry produces a well-formed `CapturedLLMCall`
with the right correlation/session/identity fields, and that the SAME captured
call fans out to BOTH exporters (CloudEvents + Langfuse) — the "one data model,
two projections" invariant.

This is the metadata-correctness bar across agents. It drives the real
`DialecticAgent` telemetry and the real `dispatch_captured_call`; the other
agents are represented by the telemetry contexts they construct (cited inline).
Full end-to-end capture through a live stack (real subprocesses feeding a trace
sink) is exercised separately, outside this unit suite.
"""

from __future__ import annotations

import pytest

from src.config import settings
from src.dialectic.core import DialecticAgent
from src.llm import capture as capture_mod
from src.llm.backend import CompletionResult
from src.llm.capture import (
    CapturedLLMCall,
    build_captured_call,
    dispatch_captured_call,
    register_exporter,
)
from src.llm.types import LLMTelemetryContext
from src.telemetry.langfuse_exporter import LangfuseExporter


class SpyExporter:
    def __init__(self) -> None:
        self.calls: list[CapturedLLMCall] = []

    def export(self, call: CapturedLLMCall) -> None:
        self.calls.append(call)


@pytest.fixture
def spy() -> SpyExporter:
    """Clean exporter registry with a single spy (restored by the telemetry
    conftest's _isolate_trace_globals)."""
    capture_mod._EXPORTERS.clear()
    exporter = SpyExporter()
    register_exporter(exporter)
    return exporter


def _dispatch(telemetry: LLMTelemetryContext, *, content: str = "answer") -> None:
    call = build_captured_call(
        telemetry=telemetry,
        transport="anthropic",
        provider_label=None,
        model="claude-x",
        messages=[{"role": "user", "content": "q"}],
        tools=None,
        tool_choice=None,
        result=CompletionResult(content=content, finish_reason="stop"),
        attempt=1,
        was_fallback=False,
        was_stream=False,
        finish_reason="stop",
    )
    dispatch_captured_call(call)


# --- Dialectic: real agent telemetry --------------------------------------


def test_dialectic_with_session_sets_session_id(spy: SpyExporter):
    agent = DialecticAgent(
        workspace_name="ws",
        session_name="my-session",
        session_id="sess-nanoid",
        observer="alice",
        observed="bob",
    )
    _dispatch(agent._telemetry_context("Dialectic Agent"))

    call = spy.calls[-1]
    assert call.session_id == "sess-nanoid"
    assert call.agent_type == "dialectic"
    # Root of the invocation: trace_id == span_id == run_id, parent normalized off.
    assert call.trace_id == call.span_id == agent._run_id
    assert call.parent_span_id is None
    assert call.track_name == "Dialectic Agent"


def test_dialectic_global_has_no_session(spy: SpyExporter):
    agent = DialecticAgent(
        workspace_name="ws",
        session_name=None,
        session_id=None,
        observer="alice",
        observed="alice",
    )
    _dispatch(agent._telemetry_context("Dialectic Agent"))
    assert spy.calls[-1].session_id is None


# --- Background agents: sessionless single-shot / shared-tree contracts -----


@pytest.mark.parametrize(
    ("call_purpose", "parent_category", "track_name"),
    [
        ("deriver.representation", "representation", "Minimal Deriver"),
        ("summary.short", "summary", None),
    ],
)
def test_background_agents_are_sessionless_single_shot(
    spy: SpyExporter,
    call_purpose: str,
    parent_category: str,
    track_name: str | None,
):
    # Deriver + summarizer mirror their src/ contexts: trace_id == span_id, no
    # run_id/session_id, self-rooted.
    tid = f"{parent_category}-trace"
    _dispatch(
        LLMTelemetryContext(
            workspace_name="ws",
            call_purpose=call_purpose,
            parent_category=parent_category,
            track_name=track_name,
            trace_id=tid,
            span_id=tid,
        )
    )
    call = spy.calls[-1]
    assert call.session_id is None
    assert call.run_id is None
    assert call.trace_id == call.span_id == tid
    assert call.parent_span_id is None


def test_dreamer_specialists_share_one_tree(spy: SpyExporter):
    # Single-dream-tree (this PR): both specialists reuse the orchestrator run_id
    # as trace_id (src/dreamer/specialists.py), session_id None.
    run_id = "dream-run"
    for agent_type in ("deduction", "induction"):
        _dispatch(
            LLMTelemetryContext(
                workspace_name="ws",
                call_purpose=f"dream.{agent_type}",
                parent_category="dream",
                agent_type=agent_type,
                run_id=run_id,
                trace_id=run_id,
                span_id=run_id,
                observer="assistant",
                observed="bob",
                iteration=1,
            )
        )
    ded, ind = spy.calls[-2], spy.calls[-1]
    assert ded.trace_id == ind.trace_id == run_id  # one shared tree
    assert ded.session_id is None and ind.session_id is None
    assert ded.agent_type == "deduction" and ind.agent_type == "induction"


# --- One data model, two projections ---------------------------------------


def test_same_call_reaches_both_exporters(
    spy: SpyExporter, monkeypatch: pytest.MonkeyPatch
):
    """A dispatched call fans out to the CloudEvents spy AND the LangfuseExporter."""
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "LANGFUSE_EXPORTER_MODE", "exporter")
    monkeypatch.setattr(settings, "NAMESPACE", "tenant1")

    created: list[dict[str, object]] = []

    class FakeOtel:
        def set_attribute(self, *_a: object) -> None: ...

    class FakeObs:
        def __init__(self, **kwargs: object) -> None:
            self.id = "obs"
            self.kwargs = kwargs
            self._otel_span = FakeOtel()

        def end(self) -> None: ...

    class FakeClient:
        def create_trace_id(self, *, seed: str | None = None) -> str:
            return f"lf-{seed}"

        def start_observation(self, **kwargs: object) -> FakeObs:
            created.append(kwargs)
            return FakeObs(**kwargs)

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", lambda: FakeClient())
    register_exporter(LangfuseExporter())

    agent = DialecticAgent(
        workspace_name="ws",
        session_name="s",
        session_id="sess-1",
        observer="alice",
        observed="bob",
    )
    _dispatch(agent._telemetry_context("Dialectic Agent"))

    # CloudEvents projection saw the raw captured call...
    assert spy.calls and spy.calls[-1].session_id == "sess-1"
    # ...and the Langfuse projection built observations from the SAME call.
    assert created, "LangfuseExporter produced no observations"
    assert any(o.get("as_type") == "generation" for o in created)

# pyright: reportPrivateUsage=false, reportUnannotatedClassAttribute=false, reportUnusedFunction=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportIndexIssue=false
"""Tests for the Langfuse projection over the captured LLM stream.

Exercises `LangfuseExporter` with a fake Langfuse client so we can assert the
reconstructed trace tree (trace ids, parent linkage, names, usage, trace-level
user attributes, session-as-metadata) without a real Langfuse backend.
"""

from __future__ import annotations

import time
from typing import cast

import pytest

from src.config import settings
from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.capture import CapturedSpan, CapturedToolCall, build_captured_call
from src.llm.types import LLMTelemetryContext
from src.telemetry import langfuse_session
from src.telemetry.langfuse_exporter import LangfuseExporter


class FakeOtelSpan:
    def __init__(self, parent: object) -> None:
        self.attributes: dict[str, object] = {}
        self._start_time: int | None = time.time_ns()
        self.end_time: int | None = None
        self._parent: object | None = parent

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class FakeObs:
    _counter = 0

    def __init__(self, **kwargs: object) -> None:
        FakeObs._counter += 1
        self.id = f"obs-{FakeObs._counter}"
        self.kwargs = kwargs
        # Like the SDK: a trace_context span without a parent_span_id still gets
        # a (placeholder) parent.
        trace_context = cast(dict[str, str], kwargs.get("trace_context") or {})
        self._otel_span = FakeOtelSpan(
            trace_context.get("parent_span_id") or "sdk-placeholder"
        )
        self.updates: dict[str, object] = {}
        self.ended = False

    def update(self, **kwargs: object) -> None:
        self.updates.update(kwargs)

    def end(self, *, end_time: int | None = None) -> None:
        self.ended = True
        self._otel_span.end_time = end_time if end_time is not None else time.time_ns()


class FakeClient:
    def __init__(self) -> None:
        self.observations: list[FakeObs] = []

    def create_trace_id(self, *, seed: str | None = None) -> str:
        return f"lf-{seed}"

    def start_observation(self, **kwargs: object) -> FakeObs:
        obs = FakeObs(**kwargs)
        self.observations.append(obs)
        return obs


@pytest.fixture(autouse=True)
def _exporter_env(monkeypatch: pytest.MonkeyPatch):
    """Enable the exporter and install a fake langfuse client + clean registry."""
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "NAMESPACE", "tenant1")
    client = FakeClient()
    import langfuse

    monkeypatch.setattr(langfuse, "get_client", lambda: client)
    langfuse_session.reset()
    FakeObs._counter = 0
    yield client
    langfuse_session.reset()


def _call(
    *,
    run_id: str | None,
    trace_id: str,
    iteration: int | None = None,
    step_seq: int = 0,
    attempt: int = 1,
    session_id: str | None = None,
    track_name: str | None = None,
    agent_type: str = "dialectic",
    parent_category: str = "dialectic",
    tool_names: list[str] | None = None,
    finish_reason: str = "stop",
    content: str = "answer",
    thinking: str | None = None,
    duration_ms: float | None = None,
):
    telemetry = LLMTelemetryContext(
        workspace_name="ws",
        call_purpose="dialectic.answer",
        parent_category=parent_category,
        agent_type=agent_type,
        run_id=run_id,
        trace_id=trace_id,
        span_id=trace_id,
        session_id=session_id,
        track_name=track_name,
        iteration=iteration,
        step_seq=step_seq,
    )
    result = CompletionResult(
        content=content,
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=2,
        finish_reason=finish_reason,
        tool_calls=[
            ToolCallResult(id=f"tc-{i}", name=name, input={"q": name})
            for i, name in enumerate(tool_names or [])
        ],
        thinking_content=thinking,
    )
    return build_captured_call(
        telemetry=telemetry,
        transport="anthropic",
        provider_label=None,
        model="claude-x",
        messages=[{"role": "user", "content": "q"}],
        tools=None,
        tool_choice=None,
        result=result,
        attempt=attempt,
        was_fallback=False,
        was_stream=False,
        finish_reason=finish_reason,
        duration_ms=duration_ms,
    )


def _span(
    kind: str,
    phase: str,
    *,
    run_id: str = "r1",
    agent_type: str | None = "dialectic",
    parent_category: str = "dialectic",
    track_name: str = "Dialectic Agent",
    iteration: int | None = None,
    input: object = None,
    output: object = None,
    is_error: bool = False,
    time_ns: int | None = None,
) -> CapturedSpan:
    return CapturedSpan(
        kind=kind,
        phase=phase,
        time_ns=time_ns or time.time_ns(),
        trace_id=run_id,
        run_id=run_id,
        span_id=run_id,
        iteration=iteration,
        workspace_name="ws",
        call_purpose="dialectic.answer",
        parent_category=parent_category,
        agent_type=agent_type,
        session_id=None,
        observer="obs",
        observed="peer",
        peer_name="peer",
        track_name=track_name,
        input=input,
        output=output,
        is_error=is_error,
    )


def _tool(
    name: str,
    *,
    iteration: int | None,
    run_id: str = "r1",
    duration_ms: float = 250.0,
    is_error: bool = False,
) -> CapturedToolCall:
    return CapturedToolCall(
        run_id=run_id,
        agent_type="dialectic",
        workspace_name="ws",
        iteration=iteration,
        tool_call_seq=0,
        tool_call_id="tc-0",
        name=name,
        input={"q": name},
        output=f"{name} result",
        is_error=is_error,
        duration_ms=duration_ms,
    )


def test_single_shot_generation_is_trace_root(_exporter_env: FakeClient):
    # Deriver/summarizer style: run_id None → no run/step span, generation is root.
    client = _exporter_env
    LangfuseExporter().export(
        _call(run_id=None, trace_id="t1", track_name="Minimal Deriver")
    )

    assert len(client.observations) == 1
    gen = client.observations[0]
    assert gen.kwargs["as_type"] == "generation"
    assert gen.kwargs["trace_context"] == {"trace_id": "lf-t1"}
    assert gen.kwargs["model"] == "claude-x"
    assert gen.kwargs["usage_details"] == {
        "input": 10,
        "output": 5,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0,
    }
    # Trace attrs stamped on the root generation; no session (session_id None).
    assert gen._otel_span.attributes.get("user.id") == "tenant1"
    assert "session.id" not in gen._otel_span.attributes
    assert gen._otel_span._parent is None


def test_agentic_run_builds_run_step_generation(_exporter_env: FakeClient):
    client = _exporter_env
    LangfuseExporter().export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            session_id="sess_abc",
            track_name="Dialectic Agent",
        )
    )

    by_type: dict[str, list[FakeObs]] = {}
    for obs in client.observations:
        by_type.setdefault(str(obs.kwargs["as_type"]), []).append(obs)
    assert len(by_type["agent"]) == 1
    assert len(by_type["span"]) == 1  # step span
    assert len(by_type["generation"]) == 1

    run_span, step_span = by_type["agent"][0], by_type["span"][0]
    gen = by_type["generation"][0]
    assert run_span.kwargs["trace_context"] == {"trace_id": "lf-r1"}
    assert step_span.kwargs["trace_context"] == {
        "trace_id": "lf-r1",
        "parent_span_id": run_span.id,
    }
    assert gen.kwargs["trace_context"] == {
        "trace_id": "lf-r1",
        "parent_span_id": step_span.id,
    }
    # Trace attrs ride on every observation. The Honcho session is NOT a
    # Langfuse session (one-shot queries aren't a conversation thread) — it
    # rides in metadata as a correlation key instead.
    for obs in client.observations:
        assert "session.id" not in obs._otel_span.attributes
        assert obs._otel_span.attributes["user.id"] == "tenant1"
        assert obs._otel_span.attributes["langfuse.trace.name"] == "Dialectic Agent"
    assert run_span.kwargs["metadata"]["honcho_session"] == "sess_abc"


def test_run_span_created_once_across_iterations(_exporter_env: FakeClient):
    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export(_call(run_id="r1", trace_id="r1", iteration=1, session_id="s"))
    exporter.export(_call(run_id="r1", trace_id="r1", iteration=2, session_id="s"))

    runs = [o for o in client.observations if o.kwargs["as_type"] == "agent"]
    spans = [o for o in client.observations if o.kwargs["as_type"] == "span"]
    gens = [o for o in client.observations if o.kwargs["as_type"] == "generation"]
    # One run span shared, one step span per iteration, one generation per call.
    assert len(gens) == 2
    assert len(runs) == 1
    assert len(spans) == 2
    stamped = [o for o in client.observations if "user.id" in o._otel_span.attributes]
    assert len(stamped) == len(client.observations)


def test_langfuse_session_lru_evicts_least_recently_used(
    monkeypatch: pytest.MonkeyPatch,
):
    """Past _MAX_TRACES the least-recently-touched trace is evicted (not refused),
    so an active trace keeps its remembered span ids no matter the run volume."""
    langfuse_session.reset()
    monkeypatch.setattr(langfuse_session, "_MAX_TRACES", 2)

    langfuse_session.ensure_run_span("t1", "b", lambda: "t1-span")
    langfuse_session.ensure_run_span("t2", "b", lambda: "t2-span")
    # Touch t1 so t2 becomes the least-recently-used trace.
    assert langfuse_session.ensure_run_span("t1", "b", lambda: "ignored") == "t1-span"
    # A third trace evicts the LRU trace (t2), keeping t1.
    langfuse_session.ensure_run_span("t3", "b", lambda: "t3-span")

    created: list[str] = []
    # t1 still tracked → remembered span returned, create NOT re-invoked.
    assert (
        langfuse_session.ensure_run_span(
            "t1", "b", lambda: created.append("t1") or "new"
        )
        == "t1-span"
    )
    assert created == []
    # t2 was evicted → fresh state, create IS re-invoked.
    assert (
        langfuse_session.ensure_run_span(
            "t2", "b", lambda: created.append("t2") or "t2-span2"
        )
        == "t2-span2"
    )
    assert created == ["t2"]


def test_error_finish_marks_generation_level(_exporter_env: FakeClient):
    client = _exporter_env
    LangfuseExporter().export(
        _call(run_id=None, trace_id="t1", finish_reason="error", content="")
    )
    gen = client.observations[0]
    assert gen.kwargs["level"] == "ERROR"


def test_exporter_disabled_without_public_key_emits_nothing(
    _exporter_env: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _exporter_env
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)
    LangfuseExporter().export(_call(run_id="r1", trace_id="r1", iteration=1))
    assert client.observations == []


def test_generation_name_uses_generation_suffix(_exporter_env: FakeClient):
    client = _exporter_env
    LangfuseExporter().export(
        _call(run_id="r1", trace_id="r1", iteration=1, track_name="Dialectic Agent")
    )
    gen = [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]
    assert gen.kwargs["name"] == "Dialectic Agent generation"
    step = [o for o in client.observations if o.kwargs["as_type"] == "span"][0]
    assert step.kwargs["name"] == "Dialectic Agent step"


def _generation(client: FakeClient) -> FakeObs:
    return [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]


def test_text_only_output_stays_a_string(_exporter_env: FakeClient):
    LangfuseExporter().export(_call(run_id=None, trace_id="t1", content="hi"))
    assert _generation(_exporter_env).kwargs["output"] == "hi"


def test_tool_call_output_is_an_assistant_message_with_arguments(
    _exporter_env: FakeClient,
):
    LangfuseExporter().export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            content="let me look",
            tool_names=["search_memory"],
        )
    )
    assert _generation(_exporter_env).kwargs["output"] == {
        "role": "assistant",
        "content": "let me look",
        "tool_calls": [
            {
                "id": "tc-0",
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "arguments": '{"q": "search_memory"}',
                },
            }
        ],
    }


def test_thinking_is_exported_with_the_output(_exporter_env: FakeClient):
    LangfuseExporter().export(
        _call(run_id=None, trace_id="t1", content="answer", thinking="hmm")
    )
    assert _generation(_exporter_env).kwargs["output"] == {
        "role": "assistant",
        "content": "answer",
        "thinking": "hmm",
    }


def test_executed_tool_calls_become_spans_under_the_step(_exporter_env: FakeClient):
    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            track_name="Dialectic Agent",
            tool_names=["search_memory", "search_messages"],
        )
    )
    # Requested tool calls alone don't make spans; executed ones do.
    assert not [o for o in client.observations if o.kwargs["as_type"] == "tool"]
    exporter.export_tool_call(_tool("search_memory", iteration=1, duration_ms=300.0))
    exporter.export_tool_call(_tool("search_messages", iteration=1, is_error=True))

    step_span = [o for o in client.observations if o.kwargs["as_type"] == "span"][0]
    gen = [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]
    tools = [o for o in client.observations if o.kwargs["as_type"] == "tool"]

    assert [t.kwargs["name"] for t in tools] == ["search_memory", "search_messages"]
    # Tool spans are siblings of the generation: same parent (the step span).
    for t in tools:
        assert t.kwargs["trace_context"]["parent_span_id"] == step_span.id
    assert gen.kwargs["trace_context"]["parent_span_id"] == step_span.id
    assert tools[0].kwargs["input"] == {"q": "search_memory"}
    assert tools[0].kwargs["output"] == "search_memory result"
    assert _latency_ns(tools[0]) == 300_000_000
    assert tools[1].kwargs["level"] == "ERROR"
    assert tools[0]._otel_span.attributes["langfuse.trace.name"] == "Dialectic Agent"


def test_tool_call_without_a_run_is_skipped(_exporter_env: FakeClient):
    LangfuseExporter().export_tool_call(_tool("search_memory", iteration=1))
    assert _exporter_env.observations == []


def test_only_the_root_span_keeps_as_root(_exporter_env: FakeClient):
    # The SDK stamps AS_ROOT on every trace_context span; the exporter must
    # demote children so exactly one root survives — otherwise Langfuse races to
    # pick the trace name/root and names the trace after a child span.
    from langfuse import LangfuseOtelSpanAttributes as Attr

    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            track_name="Dialectic Agent",
            tool_names=["search_memory"],
        )
    )
    exporter.export_tool_call(_tool("search_memory", iteration=1))

    def is_demoted(obs: FakeObs) -> bool:
        return obs._otel_span.attributes.get(Attr.AS_ROOT) is False

    run_span = [o for o in client.observations if o.kwargs["as_type"] == "agent"][0]
    step_span = [o for o in client.observations if o.kwargs["as_type"] == "span"][0]
    gen = [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]
    tools = [o for o in client.observations if o.kwargs["as_type"] == "tool"]

    # Exactly one root: the run span is never demoted and loses the SDK's
    # placeholder parent; everything with a real parent is demoted.
    assert not is_demoted(run_span)
    assert run_span._otel_span._parent is None
    assert all(
        o._otel_span._parent is not None
        for o in client.observations
        if o is not run_span
    )
    assert is_demoted(step_span)
    assert is_demoted(gen)
    assert all(is_demoted(t) for t in tools)
    demoted = [o for o in client.observations if is_demoted(o)]
    assert len(demoted) == len(client.observations) - 1


def test_single_shot_generation_keeps_as_root(_exporter_env: FakeClient):
    # No parent → the generation is the trace root and must not be demoted.
    from langfuse import LangfuseOtelSpanAttributes as Attr

    client = _exporter_env
    LangfuseExporter().export(
        _call(run_id=None, trace_id="t1", track_name="Minimal Deriver")
    )
    gen = client.observations[0]
    assert gen._otel_span.attributes.get(Attr.AS_ROOT) is not False
    assert gen._otel_span._parent is None


def test_single_shot_tool_calls_are_skipped(_exporter_env: FakeClient):
    # No step span to anchor to (deriver-style); tools don't orphan to the root.
    client = _exporter_env
    LangfuseExporter().export(
        _call(run_id=None, trace_id="t1", tool_names=["search_memory"])
    )
    assert [o.kwargs["as_type"] for o in client.observations] == ["generation"]


def test_dreamer_specialists_nest_under_one_dream_root(_exporter_env: FakeClient):
    # Both specialists share ONE dream trace (run_id) and both start at
    # iteration 1. They must nest under a single synthetic "Dream" root (so the
    # trace has one root, not one per specialist) while staying distinct
    # sub-trees (no step-span collision).
    from langfuse import LangfuseOtelSpanAttributes as Attr

    client = _exporter_env
    exporter = LangfuseExporter()
    for agent_type in ("deduction", "induction"):
        exporter.export(
            _call(
                run_id="dream1",
                trace_id="dream1",
                iteration=1,
                agent_type=agent_type,
                parent_category="dream",
                track_name=f"Dreamer/{agent_type}",
            )
        )

    by_name: dict[str, list[FakeObs]] = {}
    for o in client.observations:
        by_name.setdefault(str(o.kwargs["name"]), []).append(o)
    gens = [o for o in client.observations if o.kwargs["as_type"] == "generation"]

    def is_demoted(o: FakeObs) -> bool:
        return o._otel_span.attributes.get(Attr.AS_ROOT) is False

    # Exactly one trace root: the synthetic "Dream" span — no parent, not demoted.
    roots = [
        o
        for o in client.observations
        if o.kwargs["trace_context"] == {"trace_id": "lf-dream1"}
    ]
    assert len(roots) == 1
    dream_root = roots[0]
    assert dream_root.kwargs["name"] == "Dream"
    assert dream_root.kwargs["as_type"] == "span"
    assert not is_demoted(dream_root)
    assert [o for o in client.observations if o._otel_span._parent is None] == [
        dream_root
    ]

    # Both specialist run spans hang off the Dream root and are demoted.
    run_dd = by_name["Dreamer/deduction"][0]
    run_in = by_name["Dreamer/induction"][0]
    assert len(by_name["Dreamer/deduction"]) == 1
    assert len(by_name["Dreamer/induction"]) == 1
    for rs in (run_dd, run_in):
        assert rs.kwargs["trace_context"] == {
            "trace_id": "lf-dream1",
            "parent_span_id": dream_root.id,
        }
        assert is_demoted(rs)

    # One step span per specialist, parented to its own run span; no collapsing.
    assert len(by_name["Dreamer/deduction step"]) == 1
    assert len(by_name["Dreamer/induction step"]) == 1
    assert (
        by_name["Dreamer/deduction step"][0].kwargs["trace_context"]["parent_span_id"]
        == run_dd.id
    )
    assert (
        by_name["Dreamer/induction step"][0].kwargs["trace_context"]["parent_span_id"]
        == run_in.id
    )

    # Each generation nests under its OWN specialist's step.
    assert len({g.kwargs["trace_context"]["parent_span_id"] for g in gens}) == 2

    # Trace name is the branch-agnostic "Dream" on every observation.
    assert {
        o._otel_span.attributes.get("langfuse.trace.name") for o in client.observations
    } == {"Dream"}


def _latency_ns(obs: FakeObs) -> int:
    span = obs._otel_span
    assert span._start_time is not None and span.end_time is not None
    return span.end_time - span._start_time


def test_generation_latency_matches_call_duration(_exporter_env: FakeClient):
    client = _exporter_env
    LangfuseExporter().export(_call(run_id=None, trace_id="t1", duration_ms=1500.0))

    (gen,) = client.observations
    assert _latency_ns(gen) == 1_500_000_000


def test_run_and_step_spans_start_with_their_first_generation(
    _exporter_env: FakeClient,
):
    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export(_call(run_id="r1", trace_id="r1", iteration=0, duration_ms=2000.0))
    exporter.export(_call(run_id="r1", trace_id="r1", iteration=1, duration_ms=500.0))

    run_span, step0, gen0, step1, gen1 = client.observations
    assert gen0.kwargs["as_type"] == gen1.kwargs["as_type"] == "generation"
    assert _latency_ns(gen0) == 2_000_000_000
    assert _latency_ns(gen1) == 500_000_000
    # Parents never start after the generation that minted them.
    assert run_span._otel_span._start_time == gen0._otel_span._start_time
    assert step0._otel_span._start_time == gen0._otel_span._start_time
    assert step1._otel_span._start_time == gen1._otel_span._start_time


def test_missing_duration_leaves_start_untouched(_exporter_env: FakeClient):
    client = _exporter_env
    before = time.time_ns()
    LangfuseExporter().export(_call(run_id=None, trace_id="t1"))

    (gen,) = client.observations
    assert gen._otel_span._start_time is not None
    assert gen._otel_span._start_time >= before
    assert _latency_ns(gen) >= 0


def test_non_recording_span_is_not_backdated(_exporter_env: FakeClient):
    # Langfuse-disabled clients hand back spans without an SDK start time.
    client = _exporter_env
    original = client.start_observation

    def start_observation(**kwargs: object) -> FakeObs:
        obs = original(**kwargs)
        del obs._otel_span._start_time
        return obs

    client.start_observation = start_observation
    LangfuseExporter().export(_call(run_id=None, trace_id="t1", duration_ms=10.0))

    (gen,) = client.observations
    assert not hasattr(gen._otel_span, "_start_time")
    assert gen.ended


def test_lifecycle_run_spans_the_whole_run(_exporter_env: FakeClient):
    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export_span(_span("run", "start", input="what does alice do?"))
    exporter.export_span(_span("step", "start", iteration=1))
    exporter.export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            track_name="Dialectic Agent",
            duration_ms=800.0,
        )
    )
    exporter.export_tool_call(_tool("search_memory", iteration=1))
    exporter.export_span(_span("step", "end", iteration=1))
    # Streamed tail: no step of its own.
    exporter.export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=3,
            track_name="Dialectic Agent",
            duration_ms=400.0,
        )
    )
    run, step, gen, tool, tail = client.observations
    assert not run.ended
    end_ns = time.time_ns()
    exporter.export_span(_span("run", "end", output="robots", time_ns=end_ns))

    assert run.kwargs["input"] == "what does alice do?"
    assert run.updates["output"] == "robots"
    assert run._otel_span.end_time == end_ns
    assert run._otel_span._parent is None
    assert step.kwargs["trace_context"]["parent_span_id"] == run.id
    assert gen.kwargs["trace_context"]["parent_span_id"] == step.id
    assert tool.kwargs["trace_context"]["parent_span_id"] == step.id
    assert tail.kwargs["trace_context"]["parent_span_id"] == run.id
    # Run and step cover their children, not just the first generation.
    tool_end, step_end = tool._otel_span.end_time, step._otel_span.end_time
    assert tool_end is not None and step_end is not None
    assert step_end >= tool_end
    assert end_ns >= step_end
    for obs in client.observations:
        assert obs._otel_span.attributes["user.id"] == "tenant1"
        assert obs._otel_span.attributes["langfuse.trace.name"] == "Dialectic Agent"


def test_lifecycle_run_end_marks_errors(_exporter_env: FakeClient):
    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export_span(_span("run", "start"))
    exporter.export_span(_span("step", "start", iteration=1))
    exporter.export_span(_span("run", "end", is_error=True))

    run, step = client.observations
    assert run.updates["level"] == "ERROR"
    # A step left open when its run ends is closed with it.
    assert step.ended


def test_lifecycle_dream_root_spans_both_specialists(_exporter_env: FakeClient):
    client = _exporter_env
    exporter = LangfuseExporter()
    dream = {"run_id": "d1", "parent_category": "dream"}
    exporter.export_span(
        _span("trace", "start", agent_type=None, track_name="Dream", **dream)
    )
    for agent in ("deduction", "induction"):
        track = f"Dreamer/{agent}"
        exporter.export_span(
            _span("run", "start", agent_type=agent, track_name=track, **dream)
        )
        exporter.export(
            _call(
                run_id="d1",
                trace_id="d1",
                iteration=1,
                agent_type=agent,
                parent_category="dream",
                track_name=track,
            )
        )
        exporter.export_span(
            _span("run", "end", agent_type=agent, track_name=track, **dream)
        )
    root = client.observations[0]
    assert not root.ended
    exporter.export_span(
        _span(
            "trace",
            "end",
            agent_type=None,
            track_name="Dream",
            output={"deduction": "ok"},
            **dream,
        )
    )

    assert root.kwargs["name"] == "Dream"
    assert root.updates["output"] == {"deduction": "ok"}
    assert [o for o in client.observations if o._otel_span._parent is None] == [root]
    runs = [
        o
        for o in client.observations
        if o.kwargs["as_type"] == "agent"
        and str(o.kwargs["name"]).startswith("Dreamer/")
    ]
    assert [r.kwargs["trace_context"]["parent_span_id"] for r in runs] == [
        root.id,
        root.id,
    ]
    assert all(r.ended for r in runs)
    assert {
        o._otel_span.attributes.get("langfuse.trace.name") for o in client.observations
    } == {"Dream"}


def test_reset_ends_open_lifecycle_spans(_exporter_env: FakeClient):
    exporter = LangfuseExporter()
    exporter.export_span(_span("run", "start"))
    (run,) = _exporter_env.observations
    langfuse_session.reset()
    assert run.ended


def test_eviction_ends_open_lifecycle_spans(
    _exporter_env: FakeClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(langfuse_session, "_MAX_TRACES", 1)
    exporter = LangfuseExporter()
    exporter.export_span(_span("run", "start", run_id="r1"))
    exporter.export_span(_span("run", "start", run_id="r2"))
    first, second = _exporter_env.observations
    assert first.ended
    assert not second.ended

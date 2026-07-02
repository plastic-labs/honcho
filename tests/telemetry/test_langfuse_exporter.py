# pyright: reportPrivateUsage=false, reportUnannotatedClassAttribute=false, reportUnusedFunction=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportIndexIssue=false
"""Tests for the Langfuse projection over the captured LLM stream.

Exercises `LangfuseExporter` with a fake Langfuse client so we can assert the
reconstructed trace tree (trace ids, parent linkage, names, usage, trace-level
user attributes, session-as-metadata) without a real Langfuse backend.
"""

from __future__ import annotations

import pytest

from src.config import settings
from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.capture import build_captured_call
from src.llm.types import LLMTelemetryContext
from src.telemetry import langfuse_session
from src.telemetry.langfuse_exporter import LangfuseExporter


class FakeOtelSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class FakeObs:
    _counter = 0

    def __init__(self, **kwargs: object) -> None:
        FakeObs._counter += 1
        self.id = f"obs-{FakeObs._counter}"
        self.kwargs = kwargs
        self._otel_span = FakeOtelSpan()
        self.ended = False

    def end(self) -> None:
        self.ended = True


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
    monkeypatch.setattr(settings, "LANGFUSE_EXPORTER_MODE", "exporter")
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
    assert len(by_type["span"]) == 2  # run span + step span
    assert len(by_type["generation"]) == 1

    run_span, step_span = by_type["span"]
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
    # Trace attrs stamped once, on the run span (the root). The Honcho session is
    # NOT a Langfuse session (one-shot queries aren't a conversation thread) — it
    # rides in metadata as a correlation key instead.
    assert "session.id" not in run_span._otel_span.attributes
    assert run_span._otel_span.attributes["user.id"] == "tenant1"
    assert run_span._otel_span.attributes["langfuse.trace.name"] == "Dialectic Agent"
    assert run_span.kwargs["metadata"]["honcho_session"] == "sess_abc"


def test_run_span_created_once_across_iterations(_exporter_env: FakeClient):
    client = _exporter_env
    exporter = LangfuseExporter()
    exporter.export(_call(run_id="r1", trace_id="r1", iteration=1, session_id="s"))
    exporter.export(_call(run_id="r1", trace_id="r1", iteration=2, session_id="s"))

    spans = [o for o in client.observations if o.kwargs["as_type"] == "span"]
    gens = [o for o in client.observations if o.kwargs["as_type"] == "generation"]
    # One run span shared, one step span per iteration, one generation per call.
    assert len(gens) == 2
    assert len(spans) == 3  # 1 run + 2 step
    # Trace attrs (user/name) stamped exactly once across the whole run.
    stamped = [o for o in client.observations if "user.id" in o._otel_span.attributes]
    assert len(stamped) == 1


def test_langfuse_session_lru_evicts_least_recently_used(
    monkeypatch: pytest.MonkeyPatch,
):
    """Past _MAX_TRACES the least-recently-touched trace is evicted (not refused),
    so an active trace keeps its remembered span ids no matter the run volume."""
    langfuse_session.reset()
    monkeypatch.setattr(langfuse_session, "_MAX_TRACES", 2)

    langfuse_session.ensure_run_span("t1", "b", lambda _s: "t1-span")
    langfuse_session.ensure_run_span("t2", "b", lambda _s: "t2-span")
    # Touch t1 so t2 becomes the least-recently-used trace.
    assert (
        langfuse_session.ensure_run_span("t1", "b", lambda _s: "ignored") == "t1-span"
    )
    # A third trace evicts the LRU trace (t2), keeping t1.
    langfuse_session.ensure_run_span("t3", "b", lambda _s: "t3-span")

    created: list[str] = []
    # t1 still tracked → remembered span returned, create NOT re-invoked.
    assert (
        langfuse_session.ensure_run_span(
            "t1", "b", lambda _s: created.append("t1") or "new"
        )
        == "t1-span"
    )
    assert created == []
    # t2 was evicted → fresh state, create IS re-invoked.
    assert (
        langfuse_session.ensure_run_span(
            "t2", "b", lambda _s: created.append("t2") or "t2-span2"
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


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("LANGFUSE_EXPORTER_MODE", "inline"),  # exporter off in inline mode
        ("LANGFUSE_PUBLIC_KEY", None),  # exporter off without a public key
    ],
)
def test_exporter_disabled_emits_nothing(
    _exporter_env: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
    attr: str,
    value: object,
):
    client = _exporter_env
    monkeypatch.setattr(settings, attr, value)
    LangfuseExporter().export(_call(run_id="r1", trace_id="r1", iteration=1))
    assert client.observations == []


def test_generation_name_uses_generation_suffix(_exporter_env: FakeClient):
    client = _exporter_env
    LangfuseExporter().export(
        _call(run_id="r1", trace_id="r1", iteration=1, track_name="Dialectic Agent")
    )
    gen = [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]
    assert gen.kwargs["name"] == "Dialectic Agent generation"
    step = [o for o in client.observations if o.kwargs["as_type"] == "span"][1]
    assert step.kwargs["name"] == "Dialectic Agent step"


def test_tool_calls_become_spans_under_the_step(_exporter_env: FakeClient):
    client = _exporter_env
    LangfuseExporter().export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            track_name="Dialectic Agent",
            tool_names=["search_memory", "search_messages"],
        )
    )

    spans = [o for o in client.observations if o.kwargs["as_type"] == "span"]
    gen = [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]
    tools = [o for o in client.observations if o.kwargs["as_type"] == "tool"]
    step_span = spans[1]  # run span, then step span

    assert [t.kwargs["name"] for t in tools] == ["search_memory", "search_messages"]
    # Tool spans are siblings of the generation: same parent (the step span).
    for t in tools:
        assert t.kwargs["trace_context"]["parent_span_id"] == step_span.id
    assert gen.kwargs["trace_context"]["parent_span_id"] == step_span.id
    # The model's requested input args ride on the tool span.
    assert tools[0].kwargs["input"] == {"q": "search_memory"}


def test_only_the_root_span_keeps_as_root(_exporter_env: FakeClient):
    # The SDK stamps AS_ROOT on every trace_context span; the exporter must
    # demote children so exactly one root survives — otherwise Langfuse races to
    # pick the trace name/root and names the trace after a child span.
    from langfuse import LangfuseOtelSpanAttributes as Attr

    client = _exporter_env
    LangfuseExporter().export(
        _call(
            run_id="r1",
            trace_id="r1",
            iteration=1,
            track_name="Dialectic Agent",
            tool_names=["search_memory"],
        )
    )

    def is_demoted(obs: FakeObs) -> bool:
        return obs._otel_span.attributes.get(Attr.AS_ROOT) is False

    spans = [o for o in client.observations if o.kwargs["as_type"] == "span"]
    run_span, step_span = spans[0], spans[1]
    gen = [o for o in client.observations if o.kwargs["as_type"] == "generation"][0]
    tools = [o for o in client.observations if o.kwargs["as_type"] == "tool"]

    # Exactly one root: the run span is never demoted; everything with a real
    # parent is.
    assert not is_demoted(run_span)
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

    # Trace name is the branch-agnostic "Dream", stamped exactly once — on the
    # Dream root, not on a specialist's run span.
    named = [
        o
        for o in client.observations
        if o._otel_span.attributes.get("langfuse.trace.name")
    ]
    assert len(named) == 1
    assert named[0] is dream_root
    assert named[0]._otel_span.attributes["langfuse.trace.name"] == "Dream"

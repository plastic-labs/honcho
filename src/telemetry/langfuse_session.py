"""Per-trace span registry backing the `LangfuseExporter`.

The exporter sees one `CapturedLLMCall` at a time, but a single agentic run fans
out into many calls that must nest under one run span with per-iteration step
spans. Langfuse links observations by OTEL span id, and each id is minted fresh
and unpredictable — so this module remembers the run/step span ids created for a
trace and hands them back as the `parent_span_id` of later calls.

Spans are keyed per branch (the `agent_type`) within a trace. The Dreamer's
deduction and induction specialists share one trace but are separate sub-trees;
without the branch key their iterations and generations would collide.

Per trace, it holds the trace root span id, each branch's run span id, and the
per-(branch, iteration) step span ids, so each is created once. Spans opened from
run/step lifecycle records stay open here until their end record arrives; ids
outlive the close so late children still nest.

Bounded by an LRU over traces (`_MAX_TRACES`), lock-guarded, best-effort. Evicted
or reset traces end whatever they still hold open.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# LRU window / runaway backstop — far more than the traces ever live at once; the
# least-recently-used trace is evicted past this. Not a tuning knob.
_MAX_TRACES = 4096


@dataclass
class _TraceState:
    root_span_id: str | None = None  # synthetic trace root (multi-specialist agents)
    run_span_ids: dict[str, str] = field(default_factory=dict)  # branch -> span id
    step_span_ids: dict[tuple[str, int], str] = field(
        default_factory=dict
    )  # (branch, iteration) -> span id
    open_spans: dict[str, Any] = field(default_factory=dict)  # span id -> observation
    trace_name: str | None = None


_traces: OrderedDict[str, _TraceState] = OrderedDict()
_lock = threading.Lock()


def _get_or_create_state(trace_key: str) -> _TraceState:
    """Return the `_TraceState` for `trace_key`, creating it if new and marking it
    most-recently-used. Caller MUST hold `_lock`. Bounded by an LRU: a new trace
    past `_MAX_TRACES` evicts the least-recently-used (almost always finished) one.
    """
    state = _traces.get(trace_key)
    if state is None:
        if len(_traces) >= _MAX_TRACES:
            _, evicted = _traces.popitem(last=False)  # evict the LRU trace
            _end_all(list(evicted.open_spans.values()))
        state = _traces[trace_key] = _TraceState()
    else:
        _traces.move_to_end(trace_key)  # mark most-recently-used
    return state


def ensure_trace_root(trace_key: str, create: Callable[[], str | None]) -> str | None:
    """Return the single trace-root span id for `trace_key`, creating it once.

    Used by multi-specialist agents (the Dreamer) whose branches share one trace
    but must all hang off ONE root span. Single-specialist agents don't call
    this. Mirrors `ensure_run_span`'s retry-on-None: a failed create just yields
    None (the caller then roots the branch directly) and is retried next call.
    """
    with _lock:
        state = _get_or_create_state(trace_key)
        if state.root_span_id is None:
            state.root_span_id = create()
        return state.root_span_id


def ensure_run_span(
    trace_key: str, branch: str, create: Callable[[], str | None]
) -> str | None:
    """Return the run span id for `(trace_key, branch)`, creating it once.

    `create` must not re-enter this module (it runs under the lock).
    """
    with _lock:
        state = _get_or_create_state(trace_key)
        existing = state.run_span_ids.get(branch)
        if existing is None:
            existing = create()
            if existing is not None:
                state.run_span_ids[branch] = existing
        return existing


def ensure_step_span(
    trace_key: str, branch: str, iteration: int, create: Callable[[], str | None]
) -> str | None:
    """Return the step span id for `(trace_key, branch, iteration)`, creating once."""
    with _lock:
        state = _get_or_create_state(trace_key)
        key = (branch, iteration)
        existing = state.step_span_ids.get(key)
        if existing is None:
            existing = create()
            if existing is not None:
                state.step_span_ids[key] = existing
        return existing


def register_root(trace_key: str, span_id: str, obs: Any) -> None:
    """Track an open trace-root observation."""
    with _lock:
        state = _get_or_create_state(trace_key)
        state.root_span_id = span_id
        state.open_spans[span_id] = obs


def register_run(trace_key: str, branch: str, span_id: str, obs: Any) -> None:
    """Track an open run observation for `branch`."""
    with _lock:
        state = _get_or_create_state(trace_key)
        state.run_span_ids[branch] = span_id
        state.open_spans[span_id] = obs


def register_step(
    trace_key: str, branch: str, iteration: int, span_id: str, obs: Any
) -> None:
    """Track an open step observation for `(branch, iteration)`."""
    with _lock:
        state = _get_or_create_state(trace_key)
        state.step_span_ids[(branch, iteration)] = span_id
        state.open_spans[span_id] = obs


def run_span_id(trace_key: str, branch: str) -> str | None:
    """The run span id for `(trace_key, branch)`, if one exists."""
    with _lock:
        state = _traces.get(trace_key)
        return state.run_span_ids.get(branch) if state else None


def step_span_id(trace_key: str, branch: str, iteration: int) -> str | None:
    """The step span id for `(trace_key, branch, iteration)`, if one exists."""
    with _lock:
        state = _traces.get(trace_key)
        return state.step_span_ids.get((branch, iteration)) if state else None


def is_open(trace_key: str, span_id: str | None) -> bool:
    """True when `span_id` was opened from a lifecycle record and hasn't ended."""
    with _lock:
        state = _traces.get(trace_key)
        return bool(state and span_id and span_id in state.open_spans)


def close_step(trace_key: str, branch: str, iteration: int) -> Any | None:
    """Stop tracking the open step for `(branch, iteration)`, returning it."""
    with _lock:
        state = _traces.get(trace_key)
        span_id = state.step_span_ids.get((branch, iteration)) if state else None
        if state is None or span_id is None:
            return None
        return state.open_spans.pop(span_id, None)


def close_run(trace_key: str, branch: str) -> tuple[Any | None, list[Any]]:
    """Stop tracking `branch`'s open run, returning it and its still-open steps."""
    with _lock:
        state = _traces.get(trace_key)
        if state is None:
            return None, []
        steps = [
            obs
            for (step_branch, _), span_id in state.step_span_ids.items()
            if step_branch == branch
            and (obs := state.open_spans.pop(span_id, None)) is not None
        ]
        span_id = state.run_span_ids.get(branch)
        run = state.open_spans.pop(span_id, None) if span_id else None
        return run, steps


def close_root(trace_key: str) -> tuple[Any | None, list[Any]]:
    """Stop tracking the open trace root, returning it and everything else open."""
    with _lock:
        state = _traces.get(trace_key)
        if state is None:
            return None, []
        root_id = state.root_span_id
        root = state.open_spans.pop(root_id, None) if root_id else None
        rest = list(state.open_spans.values())
        state.open_spans.clear()
        return root, rest


def note_trace_name(trace_key: str, name: str | None) -> None:
    """Remember the trace's name for observations that can't derive it."""
    if not name:
        return
    with _lock:
        state = _get_or_create_state(trace_key)
        if state.trace_name is None:
            state.trace_name = name


def trace_name(trace_key: str) -> str | None:
    """The remembered trace name, if any."""
    with _lock:
        state = _traces.get(trace_key)
        return state.trace_name if state else None


def _end_all(observations: list[Any]) -> None:
    for obs in observations:
        try:
            obs.end()
        except Exception:  # pragma: no cover - best-effort telemetry
            logger.debug("Failed to end Langfuse observation", exc_info=True)


def reset() -> None:
    """Drop all tracked traces, ending open spans — used on shutdown and in tests."""
    with _lock:
        open_spans = [
            obs for state in _traces.values() for obs in state.open_spans.values()
        ]
        _traces.clear()
    _end_all(open_spans)

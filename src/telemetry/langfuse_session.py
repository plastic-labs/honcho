"""Per-trace span registry backing the `LangfuseExporter`.

The exporter sees one `CapturedLLMCall` at a time, but a single agentic run fans
out into many calls that must nest under one run span with per-iteration step
spans. Langfuse links observations by OTEL span id, and each id is minted fresh
and unpredictable — so this module remembers the run/step span ids created for a
trace and hands them back as the `parent_span_id` of later calls.

Spans are keyed per branch (the `agent_type`) within a trace. The Dreamer's
deduction and induction specialists share one trace but are separate sub-trees;
without the branch key their iterations and generations would collide.

Per trace, it holds each branch's run span id, the per-(branch, iteration) step
span ids, and whether trace-level attrs have been stamped — so each is created
once. The stamp decision is made inside `ensure_run_span` under the lock so it
can't double-fire across branches.

Bounded by an LRU over traces (`_MAX_TRACES`), lock-guarded, best-effort.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

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
    attrs_stamped: bool = False


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
            _traces.popitem(last=False)  # evict the least-recently-used trace
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
    trace_key: str, branch: str, create: Callable[[bool], str | None]
) -> str | None:
    """Return the run span id for `(trace_key, branch)`, creating it once.

    `create` receives `should_stamp` — True exactly once per trace, on the first
    branch's run span — and builds the Langfuse run span (stamping trace-level
    attrs iff asked), returning its span id (or None on failure). The stamp
    decision is computed here, under the lock, so it can't double-fire across the
    Dreamer's two specialist branches; `create` must not re-enter this module.
    """
    with _lock:
        state = _get_or_create_state(trace_key)
        existing = state.run_span_ids.get(branch)
        if existing is None:
            should_stamp = not state.attrs_stamped
            existing = create(should_stamp)
            if existing is not None:
                state.run_span_ids[branch] = existing
                if should_stamp:
                    state.attrs_stamped = True
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


def reset() -> None:
    """Drop all tracked traces — used on shutdown and in tests."""
    with _lock:
        _traces.clear()

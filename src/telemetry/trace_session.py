"""Per-run content dedup for the trace stream — makes bandwidth O(N).

Tracks the set of content hashes a run has already shipped, so each unique
message ships its `trace.content` exactly once per run.

The set is bounded by an LRU over runs: once more than `_MAX_RUNS` runs are
tracked, the least-recently-touched one is evicted (almost always a run that has
already finished), so dedup keeps working for active runs no matter how many the
process has handled. A single run exceeding `_MAX_HASHES_PER_RUN` unique messages
stops deduping and emits-anyway, bumping a metric — that loss is measured.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

# LRU window over runs + per-run hash cap. Generous — a run rarely has more than
# a few hundred unique messages, and far fewer than _MAX_RUNS are ever live at
# once; both are runaway backstops, not tuning knobs.
_MAX_RUNS = 4096
_MAX_HASHES_PER_RUN = 8192

# trace_id (fallback span_id) → set of content hashes already shipped this run.
# OrderedDict so we can evict the least-recently-used run when over _MAX_RUNS.
_runs: OrderedDict[str, set[str]] = OrderedDict()
_lock = threading.Lock()


def mark_emitted(run_key: str, content_hash: str) -> bool:
    """Return True if this hash should be shipped for ``run_key`` (first time),
    False if already shipped this run (skip the ``trace.content``).

    A run exceeding `_MAX_HASHES_PER_RUN` returns True (emit-anyway) and records a
    drop of the dedup *guarantee* — the event still ships, we just stopped
    tracking. Tracking a new run past `_MAX_RUNS` evicts the LRU run instead (a
    routine, lossless bound — the evicted run is almost always already finished).
    """
    with _lock:
        seen = _runs.get(run_key)
        if seen is None:
            if len(_runs) >= _MAX_RUNS:
                _runs.popitem(last=False)  # evict the least-recently-used run
            seen = _runs[run_key] = set()
        else:
            _runs.move_to_end(run_key)  # mark most-recently-used
        if content_hash in seen:
            return False
        if len(seen) >= _MAX_HASHES_PER_RUN:
            _record_overflow("max_hashes")
            return True
        seen.add(content_hash)
        return True


def reset() -> None:
    """Drop all tracked runs — used on shutdown and in tests."""
    with _lock:
        _runs.clear()


def _record_overflow(reason: str) -> None:
    try:
        from src.telemetry import prometheus_metrics

        prometheus_metrics.record_telemetry_event_dropped(
            reason=f"trace_dedup_{reason}"
        )
    except Exception:  # pragma: no cover - best-effort telemetry
        logger.debug("trace dedup overflow (%s)", reason)

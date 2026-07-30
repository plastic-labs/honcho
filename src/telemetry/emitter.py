"""
Buffered CloudEvents HTTP emitter with retry logic.

This module provides a TelemetryEmitter class that:
- Buffers events in memory for efficient batching
- Sends events as CloudEvents to a configurable HTTP endpoint
- Implements exponential backoff retry logic
- Supports graceful shutdown with flush
"""

import asyncio
import contextlib
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import httpx
from cloudevents.conversion import to_json  # pyright: ignore[reportUnknownVariableType]
from cloudevents.http import CloudEvent

from src._version import HONCHO_VERSION

if TYPE_CHECKING:
    from src.telemetry.events.base import BaseEvent

logger = logging.getLogger(__name__)


def _should_sample(
    event: "BaseEvent", rate: object, *, event_id: str | None = None
) -> bool:
    """Trace-coherent deterministic sampler for high-volume events.

    `rate` is typed as `object` (rather than `float`) because the caller
    reads it straight from `settings.TELEMETRY.HIGH_VOLUME_SAMPLE_RATE`,
    which in tests gets MagicMock'd. A MagicMock comparison against 1.0
    raises TypeError, so we validate at the boundary and fall back to
    passthrough on anything non-numeric.

    When the event carries a `run_id`, sampling decisions hash on that id —
    so every event in an agent run either passes or fails the sampler, and
    join queries downstream don't see half-traces. Events without `run_id`
    (summarizer, deriver — non-agentic call sites) sample independently per
    event using the event's deterministic id. Callers that have already
    computed `event.generate_id()` can pass it as `event_id` to avoid the
    redundant sha256 hash.
    """
    if not isinstance(rate, int | float):
        return True
    rate_f = float(rate)
    if rate_f >= 1.0:
        return True
    if rate_f <= 0.0:
        return False
    run_id = getattr(event, "run_id", None)
    if isinstance(run_id, str) and run_id:
        key = run_id
    elif event_id is not None:
        key = event_id
    else:
        key = event.generate_id()
    # Stable hash → 0..9999 → compare against rate * 10000.
    bucket = int.from_bytes(_stable_hash(key)[:4], "big") % 10_000
    return bucket < int(rate_f * 10_000)


def _stable_hash(value: str) -> bytes:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).digest()


class TelemetryEmitter:
    """Buffered, async CloudEvents emitter with retry logic.

    This emitter queues events and sends them in batches to reduce
    network overhead. It supports automatic periodic flushing and
    exponential backoff retry on failure.

    Usage:
        emitter = TelemetryEmitter(
            endpoint="https://telemetry.example.com/v1/events",
            headers={"X-API-Key": "..."},
        )
        await emitter.start()

        # Queue events (non-blocking)
        emitter.emit(my_event)

        # Graceful shutdown
        await emitter.shutdown()
    """

    endpoint: str | None
    headers: dict[str, str]
    batch_size: int
    flush_interval: float
    flush_threshold: int
    max_retries: int
    max_buffer_size: int
    enabled: bool
    drop_reason_prefix: str
    _buffer: deque[CloudEvent]
    _flush_task: asyncio.Task[None] | None
    _client: httpx.AsyncClient | None
    _running: bool
    _lock: asyncio.Lock
    _capacity_warning_active: bool
    _pending_flush_tasks: set[asyncio.Task[None]]

    def __init__(
        self,
        endpoint: str | None = None,
        headers: dict[str, str] | None = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 1.0,
        flush_threshold: int = 50,
        max_retries: int = 3,
        max_buffer_size: int = 10000,
        enabled: bool = True,
        drop_reason_prefix: str = "",
    ):
        """Initialize the telemetry emitter.

        Args:
            endpoint: CloudEvents HTTP endpoint URL
            headers: Optional HTTP headers for authentication
            batch_size: Maximum events per batch
            flush_interval_seconds: How often to flush the buffer
            flush_threshold: Trigger flush when buffer reaches this size
            max_retries: Maximum retry attempts on failure
            max_buffer_size: Maximum events to buffer (oldest dropped if exceeded)
            enabled: Whether emission is enabled
            drop_reason_prefix: Prefix for the dropped-event metric reason label
                (e.g. "trace_") so a second emitter's drops are distinguishable
                from the primary metrics emitter's in Prometheus.
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.flush_threshold = flush_threshold
        self.max_retries = max_retries
        self.max_buffer_size = max_buffer_size
        self.enabled = enabled and endpoint is not None
        self.drop_reason_prefix = drop_reason_prefix

        self._buffer = deque(maxlen=max_buffer_size)
        self._flush_task = None
        self._client = None
        self._running = False
        self._lock = asyncio.Lock()
        self._capacity_warning_active = False
        self._pending_flush_tasks = set()

    async def start(self) -> None:
        """Start the emitter background tasks.

        Creates the HTTP client and starts the periodic flush task.
        """
        if not self.enabled:
            logger.info("Telemetry emitter disabled (no endpoint or disabled)")
            return

        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Content-Type": "application/cloudevents+json",
                **self.headers,
            },
        )
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Telemetry emitter started, endpoint: %s", self.endpoint)

    async def shutdown(self) -> None:
        """Gracefully shutdown the emitter.

        Stops the periodic flush task, drains any in-flight threshold
        flushes, flushes remaining events, and closes the HTTP client.

        Threshold flushes are spawned from emit() and pop their batch
        under lock before releasing it for the HTTP send. If we don't
        await those tasks first, the final flush() can see an empty
        buffer and return while the in-flight task is still mid-send —
        closing the HTTP client then orphans that batch.
        """
        if not self.enabled:
            return

        self._running = False

        # Cancel the periodic flush task
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task

        # Drain in-flight threshold flushes. Snapshot the set first because
        # the done-callback mutates it. Exceptions here mustn't block shutdown.
        pending = list(self._pending_flush_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Final flush of remaining events
        await self.flush()

        # Close HTTP client
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        logger.info("Telemetry emitter shutdown complete")

    def emit(self, event: "BaseEvent") -> None:
        """Queue an event for emission. Non-blocking.

        The event is converted to a CloudEvent and added to the buffer.
        If the buffer is full, the oldest events are dropped.
        Triggers an immediate flush when buffer reaches flush_threshold.

        Args:
            event: The Honcho event to emit
        """
        if not self.enabled:
            return

        from src.config import settings
        from src.telemetry.prometheus.metrics import prometheus_metrics

        # High-volume events are subject to HIGH_VOLUME_SAMPLE_RATE. Aggregate
        # envelopes (representation.completed, dialectic.completed, dream.run,
        # etc.) declare _volume_class="ground_truth" and skip the sampler.
        # Sampling is deterministic on run_id when available so an entire
        # agentic trace is either fully kept or fully dropped — never a
        # half-sampled run that breaks join queries downstream.
        #
        # Trade-off: at rate < 1.0, ground_truth aggregates still emit but
        # their high-volume children get sampled out. Downstream JOIN ... ON
        # run_id sees orphaned parents; aggregates carry totals so this is
        # intentional, but per-call analytics rebuilt from the sampled
        # children alone will undercount. See HIGH_VOLUME_SAMPLE_RATE
        # docstring in src/config.py for the full implications.
        # Lazy event_id generation. The sampler only needs it for high-volume
        # events without a run_id (run_id events sample on run_id directly).
        # For sampled-out children of an agent run, deferring saves a sha256
        # hash per event; for events that pass the sampler or skip it, we
        # still only compute the id once and reuse it for the CloudEvent.
        event_id: str | None = None

        if event.volume_class() == "high_volume":
            run_id = getattr(event, "run_id", None)
            has_run_id = isinstance(run_id, str) and bool(run_id)
            if not has_run_id:
                event_id = event.generate_id()
            if not _should_sample(
                event,
                settings.TELEMETRY.HIGH_VOLUME_SAMPLE_RATE,
                event_id=event_id,
            ):
                prometheus_metrics.record_telemetry_event_sampled_out(
                    event_type=event.event_type()
                )
                return

        if event_id is None:
            event_id = event.generate_id()

        # Build source with namespace for tenant routing
        # Format: /honcho/{namespace}/{category} or /honcho/{category}
        namespace = settings.TELEMETRY.NAMESPACE
        source = f"/honcho/{event.category()}"
        if namespace:
            source = f"/honcho/{namespace}/{event.category()}"

        # Build CloudEvent attributes
        attributes: dict[str, str] = {
            "id": event_id,
            "source": source,
            "type": event.event_type(),
            "time": event.timestamp.isoformat(),
            "dataschema": f"https://honcho.dev/schemas/{event.event_type()}/v{event.schema_version()}",
        }

        # Build body and inject envelope-level identity. We do NOT mutate the event
        # instance — tests and callers that observe the event after emit() see it
        # unchanged. Only the serialized body that hits the wire carries the extras.
        body: dict[str, Any] = event.model_dump(mode="json")
        body["honcho_version"] = HONCHO_VERSION

        # Buffer-full check happens here because deque(maxlen=) silently evicts.
        # Detect by length-before-append; if at capacity, the append will displace
        # the oldest event — that's a drop.
        will_drop_oldest = len(self._buffer) >= self.max_buffer_size

        # Create CloudEvent
        cloud_event = CloudEvent(attributes, body)

        if will_drop_oldest:
            prometheus_metrics.record_telemetry_event_dropped(
                reason=f"{self.drop_reason_prefix}buffer_full"
            )

        self._buffer.append(cloud_event)
        buffer_size = len(self._buffer)
        prometheus_metrics.record_telemetry_event_emitted(event_type=event.event_type())
        prometheus_metrics.set_telemetry_buffer_size(size=buffer_size)
        logger.debug("Queued event %s (buffer size: %d)", event_id, buffer_size)

        # Warning logs as buffer approaches max capacity. Edge-triggered so
        # sustained backpressure doesn't spam thousands of WARN lines per
        # second — exactly when log pipelines are most fragile.
        capacity_ratio = buffer_size / self.max_buffer_size
        if capacity_ratio >= 0.8:
            if not self._capacity_warning_active:
                self._capacity_warning_active = True
                logger.warning(
                    "Telemetry buffer at %.0f%% capacity (%d/%d events)",
                    capacity_ratio * 100,
                    buffer_size,
                    self.max_buffer_size,
                )
        else:
            self._capacity_warning_active = False

        logger.debug("Event added to emitter (buffer size: %d)", buffer_size)
        # Threshold-based flush trigger. emit() is sync — guard against the
        # case where a future caller invokes it from outside an event loop;
        # the periodic flush task will still pick up the buffered events.
        # Track spawned tasks so shutdown() can await them before closing the
        # HTTP client — otherwise an in-flight threshold flush (which pops its
        # batch under lock, then sends without the lock) can be orphaned and
        # lose its batch when the client closes underneath it.
        if buffer_size >= self.flush_threshold and self._running:
            logger.debug("Triggering flush (buffer size: %d)", buffer_size)
            try:
                asyncio.get_running_loop()
                flush_task = asyncio.create_task(self.flush())
                self._pending_flush_tasks.add(flush_task)
                flush_task.add_done_callback(self._pending_flush_tasks.discard)
            except RuntimeError:
                logger.debug(
                    "emit() called outside an event loop; deferring flush to periodic task"
                )

    async def flush(self) -> None:
        """Flush buffered events to the endpoint.

        Sends events in batches up to batch_size. Uses exponential
        backoff retry on failure. Events are returned to the buffer
        on permanent failure.

        The lock is held only for buffer mutations (pop batch / restore on
        failure) — never across the HTTP send. A failing endpoint can spend
        tens of seconds in retry + backoff; keeping that out of the lock
        lets concurrent flushers make progress on disjoint batches.
        """
        if not self.enabled or not self._buffer or self._client is None:
            return

        while True:
            async with self._lock:
                if not self._buffer:
                    return
                batch: list[CloudEvent] = []
                while self._buffer and len(batch) < self.batch_size:
                    batch.append(self._buffer.popleft())

            if not batch:
                return

            success = await self._send_batch(batch)
            if success:
                continue

            from src.telemetry.prometheus.metrics import prometheus_metrics

            async with self._lock:
                # Put events back at the front of the buffer. If the buffer is
                # already full, deque.appendleft silently evicts from the right
                # — those events are lost. Count the eviction as send_failed.
                for event in reversed(batch):
                    if len(self._buffer) >= self.max_buffer_size:
                        prometheus_metrics.record_telemetry_event_dropped(
                            reason=f"{self.drop_reason_prefix}send_failed"
                        )
                    self._buffer.appendleft(event)
            logger.warning(
                "Failed to send batch of %d events, returned to buffer",
                len(batch),
            )
            return

    async def _send_batch(self, batch: list[CloudEvent]) -> bool:
        """Send a batch of events to the endpoint with retry logic.

        Args:
            batch: List of CloudEvents to send

        Returns:
            True if successful, False if all retries failed
        """
        if self._client is None or self.endpoint is None:
            return False

        for attempt in range(self.max_retries):
            try:
                # Convert batch to JSON
                # For a single event, use structured format
                # For multiple events, use batch format (array)
                if len(batch) == 1:
                    body = to_json(batch[0])
                else:
                    # CloudEvents batch format is a JSON array
                    events_json = [json.loads(to_json(e)) for e in batch]
                    body = json.dumps(events_json).encode()

                response = await self._client.post(
                    self.endpoint,
                    content=body,
                    headers=(
                        {"Content-Type": "application/cloudevents-batch+json"}
                        if len(batch) > 1
                        else None
                    ),
                )
                response.raise_for_status()

                logger.debug(
                    "Sent batch of %d events (status: %d)",
                    len(batch),
                    response.status_code,
                )
                return True

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "HTTP error sending events (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
            except httpx.RequestError as e:
                logger.warning(
                    "Request error sending events (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = 2**attempt
                await asyncio.sleep(delay)

        return False

    async def _periodic_flush(self) -> None:
        """Background task that periodically flushes the buffer."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                if self._buffer:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in periodic flush: %s", e)

    @property
    def buffer_size(self) -> int:
        """Return the current number of buffered events."""
        return len(self._buffer)

    @property
    def is_running(self) -> bool:
        """Return whether the emitter is running."""
        return self._running


# Global emitter instance (initialized by initialize_telemetry_events)
_emitter: TelemetryEmitter | None = None


def get_emitter() -> TelemetryEmitter | None:
    """Get the global telemetry emitter instance."""
    return _emitter


async def initialize_emitter(
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    batch_size: int = 100,
    flush_interval_seconds: float = 1.0,
    flush_threshold: int = 50,
    max_retries: int = 3,
    max_buffer_size: int = 10000,
    enabled: bool = True,
) -> TelemetryEmitter:
    """Initialize and start the global telemetry emitter.

    Args:
        endpoint: CloudEvents HTTP endpoint URL
        headers: Optional HTTP headers for authentication
        batch_size: Maximum events per batch
        flush_interval_seconds: How often to flush the buffer
        flush_threshold: Trigger flush when buffer reaches this size
        max_retries: Maximum retry attempts on failure
        max_buffer_size: Maximum events to buffer
        enabled: Whether emission is enabled

    Returns:
        The initialized emitter instance
    """
    global _emitter

    _emitter = TelemetryEmitter(
        endpoint=endpoint,
        headers=headers,
        batch_size=batch_size,
        flush_interval_seconds=flush_interval_seconds,
        flush_threshold=flush_threshold,
        max_retries=max_retries,
        max_buffer_size=max_buffer_size,
        enabled=enabled,
    )
    await _emitter.start()
    return _emitter


async def shutdown_emitter() -> None:
    """Shutdown the global telemetry emitter."""
    global _emitter
    if _emitter is not None:
        await _emitter.shutdown()
        _emitter = None


# Separate emitter for the full-fidelity trace stream (llm.call.traced /
# trace.content). Kept distinct from the metrics `_emitter` so a trace burst
# can never evict billing events from the metrics buffer.
_trace_emitter: TelemetryEmitter | None = None


def get_trace_emitter() -> TelemetryEmitter | None:
    """Get the global trace-stream emitter instance (None when payload tracing off)."""
    return _trace_emitter


async def initialize_trace_emitter(
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    batch_size: int = 100,
    flush_interval_seconds: float = 1.0,
    flush_threshold: int = 50,
    max_retries: int = 3,
    max_buffer_size: int = 10000,
    enabled: bool = True,
) -> TelemetryEmitter:
    """Initialize and start the global trace-stream emitter.

    Drops are recorded under the ``trace_`` reason prefix so they're
    distinguishable from the metrics emitter's drops in Prometheus.
    """
    global _trace_emitter

    _trace_emitter = TelemetryEmitter(
        endpoint=endpoint,
        headers=headers,
        batch_size=batch_size,
        flush_interval_seconds=flush_interval_seconds,
        flush_threshold=flush_threshold,
        max_retries=max_retries,
        max_buffer_size=max_buffer_size,
        enabled=enabled,
        drop_reason_prefix="trace_",
    )
    await _trace_emitter.start()
    return _trace_emitter


async def shutdown_trace_emitter() -> None:
    """Shutdown the global trace-stream emitter."""
    global _trace_emitter
    if _trace_emitter is not None:
        await _trace_emitter.shutdown()
        _trace_emitter = None

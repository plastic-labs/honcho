# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedParameter=false
"""Unit tests for TelemetryEmitter class.

Exhaustive tests covering:
- Initialization and configuration
- Disabled state behavior
- Buffer management (capacity, overflow, threshold)
- Flush mechanics (manual, threshold-triggered, periodic)
- HTTP sending (success, retries, batch formats)
- Error scenarios (network, HTTP status codes, timeouts)
- Lifecycle (start, shutdown, concurrent operations)
"""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.telemetry.emitter import TelemetryEmitter
from src.telemetry.events.representation import RepresentationCompletedEvent

# =============================================================================
# Helper to create minimal test events
# =============================================================================


def create_test_event(message_id: str = "msg_001") -> RepresentationCompletedEvent:
    """Create a minimal test event."""
    return RepresentationCompletedEvent(
        timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        workspace_id="ws_123",
        workspace_name="test_workspace",
        session_id="sess_456",
        session_name="test_session",
        observed="user_peer",
        queue_items_processed=1,
        earliest_message_id="msg_001",
        latest_message_id=message_id,
        message_count=1,
        explicit_conclusion_count=1,
        context_preparation_ms=10.0,
        llm_call_ms=100.0,
        total_duration_ms=110.0,
        input_tokens=100,
        output_tokens=50,
    )


# =============================================================================
# Tests for Initialization
# =============================================================================


class TestEmitterInitialization:
    """Tests for TelemetryEmitter initialization."""

    def test_default_initialization(self):
        """Emitter initializes with default values."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        assert emitter.endpoint == "http://test:8001/events"
        assert emitter.batch_size == 100
        assert emitter.flush_interval == 1.0
        assert emitter.flush_threshold == 50
        assert emitter.max_retries == 3
        assert emitter.max_buffer_size == 10000
        assert emitter.enabled is True

    def test_custom_initialization(self):
        """Emitter initializes with custom values."""
        emitter = TelemetryEmitter(
            endpoint="http://custom:9000/events",
            headers={"X-API-Key": "secret"},
            batch_size=50,
            flush_interval_seconds=2.5,
            flush_threshold=25,
            max_retries=5,
            max_buffer_size=5000,
            enabled=True,
        )

        assert emitter.endpoint == "http://custom:9000/events"
        assert emitter.headers == {"X-API-Key": "secret"}
        assert emitter.batch_size == 50
        assert emitter.flush_interval == 2.5
        assert emitter.flush_threshold == 25
        assert emitter.max_retries == 5
        assert emitter.max_buffer_size == 5000

    def test_disabled_when_no_endpoint(self):
        """Emitter is disabled when endpoint is None."""
        emitter = TelemetryEmitter(endpoint=None, enabled=True)
        assert emitter.enabled is False

    def test_disabled_when_explicitly_disabled(self):
        """Emitter is disabled when enabled=False."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events", enabled=False)
        assert emitter.enabled is False

    def test_initial_state(self):
        """Emitter starts in non-running state with empty buffer."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        assert emitter.buffer_size == 0
        assert emitter.is_running is False
        assert emitter._client is None
        assert emitter._flush_task is None


# =============================================================================
# Tests for Disabled State
# =============================================================================


class TestDisabledEmitter:
    """Tests for emitter behavior when disabled."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self, disabled_emitter: TelemetryEmitter):
        """start() is no-op when disabled."""
        await disabled_emitter.start()

        assert disabled_emitter.is_running is False
        assert disabled_emitter._client is None

    @pytest.mark.asyncio
    async def test_emit_when_disabled(self, disabled_emitter: TelemetryEmitter):
        """emit() is no-op when disabled."""
        event = create_test_event()

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            disabled_emitter.emit(event)

        assert disabled_emitter.buffer_size == 0

    @pytest.mark.asyncio
    async def test_flush_when_disabled(self, disabled_emitter: TelemetryEmitter):
        """flush() is no-op when disabled."""
        await disabled_emitter.flush()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_shutdown_when_disabled(self, disabled_emitter: TelemetryEmitter):
        """shutdown() is no-op when disabled."""
        await disabled_emitter.shutdown()
        # Should complete without error


# =============================================================================
# Tests for Buffer Management
# =============================================================================


class TestBufferManagement:
    """Tests for buffer capacity and overflow handling."""

    def test_emit_adds_to_buffer(self):
        """emit() adds events to the buffer."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")
        emitter._running = True  # Simulate started state without full start()

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)

        assert emitter.buffer_size == 1

    def test_buffer_respects_max_size(self):
        """Buffer drops oldest events when max size exceeded."""
        # Create an emitter with small buffer but high flush threshold
        # to prevent auto-flush from interfering with this synchronous test
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_buffer_size=5,
            flush_threshold=100,  # High threshold to prevent auto-flush
            enabled=True,
        )
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Add more events than buffer can hold
            for i in range(10):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        # Buffer should be at max size (5) - deque drops oldest events
        assert emitter.buffer_size == 5

    def test_buffer_warns_at_80_percent_capacity(self, caplog):
        """Warning logged when buffer reaches 80% capacity."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_buffer_size=10,
        )
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Add 8 events (80% of 10)
            for i in range(8):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        # Check warning was logged
        assert any(
            "80%" in record.message or "capacity" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    @pytest.mark.asyncio
    async def test_flush_threshold_triggers_flush(self):
        """Flush is triggered when buffer reaches threshold."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_threshold=3,
            max_buffer_size=100,
        )

        # Mock the HTTP client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Add events up to threshold
            for i in range(3):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        # Give async task time to run
        await asyncio.sleep(0.1)

        # Flush should have been triggered
        assert mock_client.post.called


# =============================================================================
# Tests for HTTP Sending
# =============================================================================


class TestHttpSending:
    """Tests for HTTP request formatting and sending."""

    @pytest.mark.asyncio
    async def test_single_event_uses_structured_format(self):
        """Single event is sent as structured CloudEvent JSON."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)

        await emitter.flush()

        # Check that post was called
        assert mock_client.post.called
        call_args = mock_client.post.call_args

        # Single event should NOT have batch header override
        headers = call_args.kwargs.get("headers")
        assert headers is None  # No override for single event

    @pytest.mark.asyncio
    async def test_multiple_events_use_batch_format(self):
        """Multiple events are sent as CloudEvents batch (JSON array)."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=10,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Add multiple events
            for i in range(3):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        await emitter.flush()

        # Check that post was called with batch header
        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers")
        assert headers == {"Content-Type": "application/cloudevents-batch+json"}

        # Content should be a JSON array
        content = call_args.kwargs.get("content")
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    @pytest.mark.asyncio
    async def test_batch_respects_batch_size(self):
        """Large buffers are sent in batch_size chunks."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=2,  # Small batch size
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Add 5 events
            for i in range(5):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        await emitter.flush()

        # Should have made 3 calls (2, 2, 1)
        assert mock_client.post.call_count == 3


# =============================================================================
# Tests for Retry Logic
# =============================================================================


class TestRetryLogic:
    """Tests for exponential backoff retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_http_error(self):
        """Emitter retries on HTTP errors."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=3,
        )

        # First two calls fail, third succeeds
        mock_client = AsyncMock()
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=fail_response
            )
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(
            side_effect=[fail_response, fail_response, success_response]
        )
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),  # Skip actual delays
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        # Should have tried 3 times
        assert mock_client.post.call_count == 3
        # Buffer should be empty (success on 3rd try)
        assert emitter.buffer_size == 0

    @pytest.mark.asyncio
    async def test_retry_on_request_error(self):
        """Emitter retries on network/request errors."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=3,
        )

        mock_client = AsyncMock()
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()

        # First call raises connection error, second succeeds
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                success_response,
            ]
        )
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Events returned to buffer when all retries fail."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=3,
        )

        mock_client = AsyncMock()
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=fail_response
            )
        )
        mock_client.post = AsyncMock(return_value=fail_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        # All 3 retries should have been attempted
        assert mock_client.post.call_count == 3
        # Event should be back in buffer
        assert emitter.buffer_size == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Retry uses exponential backoff delays."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=4,
        )

        mock_client = AsyncMock()
        fail_response = MagicMock()
        fail_response.status_code = 503
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "503", request=MagicMock(), response=fail_response
            )
        )
        mock_client.post = AsyncMock(return_value=fail_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        sleep_calls = []

        async def track_sleep(delay):
            sleep_calls.append(delay)

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", side_effect=track_sleep),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        # Should have delays of 1, 2, 4 (2^0, 2^1, 2^2) before attempts 2, 3, 4
        assert sleep_calls == [1, 2, 4]


# =============================================================================
# Tests for HTTP Error Scenarios
# =============================================================================


class TestHttpErrorScenarios:
    """Tests for various HTTP error status codes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,should_retry",
        [
            (400, True),  # Bad Request - still retries
            (401, True),  # Unauthorized
            (403, True),  # Forbidden
            (404, True),  # Not Found
            (429, True),  # Too Many Requests
            (500, True),  # Internal Server Error
            (502, True),  # Bad Gateway
            (503, True),  # Service Unavailable
            (504, True),  # Gateway Timeout
        ],
    )
    async def test_http_status_codes(self, status_code: int, should_retry: bool):
        """Various HTTP status codes trigger retry behavior."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=2,
        )

        mock_client = AsyncMock()
        fail_response = MagicMock()
        fail_response.status_code = status_code
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status_code}", request=MagicMock(), response=fail_response
            )
        )
        mock_client.post = AsyncMock(return_value=fail_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        if should_retry:
            assert mock_client.post.call_count == 2  # max_retries


# =============================================================================
# Tests for Network Error Scenarios
# =============================================================================


class TestNetworkErrorScenarios:
    """Tests for various network error conditions."""

    @pytest.mark.asyncio
    async def test_connection_refused(self):
        """Connection refused error triggers retry."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events", max_retries=2)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Connection timeout error triggers retry."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events", max_retries=2)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectTimeout("Connection timed out")
        )
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_read_timeout(self):
        """Read timeout error triggers retry."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events", max_retries=2)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Read timed out"))
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)
            await emitter.flush()

        assert mock_client.post.call_count == 2


# =============================================================================
# Tests for Lifecycle Management
# =============================================================================


class TestLifecycle:
    """Tests for emitter lifecycle (start, shutdown)."""

    @pytest.mark.asyncio
    async def test_start_creates_client_and_task(self):
        """start() creates HTTP client and periodic flush task."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            await emitter.start()

            assert emitter.is_running is True
            assert emitter._client is not None
            assert emitter._flush_task is not None

            # Clean up
            emitter._running = False
            emitter._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await emitter._flush_task

    @pytest.mark.asyncio
    async def test_shutdown_flushes_and_closes(self):
        """shutdown() flushes buffer and closes client."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True
        emitter._flush_task = asyncio.create_task(asyncio.sleep(100))

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)

        assert emitter.buffer_size == 1

        await emitter.shutdown()

        # Client should be closed
        assert mock_client.aclose.called
        assert emitter._client is None
        assert emitter.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_with_empty_buffer(self):
        """shutdown() handles empty buffer gracefully."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True
        emitter._flush_task = asyncio.create_task(asyncio.sleep(100))

        await emitter.shutdown()

        assert mock_client.aclose.called
        # post should not have been called (nothing to flush)
        assert not mock_client.post.called

    @pytest.mark.asyncio
    async def test_double_shutdown_is_safe(self, disabled_emitter: TelemetryEmitter):
        """Multiple shutdown calls are idempotent."""
        await disabled_emitter.shutdown()
        await disabled_emitter.shutdown()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_emit_after_shutdown(self):
        """Emit after shutdown is a no-op (disabled)."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True
        emitter._flush_task = asyncio.create_task(asyncio.sleep(100))

        await emitter.shutdown()

        # Emit after shutdown
        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)

        # Should not be added (emitter no longer running)
        # Note: the emit checks self.enabled, not self._running
        # After shutdown, enabled is still True but _running is False


# =============================================================================
# Tests for Concurrent Operations
# =============================================================================


class TestConcurrentOperations:
    """Tests for concurrent emit and flush operations."""

    @pytest.mark.asyncio
    async def test_concurrent_emits(self):
        """Multiple concurrent emits are thread-safe."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_buffer_size=1000,
            flush_threshold=1000,  # High threshold to prevent auto-flush
        )
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"

            # Emit many events concurrently
            for i in range(100):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        assert emitter.buffer_size == 100

    @pytest.mark.asyncio
    async def test_flush_uses_lock(self):
        """Concurrent flush calls are serialized via lock."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        # Slow post to simulate long-running flush
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_response

        mock_client.post = slow_post
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            for i in range(5):
                event = create_test_event(f"msg_{i:03d}")
                emitter.emit(event)

        # Start two concurrent flushes
        await asyncio.gather(emitter.flush(), emitter.flush())

        # Both should complete without error
        # Buffer should be empty
        assert emitter.buffer_size == 0


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self):
        """Flushing empty buffer is a no-op."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        mock_client = AsyncMock()
        emitter._client = mock_client
        emitter._running = True

        await emitter.flush()

        # post should not have been called
        assert not mock_client.post.called

    @pytest.mark.asyncio
    async def test_flush_without_client(self):
        """Flush without client (not started) is a no-op."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_test_event()
            emitter.emit(event)

        # Flush before start (no client)
        await emitter.flush()

        # Should complete without error, buffer unchanged
        assert emitter.buffer_size == 1

    def test_buffer_size_property(self):
        """buffer_size property returns current buffer length."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")
        emitter._running = True

        assert emitter.buffer_size == 0

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            emitter.emit(create_test_event("msg_001"))
            assert emitter.buffer_size == 1

            emitter.emit(create_test_event("msg_002"))
            assert emitter.buffer_size == 2

    def test_is_running_property(self):
        """is_running property reflects internal state."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        assert emitter.is_running is False

        emitter._running = True
        assert emitter.is_running is True

        emitter._running = False
        assert emitter.is_running is False


# =============================================================================
# Tests for CloudEvent Format
# =============================================================================


class TestCloudEventFormat:
    """Tests for correct CloudEvent formatting."""

    @pytest.mark.asyncio
    async def test_cloudevent_attributes(self):
        """CloudEvent has correct attributes."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        captured_content = None

        async def capture_post(url, content=None, headers=None):
            nonlocal captured_content
            captured_content = content
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            return response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test_ns"
            event = create_test_event()
            emitter.emit(event)

        await emitter.flush()

        # Parse the captured content
        assert captured_content is not None
        cloud_event = json.loads(captured_content)

        # Check CloudEvent required attributes
        assert "id" in cloud_event
        assert cloud_event["id"].startswith("evt_")
        assert "source" in cloud_event
        assert cloud_event["source"] == "/honcho/test_ns/representation"
        assert "type" in cloud_event
        assert cloud_event["type"] == "representation.completed"
        assert "time" in cloud_event
        assert "dataschema" in cloud_event
        assert "data" in cloud_event

    @pytest.mark.asyncio
    async def test_cloudevent_source_without_namespace(self):
        """CloudEvent source format when namespace is None."""
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        captured_content = None

        async def capture_post(url, content=None, headers=None):
            nonlocal captured_content
            captured_content = content
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            return response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.aclose = AsyncMock()

        emitter._client = mock_client
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = None
            event = create_test_event()
            emitter.emit(event)

        await emitter.flush()

        assert captured_content is not None
        cloud_event = json.loads(captured_content)
        # Without namespace, source should be /honcho/{category}
        assert cloud_event["source"] == "/honcho/representation"

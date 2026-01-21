# pyright: reportPrivateUsage=false
"""Unit tests for emit() function and initialization lifecycle.

Tests for:
- emit() function behavior with/without initialized emitter
- initialize_telemetry_events() configuration handling
- shutdown_telemetry_events() cleanup
- initialize_telemetry() and initialize_telemetry_async() orchestration
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.telemetry.events.representation import RepresentationCompletedEvent


def create_test_event() -> RepresentationCompletedEvent:
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
        latest_message_id="msg_001",
        message_count=1,
        explicit_conclusion_count=1,
        context_preparation_ms=10.0,
        llm_call_ms=100.0,
        total_duration_ms=110.0,
        input_tokens=100,
        output_tokens=50,
    )


# =============================================================================
# Tests for emit() function
# =============================================================================


class TestEmitFunction:
    """Tests for the emit() function in src.telemetry.events."""

    def test_emit_when_emitter_is_none(self):
        """emit() silently returns when emitter is not initialized."""
        from src.telemetry.events import emit

        # Patch where get_emitter is imported from (inside emit function)
        with patch("src.telemetry.emitter.get_emitter", return_value=None):
            event = create_test_event()
            # Should not raise, just silently return
            emit(event)

    def test_emit_calls_emitter_emit(self):
        """emit() calls the emitter's emit method when initialized."""
        from src.telemetry.events import emit

        mock_emitter = MagicMock()

        with patch("src.telemetry.emitter.get_emitter", return_value=mock_emitter):
            event = create_test_event()
            emit(event)

        mock_emitter.emit.assert_called_once_with(event)

    def test_emit_passes_event_unchanged(self):
        """emit() passes the event object unchanged to emitter."""
        from src.telemetry.events import emit

        mock_emitter = MagicMock()

        with patch("src.telemetry.emitter.get_emitter", return_value=mock_emitter):
            event = create_test_event()
            emit(event)

            # Get the event that was passed to emit
            passed_event = mock_emitter.emit.call_args[0][0]
            assert passed_event is event


# =============================================================================
# Tests for get_emitter() function
# =============================================================================


class TestGetEmitter:
    """Tests for the get_emitter() function."""

    def test_get_emitter_returns_none_when_not_initialized(self):
        """get_emitter() returns None before initialization."""
        from src.telemetry import emitter as emitter_module

        # Save original value
        original = emitter_module._emitter

        try:
            emitter_module._emitter = None
            result = emitter_module.get_emitter()
            assert result is None
        finally:
            # Restore original
            emitter_module._emitter = original

    def test_get_emitter_returns_emitter_when_initialized(self):
        """get_emitter() returns the emitter after initialization."""
        from src.telemetry import emitter as emitter_module
        from src.telemetry.emitter import TelemetryEmitter

        # Save original value
        original = emitter_module._emitter

        try:
            test_emitter = TelemetryEmitter(endpoint="http://test:8001/events")
            emitter_module._emitter = test_emitter

            result = emitter_module.get_emitter()
            assert result is test_emitter
        finally:
            # Restore original
            emitter_module._emitter = original


# =============================================================================
# Tests for initialize_emitter() function
# =============================================================================


class TestInitializeEmitter:
    """Tests for the initialize_emitter() function."""

    @pytest.mark.asyncio
    async def test_initialize_emitter_creates_and_starts(self):
        """initialize_emitter() creates emitter and starts it."""
        from src.telemetry import emitter as emitter_module
        from src.telemetry.emitter import initialize_emitter

        # Save original value
        original = emitter_module._emitter

        # Create a properly async mock client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        try:
            emitter_module._emitter = None

            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await initialize_emitter(
                    endpoint="http://test:8001/events",
                    enabled=True,
                )

            assert result is not None
            assert result.endpoint == "http://test:8001/events"
            assert result.is_running is True

            # Should be stored in global
            assert emitter_module._emitter is result

            # Clean up
            await result.shutdown()
        finally:
            emitter_module._emitter = original

    @pytest.mark.asyncio
    async def test_initialize_emitter_with_custom_settings(self):
        """initialize_emitter() accepts custom configuration."""
        from src.telemetry import emitter as emitter_module
        from src.telemetry.emitter import initialize_emitter

        original = emitter_module._emitter

        # Create a properly async mock client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        try:
            emitter_module._emitter = None

            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await initialize_emitter(
                    endpoint="http://custom:9000/events",
                    headers={"X-API-Key": "secret"},
                    batch_size=50,
                    flush_interval_seconds=2.0,
                    flush_threshold=25,
                    max_retries=5,
                    max_buffer_size=5000,
                    enabled=True,
                )

            assert result.endpoint == "http://custom:9000/events"
            assert result.headers == {"X-API-Key": "secret"}
            assert result.batch_size == 50
            assert result.flush_interval == 2.0
            assert result.flush_threshold == 25
            assert result.max_retries == 5
            assert result.max_buffer_size == 5000

            await result.shutdown()
        finally:
            emitter_module._emitter = original

    @pytest.mark.asyncio
    async def test_initialize_emitter_disabled(self):
        """initialize_emitter() creates disabled emitter when enabled=False."""
        from src.telemetry import emitter as emitter_module
        from src.telemetry.emitter import initialize_emitter

        original = emitter_module._emitter

        try:
            emitter_module._emitter = None

            result = await initialize_emitter(
                endpoint="http://test:8001/events",
                enabled=False,
            )

            assert result is not None
            assert result.enabled is False
            assert result.is_running is False

            await result.shutdown()
        finally:
            emitter_module._emitter = original


# =============================================================================
# Tests for shutdown_emitter() function
# =============================================================================


class TestShutdownEmitter:
    """Tests for the shutdown_emitter() function."""

    @pytest.mark.asyncio
    async def test_shutdown_emitter_when_none(self):
        """shutdown_emitter() handles None emitter gracefully."""
        from src.telemetry import emitter as emitter_module
        from src.telemetry.emitter import shutdown_emitter

        original = emitter_module._emitter

        try:
            emitter_module._emitter = None
            # Should not raise
            await shutdown_emitter()
        finally:
            emitter_module._emitter = original

    @pytest.mark.asyncio
    async def test_shutdown_emitter_calls_shutdown(self):
        """shutdown_emitter() calls shutdown on the emitter."""
        from src.telemetry import emitter as emitter_module
        from src.telemetry.emitter import shutdown_emitter

        original = emitter_module._emitter

        try:
            mock_emitter = AsyncMock()
            emitter_module._emitter = mock_emitter

            await shutdown_emitter()

            mock_emitter.shutdown.assert_called_once()
            # Should clear global
            assert emitter_module._emitter is None
        finally:
            emitter_module._emitter = original


# =============================================================================
# Tests for initialize_telemetry_events() function
# =============================================================================


class TestInitializeTelemetryEvents:
    """Tests for initialize_telemetry_events() in src.telemetry.events."""

    @pytest.mark.asyncio
    async def test_initialize_when_disabled(self):
        """initialize_telemetry_events() is no-op when TELEMETRY.ENABLED=False."""
        from src.telemetry.events import initialize_telemetry_events

        with (
            patch("src.config.settings") as mock_settings,
            patch("src.telemetry.emitter.initialize_emitter") as mock_init,
        ):
            mock_settings.TELEMETRY.ENABLED = False

            await initialize_telemetry_events()

            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_when_enabled(self):
        """initialize_telemetry_events() initializes emitter when enabled."""
        from src.telemetry.events import initialize_telemetry_events

        with (
            patch("src.config.settings") as mock_settings,
            patch("src.telemetry.emitter.initialize_emitter") as mock_init,
        ):
            mock_settings.TELEMETRY.ENABLED = True
            mock_settings.TELEMETRY.ENDPOINT = "http://test:8001/events"
            mock_settings.TELEMETRY.HEADERS = {"X-API-Key": "secret"}
            mock_settings.TELEMETRY.BATCH_SIZE = 100
            mock_settings.TELEMETRY.FLUSH_INTERVAL_SECONDS = 1.0
            mock_settings.TELEMETRY.FLUSH_THRESHOLD = 50
            mock_settings.TELEMETRY.MAX_RETRIES = 3
            mock_settings.TELEMETRY.MAX_BUFFER_SIZE = 10000
            mock_init.return_value = AsyncMock()

            await initialize_telemetry_events()

            mock_init.assert_called_once_with(
                endpoint="http://test:8001/events",
                headers={"X-API-Key": "secret"},
                batch_size=100,
                flush_interval_seconds=1.0,
                flush_threshold=50,
                max_retries=3,
                max_buffer_size=10000,
                enabled=True,
            )


# =============================================================================
# Tests for shutdown_telemetry_events() function
# =============================================================================


class TestShutdownTelemetryEvents:
    """Tests for shutdown_telemetry_events() in src.telemetry.events."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_shutdown_emitter(self):
        """shutdown_telemetry_events() calls shutdown_emitter."""
        from src.telemetry.events import shutdown_telemetry_events

        with patch(
            "src.telemetry.emitter.shutdown_emitter", new_callable=AsyncMock
        ) as mock_shutdown:
            await shutdown_telemetry_events()

            mock_shutdown.assert_called_once()


# =============================================================================
# Tests for initialize_telemetry() function
# =============================================================================


class TestInitializeTelemetry:
    """Tests for initialize_telemetry() in src.telemetry."""

    def test_initialize_otel_when_enabled(self):
        """initialize_telemetry() initializes OTel metrics when enabled."""
        import src.telemetry as telemetry_module

        with (
            patch("src.config.settings") as mock_settings,
            # Patch where it's used (in src.telemetry), not where it's defined
            patch.object(telemetry_module, "initialize_otel_metrics") as mock_otel_init,
        ):
            mock_settings.OTEL.ENABLED = True
            mock_settings.OTEL.ENDPOINT = "http://otel:9009/metrics"
            mock_settings.OTEL.HEADERS = None
            mock_settings.OTEL.EXPORT_INTERVAL_MILLIS = 60000
            mock_settings.OTEL.SERVICE_NAME = "honcho"
            mock_settings.OTEL.SERVICE_NAMESPACE = "test"

            telemetry_module.initialize_telemetry()

            mock_otel_init.assert_called_once_with(
                endpoint="http://otel:9009/metrics",
                headers=None,
                export_interval_millis=60000,
                service_name="honcho",
                service_namespace="test",
                enabled=True,
            )

    def test_skip_otel_when_disabled(self):
        """initialize_telemetry() skips OTel when disabled."""
        import src.telemetry as telemetry_module

        with (
            patch("src.config.settings") as mock_settings,
            patch.object(telemetry_module, "initialize_otel_metrics") as mock_otel_init,
        ):
            mock_settings.OTEL.ENABLED = False

            telemetry_module.initialize_telemetry()

            mock_otel_init.assert_not_called()


# =============================================================================
# Tests for initialize_telemetry_async() function
# =============================================================================


class TestInitializeTelemetryAsync:
    """Tests for initialize_telemetry_async() in src.telemetry."""

    @pytest.mark.asyncio
    async def test_initialize_cloudevents_when_enabled(self):
        """initialize_telemetry_async() initializes CloudEvents when enabled."""
        from src.telemetry import initialize_telemetry_async

        with (
            patch("src.config.settings") as mock_settings,
            patch(
                "src.telemetry.events.initialize_telemetry_events",
                new_callable=AsyncMock,
            ) as mock_ce_init,
        ):
            mock_settings.TELEMETRY.ENABLED = True

            await initialize_telemetry_async()

            mock_ce_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_cloudevents_when_disabled(self):
        """initialize_telemetry_async() skips CloudEvents when disabled."""
        from src.telemetry import initialize_telemetry_async

        with (
            patch("src.config.settings") as mock_settings,
            patch(
                "src.telemetry.events.initialize_telemetry_events",
                new_callable=AsyncMock,
            ) as mock_ce_init,
        ):
            mock_settings.TELEMETRY.ENABLED = False

            await initialize_telemetry_async()

            mock_ce_init.assert_not_called()


# =============================================================================
# Tests for shutdown_telemetry() function
# =============================================================================


class TestShutdownTelemetry:
    """Tests for shutdown_telemetry() in src.telemetry."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_all_subsystems(self):
        """shutdown_telemetry() shuts down all telemetry subsystems."""
        from src.telemetry import shutdown_telemetry

        with (
            patch(
                "src.telemetry.events.shutdown_telemetry_events", new_callable=AsyncMock
            ) as mock_ce_shutdown,
            patch("src.telemetry.shutdown_otel_metrics") as mock_otel_shutdown,
        ):
            await shutdown_telemetry()

            mock_ce_shutdown.assert_called_once()
            mock_otel_shutdown.assert_called_once()

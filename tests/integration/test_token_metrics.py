"""Integration tests for OpenTelemetry token metrics tracking.

These tests verify that deriver and dialectic token metrics are correctly
emitted with accurate token counts when processing messages and dialectic queries.

The approach uses delta-based verification with OTel's InMemoryMetricReader:
1. Capture counter values before test execution
2. Run the code under test (with mocked LLM)
3. Verify deltas match expected values
"""

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.models import Peer, Workspace
from src.schemas import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.telemetry.otel.metrics import otel_metrics
from src.utils.clients import HonchoLLMCallResponse
from src.utils.representation import ExplicitObservationBase, PromptRepresentation
from src.utils.summarizer import (
    SummaryType,
    _create_and_save_summary,  # pyright: ignore[reportPrivateUsage]
    estimate_short_summary_prompt_tokens,
)

# =============================================================================
# Fixtures
# =============================================================================


class OTelMetricChecker:
    """Utility class to capture and verify OTel counter deltas."""

    _reader: InMemoryMetricReader

    def __init__(self, reader: InMemoryMetricReader):
        self._reader = reader

    def _get_metric_value(self, metric_name: str, labels: dict[str, str]) -> float:
        """Get current value of a counter with specific labels.

        Note: The OTel SDK's MetricsData types are not fully typed, so we cast
        to Any to avoid type warnings when traversing the metrics data structure.
        """
        raw_data = self._reader.get_metrics_data()  # pyright: ignore[reportUnknownVariableType]
        if raw_data is None:
            return 0.0
        data = cast(Any, raw_data)

        for resource_metrics in data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == metric_name and hasattr(
                        metric.data, "data_points"
                    ):
                        for point in metric.data.data_points:
                            # Check if labels match
                            point_attrs: dict[str, str] = (
                                dict(point.attributes) if point.attributes else {}
                            )
                            if all(point_attrs.get(k) == v for k, v in labels.items()):
                                return float(point.value)
        return 0.0

    def capture(self, metric_name: str, labels: dict[str, str]) -> float:
        """Capture current value of a counter with specific labels."""
        return self._get_metric_value(metric_name, labels)

    def get_delta(
        self, metric_name: str, labels: dict[str, str], before: float
    ) -> float:
        """Get the delta between a before value and current."""
        return self.capture(metric_name, labels) - before

    def assert_delta(
        self,
        metric_name: str,
        labels: dict[str, str],
        before: float,
        expected: int | float,
        message: str = "",
    ) -> None:
        """Assert that the delta matches expected value."""
        delta = self.get_delta(metric_name, labels, before)
        assert (
            delta == expected
        ), f"{message}: expected delta {expected}, got {delta}. Labels: {labels}"


def _reset_otel_metrics_singleton() -> None:
    """Reset the otel_metrics singleton instance so it reinitializes with a new provider.

    This clears all instance attributes so _ensure_initialized() will create
    new meters tied to the current global MeterProvider.
    """
    # Reset initialization flag (instance attribute shadows class attribute)
    otel_metrics._is_initialized = False  # pyright: ignore[reportPrivateUsage]

    # Reset all meters
    otel_metrics._api_meter = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._deriver_meter = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._dialectic_meter = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._dreamer_meter = None  # pyright: ignore[reportPrivateUsage]

    # Reset all counters
    otel_metrics._api_requests = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._messages_created = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._dialectic_calls = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._deriver_queue_items = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._deriver_tokens = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._dialectic_tokens = None  # pyright: ignore[reportPrivateUsage]
    otel_metrics._dreamer_tokens = None  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def otel_test_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[InMemoryMetricReader, OTelMetricChecker]]:
    """Set up OTel metrics with in-memory reader for testing.

    The OTel SDK only allows set_meter_provider() to be called once per process.
    To work around this for testing, we patch get_meter() to return meters
    from our test provider directly.

    Yields:
        Tuple of (reader, checker) for verifying metrics
    """
    # Create in-memory reader
    reader = InMemoryMetricReader()

    # Create a test meter provider
    provider = MeterProvider(metric_readers=[reader])

    # Patch get_meter to return meters from our test provider
    # This is necessary because set_meter_provider() can only be called once per process
    def test_get_meter(name: str, version: str = "") -> Meter:
        return provider.get_meter(name, version)

    monkeypatch.setattr("src.telemetry.otel.metrics.get_meter", test_get_meter)

    # Reset the otel_metrics singleton instance so it reinitializes with our test provider
    _reset_otel_metrics_singleton()

    # Enable OTEL in settings
    monkeypatch.setattr("src.config.settings.OTEL.ENABLED", True)
    monkeypatch.setattr("src.config.settings.METRICS.NAMESPACE", "test")

    checker = OTelMetricChecker(reader)

    yield reader, checker

    # Cleanup: reset singleton so other tests aren't affected
    _reset_otel_metrics_singleton()


@pytest.fixture
def metric_checker(
    otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
) -> OTelMetricChecker:
    """Fixture providing a metric checker instance."""
    return otel_test_setup[1]


# =============================================================================
# Test Data Helpers
# =============================================================================


async def create_test_session_with_peer(
    db_session: AsyncSession,
    workspace: Workspace,
    peer: Peer,
) -> models.Session:
    """Create a session with a peer configured for observation."""
    session = (
        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={peer.name: schemas.SessionPeerConfig(observe_me=True)},
            ),
            workspace.name,
        )
    ).resource
    await db_session.commit()
    return session


async def create_test_messages(
    db_session: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str,
    count: int = 1,
    content_prefix: str = "Test message",
) -> list[models.Message]:
    """Create test messages in the database."""
    messages: list[models.Message] = []
    for i in range(count):
        message = models.Message(
            workspace_name=workspace_name,
            session_name=session_name,
            peer_name=peer_name,
            content=f"{content_prefix} {i}",
            public_id=generate_nanoid(),
            seq_in_session=i + 1,
            token_count=10,
        )
        db_session.add(message)
        messages.append(message)

    await db_session.commit()
    # Refresh to get IDs
    for msg in messages:
        await db_session.refresh(msg)
    return messages


def create_test_configuration() -> ResolvedConfiguration:
    """Create a test configuration to avoid DB lookups in tests."""
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
        summary=ResolvedSummaryConfiguration(
            enabled=True, messages_per_short_summary=20, messages_per_long_summary=60
        ),
        dream=ResolvedDreamConfiguration(enabled=False),
    )


def create_mock_deriver_response(
    output_tokens: int = 42,
) -> HonchoLLMCallResponse[PromptRepresentation]:
    """Create a mock LLM response for the deriver."""
    return HonchoLLMCallResponse(
        content=PromptRepresentation(
            explicit=[ExplicitObservationBase(content="Test observation from deriver")],
        ),
        input_tokens=100,
        output_tokens=output_tokens,
        finish_reasons=["end_turn"],
    )


def create_mock_dialectic_response(
    input_tokens: int = 150, output_tokens: int = 75
) -> HonchoLLMCallResponse[str]:
    """Create a mock LLM response for the dialectic."""
    return HonchoLLMCallResponse(
        content="This is a test dialectic response about the peer.",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        finish_reasons=["end_turn"],
        tool_calls_made=[],
    )


# =============================================================================
# Deriver Ingestion Metrics Tests
# =============================================================================


@pytest.mark.asyncio
class TestDeriverIngestionMetrics:
    """Test token metrics for deriver INGESTION task type."""

    async def test_ingestion_tracks_output_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify OUTPUT_TOTAL tokens match response.output_tokens from LLM."""
        from src.deriver.deriver import process_representation_tasks_batch

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session, workspace.name, session.name, peer.name, count=1
        )

        expected_output_tokens = 42
        mock_response = create_mock_deriver_response(
            output_tokens=expected_output_tokens
        )

        # Capture metrics before
        labels = {
            "namespace": "test",
            "task_type": "ingestion",
            "token_type": "output",
            "component": "output_total",
        }
        before = metric_checker.capture("deriver_tokens_processed", labels)

        # Mock the LLM call and save_representation (we're testing metrics, not DB writes)
        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(return_value=mock_response),
            ),
            patch(
                "src.crud.representation.RepresentationManager.save_representation",
                new=AsyncMock(),
            ),
        ):
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=create_test_configuration(),
                observer=peer.name,
                observed=peer.name,
            )

        # Verify output tokens metric
        metric_checker.assert_delta(
            "deriver_tokens_processed",
            labels,
            before,
            expected_output_tokens,
            "Ingestion output tokens",
        )

    async def test_ingestion_tracks_prompt_input_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify PROMPT component is tracked for ingestion input."""
        from src.deriver.deriver import process_representation_tasks_batch
        from src.deriver.prompts import estimate_minimal_deriver_prompt_tokens

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session, workspace.name, session.name, peer.name, count=1
        )

        mock_response = create_mock_deriver_response()

        # Get expected prompt tokens
        expected_prompt_tokens = estimate_minimal_deriver_prompt_tokens()

        labels = {
            "namespace": "test",
            "task_type": "ingestion",
            "token_type": "input",
            "component": "prompt",
        }
        before = metric_checker.capture("deriver_tokens_processed", labels)

        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(return_value=mock_response),
            ),
            patch(
                "src.crud.representation.RepresentationManager.save_representation",
                new=AsyncMock(),
            ),
        ):
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=create_test_configuration(),
                observer=peer.name,
                observed=peer.name,
            )

        metric_checker.assert_delta(
            "deriver_tokens_processed",
            labels,
            before,
            expected_prompt_tokens,
            "Ingestion prompt input tokens",
        )

    async def test_ingestion_tracks_messages_input_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify MESSAGES component is tracked for ingestion input."""
        from src.deriver.deriver import process_representation_tasks_batch

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session,
            workspace.name,
            session.name,
            peer.name,
            count=1,
            content_prefix="Hello this is a test message",
        )

        mock_response = create_mock_deriver_response()

        labels = {
            "namespace": "test",
            "task_type": "ingestion",
            "token_type": "input",
            "component": "messages",
        }
        before = metric_checker.capture("deriver_tokens_processed", labels)

        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(return_value=mock_response),
            ),
            patch(
                "src.crud.representation.RepresentationManager.save_representation",
                new=AsyncMock(),
            ),
        ):
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=create_test_configuration(),
                observer=peer.name,
                observed=peer.name,
            )

        # Verify messages tokens were tracked (should be > 0)
        delta = metric_checker.get_delta("deriver_tokens_processed", labels, before)
        assert delta > 0, f"Expected messages input tokens > 0, got {delta}"


# =============================================================================
# Deriver Summary Metrics Tests
# =============================================================================


@pytest.mark.asyncio
class TestDeriverSummaryMetrics:
    """Test token metrics for deriver SUMMARY task type."""

    async def test_summary_tracks_output_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify OUTPUT_TOTAL tokens are tracked for summary."""

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)

        # Create messages for summary
        messages = await create_test_messages(
            db_session, workspace.name, session.name, peer.name, count=5
        )
        last_message = messages[-1]

        # Mock _create_summary to return a summary with known token count
        expected_output_tokens = 25
        mock_summary = {
            "content": "This is a test summary.",
            "token_count": expected_output_tokens,
            "message_id": last_message.id,
            "message_public_id": last_message.public_id,
        }

        labels = {
            "namespace": "test",
            "task_type": "summary",
            "token_type": "output",
            "component": "output_total",
        }
        before = metric_checker.capture("deriver_tokens_processed", labels)

        with (
            patch(
                "src.utils.summarizer._create_summary",
                new=AsyncMock(
                    return_value=(mock_summary, False, 100, expected_output_tokens)
                ),  # (summary, is_fallback, llm_input_tokens, llm_output_tokens)
            ),
            patch(
                "src.utils.summarizer._save_summary",
                new=AsyncMock(),
            ),
        ):
            await _create_and_save_summary(
                db=db_session,
                workspace_name=workspace.name,
                session_name=session.name,
                message_id=last_message.id,
                message_seq_in_session=last_message.seq_in_session,
                summary_type=SummaryType.SHORT,
                message_public_id=last_message.public_id,
                configuration=create_test_configuration(),
            )

        # Verify output tokens match the summary token_count
        metric_checker.assert_delta(
            "deriver_tokens_processed",
            labels,
            before,
            expected_output_tokens,
            "Summary output tokens",
        )

    async def test_summary_tracks_prompt_input_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify PROMPT component is tracked for summary input."""

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session, workspace.name, session.name, peer.name, count=5
        )
        last_message = messages[-1]

        expected_prompt_tokens = estimate_short_summary_prompt_tokens()

        mock_summary = {
            "content": "Test summary.",
            "token_count": 10,
            "message_id": last_message.id,
            "message_public_id": last_message.public_id,
        }

        labels = {
            "namespace": "test",
            "task_type": "summary",
            "token_type": "input",
            "component": "prompt",
        }
        before = metric_checker.capture("deriver_tokens_processed", labels)

        with (
            patch(
                "src.utils.summarizer._create_summary",
                new=AsyncMock(
                    return_value=(mock_summary, False, 100, 10)
                ),  # (summary, is_fallback, llm_input_tokens, llm_output_tokens)
            ),
            patch(
                "src.utils.summarizer._save_summary",
                new=AsyncMock(),
            ),
        ):
            await _create_and_save_summary(
                db=db_session,
                workspace_name=workspace.name,
                session_name=session.name,
                message_id=last_message.id,
                message_seq_in_session=last_message.seq_in_session,
                summary_type=SummaryType.SHORT,
                message_public_id=last_message.public_id,
                configuration=create_test_configuration(),
            )

        metric_checker.assert_delta(
            "deriver_tokens_processed",
            labels,
            before,
            expected_prompt_tokens,
            "Summary prompt input tokens",
        )

    async def test_summary_tracks_messages_input_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify MESSAGES component is tracked for summary input."""

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session, workspace.name, session.name, peer.name, count=5
        )
        last_message = messages[-1]

        # Get the actual messages that would be included in the summary
        # (to match what _create_and_save_summary computes via get_messages_by_seq_range)
        actual_messages = await crud.get_messages_by_seq_range(
            db_session,
            workspace.name,
            session.name,
            start_seq=1,
            end_seq=last_message.seq_in_session,
        )
        expected_messages_tokens = sum(m.token_count for m in actual_messages)

        mock_summary = {
            "content": "Test summary.",
            "token_count": 10,
            "message_id": last_message.id,
            "message_public_id": last_message.public_id,
        }

        labels = {
            "namespace": "test",
            "task_type": "summary",
            "token_type": "input",
            "component": "messages",
        }
        before = metric_checker.capture("deriver_tokens_processed", labels)

        with (
            patch(
                "src.utils.summarizer._create_summary",
                new=AsyncMock(
                    return_value=(mock_summary, False, 100, 10)
                ),  # (summary, is_fallback, llm_input_tokens, llm_output_tokens)
            ),
            patch(
                "src.utils.summarizer._save_summary",
                new=AsyncMock(),
            ),
        ):
            await _create_and_save_summary(
                db=db_session,
                workspace_name=workspace.name,
                session_name=session.name,
                message_id=last_message.id,
                message_seq_in_session=last_message.seq_in_session,
                summary_type=SummaryType.SHORT,
                message_public_id=last_message.public_id,
                configuration=create_test_configuration(),
            )

        # Verify messages tokens match what summarizer actually computed
        delta = metric_checker.get_delta("deriver_tokens_processed", labels, before)
        assert (
            delta == expected_messages_tokens
        ), f"Expected messages input tokens {expected_messages_tokens}, got {delta}"
        assert delta > 0, "Expected at least some message tokens to be tracked"

    async def test_summary_fallback_does_not_track(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify metrics are NOT emitted when _create_summary returns is_fallback=True."""

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)
        messages = await create_test_messages(
            db_session, workspace.name, session.name, peer.name, count=5
        )
        last_message = messages[-1]

        mock_summary = {
            "content": "Fallback summary.",
            "token_count": 10,
            "message_id": last_message.id,
            "message_public_id": last_message.public_id,
        }

        # Capture all relevant metric labels before
        output_labels = {
            "namespace": "test",
            "task_type": "summary",
            "token_type": "output",
            "component": "output_total",
        }
        prompt_labels = {
            "namespace": "test",
            "task_type": "summary",
            "token_type": "input",
            "component": "prompt",
        }
        before_output = metric_checker.capture(
            "deriver_tokens_processed", output_labels
        )
        before_prompt = metric_checker.capture(
            "deriver_tokens_processed", prompt_labels
        )

        with patch(
            "src.utils.summarizer._create_summary",
            new=AsyncMock(
                return_value=(mock_summary, True, 0, 0)
            ),  # (summary, is_fallback=True, llm_input_tokens, llm_output_tokens)
        ):
            await _create_and_save_summary(
                db=db_session,
                workspace_name=workspace.name,
                session_name=session.name,
                message_id=last_message.id,
                message_seq_in_session=last_message.seq_in_session,
                summary_type=SummaryType.SHORT,
                message_public_id=last_message.public_id,
                configuration=create_test_configuration(),
            )

        # Verify NO change in metrics when fallback
        output_delta = metric_checker.get_delta(
            "deriver_tokens_processed", output_labels, before_output
        )
        prompt_delta = metric_checker.get_delta(
            "deriver_tokens_processed", prompt_labels, before_prompt
        )

        assert (
            output_delta == 0
        ), f"Expected no output token change on fallback, got {output_delta}"
        assert (
            prompt_delta == 0
        ), f"Expected no prompt token change on fallback, got {prompt_delta}"


# =============================================================================
# Dialectic Token Metrics Tests
# =============================================================================


@pytest.mark.asyncio
class TestDialecticTokenMetrics:
    """Test token metrics for dialectic calls."""

    async def test_dialectic_tracks_input_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify INPUT tokens are tracked from LLM response."""
        from src.dialectic.core import DialecticAgent

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)

        expected_input_tokens = 150
        mock_response = create_mock_dialectic_response(
            input_tokens=expected_input_tokens, output_tokens=75
        )

        labels = {
            "namespace": "test",
            "token_type": "input",
            "component": "total",
            "reasoning_level": "low",
        }
        before = metric_checker.capture("dialectic_tokens_processed", labels)

        agent = DialecticAgent(
            db=db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            observer=peer.name,
            observed=peer.name,
        )

        with patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ):
            await agent.answer("What do you know about this user?")

        metric_checker.assert_delta(
            "dialectic_tokens_processed",
            labels,
            before,
            expected_input_tokens,
            "Dialectic input tokens",
        )

    async def test_dialectic_tracks_output_tokens(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
    ):
        """Verify OUTPUT tokens are tracked from LLM response."""
        from src.dialectic.core import DialecticAgent

        _, metric_checker = otel_test_setup
        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)

        expected_output_tokens = 75
        mock_response = create_mock_dialectic_response(
            input_tokens=150, output_tokens=expected_output_tokens
        )

        labels = {
            "namespace": "test",
            "token_type": "output",
            "component": "total",
            "reasoning_level": "low",
        }
        before = metric_checker.capture("dialectic_tokens_processed", labels)

        agent = DialecticAgent(
            db=db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            observer=peer.name,
            observed=peer.name,
        )

        with patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ):
            await agent.answer("What do you know about this user?")

        metric_checker.assert_delta(
            "dialectic_tokens_processed",
            labels,
            before,
            expected_output_tokens,
            "Dialectic output tokens",
        )

    async def test_dialectic_metrics_disabled_no_emission(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        otel_test_setup: tuple[InMemoryMetricReader, OTelMetricChecker],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Verify metrics are NOT emitted when OTEL.ENABLED=False."""
        from src.dialectic.core import DialecticAgent

        _, metric_checker = otel_test_setup

        # Explicitly disable OTEL metrics
        monkeypatch.setattr("src.config.settings.OTEL.ENABLED", False)

        workspace, peer = sample_data
        session = await create_test_session_with_peer(db_session, workspace, peer)

        mock_response = create_mock_dialectic_response(
            input_tokens=200, output_tokens=100
        )

        # Capture before values
        input_labels = {
            "namespace": "test",
            "token_type": "input",
            "component": "total",
            "reasoning_level": "low",
        }
        output_labels = {
            "namespace": "test",
            "token_type": "output",
            "component": "total",
            "reasoning_level": "low",
        }
        before_input = metric_checker.capture(
            "dialectic_tokens_processed", input_labels
        )
        before_output = metric_checker.capture(
            "dialectic_tokens_processed", output_labels
        )

        agent = DialecticAgent(
            db=db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            observer=peer.name,
            observed=peer.name,
        )

        with patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ):
            await agent.answer("What do you know about this user?")

        # Verify NO change in metrics
        input_delta = metric_checker.get_delta(
            "dialectic_tokens_processed", input_labels, before_input
        )
        output_delta = metric_checker.get_delta(
            "dialectic_tokens_processed", output_labels, before_output
        )

        assert (
            input_delta == 0
        ), f"Expected no input token change when disabled, got {input_delta}"
        assert (
            output_delta == 0
        ), f"Expected no output token change when disabled, got {output_delta}"

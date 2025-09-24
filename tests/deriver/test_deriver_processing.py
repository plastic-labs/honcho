import signal
from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src import models
from src.deriver.deriver import process_representation_tasks_batch
from src.deriver.queue_payload import RepresentationPayload
from src.utils.representation import Representation


@pytest.mark.asyncio
class TestDeriverProcessing:
    """Test suite for deriver processing using the conftest fixtures"""

    async def test_mock_critical_analysis_call(
        self,
        mock_critical_analysis_call: Generator[Callable[..., Any], None, None],
        sample_messages: list[models.Message],
    ):
        """Test that the critical analysis call is properly mocked"""
        assert mock_critical_analysis_call is not None
        assert len(sample_messages) > 0  # Verify we have messages for testing

        # The mock should be in place and return a predefined response
        # This ensures no actual LLM calls are made during testing

    async def test_work_unit_key_generation(
        self,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that work unit keys are generated correctly"""
        from src.deriver.utils import get_work_unit_key

        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create a payload for representation task
        representation_payload = {
            "workspace_name": "workspace1",
            "session_name": session.name,
            "sender_name": peer1.name,
            "target_name": peer2.name,
            "task_type": "representation",
        }

        # Generate work unit key for representation
        work_unit_key = get_work_unit_key("representation", representation_payload)
        expected_key = (
            f"representation:workspace1:{session.name}:{peer1.name}:{peer2.name}"
        )
        assert work_unit_key == expected_key

        # Create a payload for summary task (sender_name and target_name should be None)
        summary_payload = {
            "workspace_name": "workspace1",
            "session_name": session.name,
            "task_type": "summary",
        }

        # Generate work unit key for summary
        summary_work_unit_key = get_work_unit_key("summary", summary_payload)
        expected_summary_key = f"summary:workspace1:{session.name}:None:None"
        assert summary_work_unit_key == expected_summary_key

    async def test_mock_queue_manager(
        self,
        mock_queue_manager: Any,  # AsyncMock object
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that the queue manager is properly mocked"""
        session, peers = sample_session_with_peers
        assert session is not None
        assert len(peers) == 3

        # Verify the mock has the expected attributes
        assert mock_queue_manager is not None
        assert hasattr(mock_queue_manager, "initialize")
        assert hasattr(mock_queue_manager, "shutdown")
        assert hasattr(mock_queue_manager, "process_work_unit")

        # Verify we can call the mocked methods
        await mock_queue_manager.initialize()
        await mock_queue_manager.shutdown(signal.SIGTERM)

        # Verify the mocked methods were called
        mock_queue_manager.initialize.assert_called_once()  # type: ignore[attr-defined]
        mock_queue_manager.shutdown.assert_called_once()  # type: ignore[attr-defined]

    async def test_mock_embedding_store(
        self,
        mock_embedding_store: Any,  # AsyncMock object
    ):
        """Test that the embedding store is properly mocked"""
        assert mock_embedding_store is not None

        # Verify we can call the mocked methods
        await mock_embedding_store.save_representation([])
        mock_embedding_store.get_relevant_observations.return_value = []  # type: ignore[attr-defined]

        # Verify the methods were called
        assert mock_embedding_store.save_representation.called  # type: ignore[attr-defined]

    async def test_representation_batch_uses_earliest_cutoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure batching history cutoff uses the earliest payload in the batch."""
        captured_cutoffs: list[int] = []

        async def fake_get_session_context_formatted(*_args: Any, **kwargs: Any) -> str:
            captured_cutoffs.append(kwargs["cutoff"])
            return "formatted-history"

        # Mock only the function we need to inspect for the test assertion
        monkeypatch.setattr(
            "src.deriver.deriver.summarizer.get_session_context_formatted",
            fake_get_session_context_formatted,
        )

        # Provide a stub working representation so embedding lookups are skipped.
        monkeypatch.setattr(
            "src.crud.get_working_representation",
            AsyncMock(
                return_value=Representation(
                    explicit=[],
                    deductive=[],
                )
            ),
        )

        # Avoid DB access for collection and peer card
        monkeypatch.setattr(
            "src.crud.get_or_create_collection",
            AsyncMock(return_value=type("Collection", (), {"name": "dummy"})()),
        )
        monkeypatch.setattr(
            "src.crud.get_peer_card",
            AsyncMock(return_value=[]),
        )
        # Short-circuit tracked_db context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _no_db(_label: str):
            yield object()

        monkeypatch.setattr("src.deriver.deriver.tracked_db", _no_db)

        # Avoid executing the full reasoning pipeline; we only care about cutoff behavior.
        monkeypatch.setattr(
            "src.deriver.deriver.CertaintyReasoner.reason",
            AsyncMock(return_value=Representation(explicit=[], deductive=[])),
        )

        # Create test payloads with different message IDs (earlier message has lower ID)
        now = datetime.now(timezone.utc)
        payloads: list[RepresentationPayload] = []
        for i in range(8):
            message_id = 100 + i  # 100, 101, 102, ..., 107
            payloads.append(
                RepresentationPayload(
                    workspace_name="test_workspace",
                    session_name="test_session",
                    message_id=message_id,
                    content=f"message {message_id}",
                    sender_name="alice",
                    target_name="alice",
                    created_at=now
                    - timedelta(
                        minutes=7 - i
                    ),  # Earlier messages have earlier timestamps
                )
            )

        await process_representation_tasks_batch(payloads)

        # Verify that the earliest message ID was used as the cutoff
        assert captured_cutoffs == [payloads[0].message_id]

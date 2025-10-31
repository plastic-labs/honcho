import signal
from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.deriver import process_representation_tasks_batch
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
        from src.utils.work_unit import get_work_unit_key

        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create a payload for representation task
        representation_payload = {
            "workspace_name": "workspace1",
            "session_name": session.name,
            "observer": peer2.name,
            "observed": peer1.name,
            "task_type": "representation",
        }

        # Generate work unit key for representation
        work_unit_key = get_work_unit_key(representation_payload)
        expected_key = (
            f"representation:workspace1:{session.name}:{peer2.name}:{peer1.name}"
        )
        assert work_unit_key == expected_key

        # Create a payload for summary task
        summary_payload = {
            "workspace_name": "workspace1",
            "session_name": session.name,
            "task_type": "summary",
        }

        # Generate work unit key for summary
        summary_work_unit_key = get_work_unit_key(summary_payload)
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

    async def test_mock_representation_manager(
        self,
        mock_representation_manager: Any,  # AsyncMock object
    ):
        """Test that the representation manager is properly mocked"""
        assert mock_representation_manager is not None

        # Verify we can call the mocked methods
        await mock_representation_manager.save_representation(
            Representation(explicit=[], deductive=[])
        )
        mock_representation_manager.get_relevant_observations.return_value = []  # type: ignore[attr-defined]

        # Verify the methods were called
        assert mock_representation_manager.save_representation.called  # type: ignore[attr-defined]

    async def test_representation_batch_uses_earliest_cutoff(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
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

        # Avoid executing the full reasoning pipeline; we only care about cutoff behavior.
        monkeypatch.setattr(
            "src.deriver.deriver.CertaintyReasoner.reason",
            AsyncMock(return_value=Representation(explicit=[], deductive=[])),
        )

        # Use the real session and workspace from fixtures
        session, peers = sample_session_with_peers
        alice = peers[0]

        # Create test messages with different IDs in the database
        now = datetime.now(timezone.utc)
        messages: list[models.Message] = []
        for i in range(8):
            message = models.Message(
                workspace_name=session.workspace_name,
                session_name=session.name,
                peer_name=alice.name,
                content=f"message {i}",
                seq_in_session=i + 1,
                token_count=10,
                created_at=now - timedelta(minutes=7 - i),
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh messages to get their IDs
        for message in messages:
            await db_session.refresh(message)

        await process_representation_tasks_batch(
            observer=alice.name, observed=alice.name, messages=messages
        )

        # Verify that the earliest message ID was used as the cutoff
        assert captured_cutoffs == [messages[0].id]

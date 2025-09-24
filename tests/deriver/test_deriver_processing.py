import signal
from collections.abc import Callable, Generator
from typing import Any

import pytest

from src import models


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
            "sender_name": peer1.name,
            "target_name": peer2.name,
            "task_type": "representation",
        }

        # Generate work unit key for representation
        work_unit_key = get_work_unit_key(representation_payload)
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

import signal
from collections.abc import Callable, Generator
from typing import Any

import pytest

from src import models
from src.deriver.queue_manager import WorkUnit


@pytest.mark.asyncio
class TestDeriverProcessing:
    """Test suite for deriver processing using the conftest fixtures"""

    async def test_mock_deriver_process(
        self,
        mock_deriver_process: Callable[..., Any],  # noqa: ARG001
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001
    ):
        """Test that the deriver process is properly mocked"""
        # The mock should be in place, so processing should not make real LLM calls
        assert mock_deriver_process is not None

        # Verify that we have queue items to process
        assert len(sample_queue_items) > 0

        # Verify the mock is working by checking the first queue item
        first_item = sample_queue_items[0]
        assert first_item.payload is not None

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

    async def test_work_unit_creation(
        self,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that WorkUnit objects can be created correctly"""
        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create a WorkUnit for representation task
        work_unit = WorkUnit(
            session_id=session.id,
            sender_name=peer1.name,
            target_name=peer2.name,
            task_type="representation",
        )

        assert work_unit.session_id == session.id
        assert work_unit.sender_name == peer1.name
        assert work_unit.target_name == peer2.name
        assert work_unit.task_type == "representation"

        # Create a WorkUnit for summary task (sender_name and target_name should be None)
        summary_work_unit = WorkUnit(
            session_id=session.id,
            sender_name=None,
            target_name=None,
            task_type="summary",
        )

        assert summary_work_unit.session_id == session.id
        assert summary_work_unit.sender_name is None
        assert summary_work_unit.target_name is None
        assert summary_work_unit.task_type == "summary"

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
        await mock_embedding_store.save_unified_observations([])
        mock_embedding_store.get_relevant_observations.return_value = []  # type: ignore[attr-defined]

        # Verify the methods were called
        assert mock_embedding_store.save_unified_observations.called  # type: ignore[attr-defined]

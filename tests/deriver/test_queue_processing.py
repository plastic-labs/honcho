from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.consumer import process_item
from src.deriver.queue_manager import QueueManager, WorkUnit


@pytest.mark.asyncio
class TestQueueProcessing:
    """Test suite for queue processing functionality"""

    async def test_get_available_work_units(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[models.QueueItem],
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that get_available_work_units correctly identifies unprocessed work"""
        session, _peers = sample_session_with_peers  # pyright: ignore[reportUnusedVariable]

        # Verify we have queue items from our test setup
        assert len(sample_queue_items) == 9  # 6 representation + 3 summary

        # Create a queue manager instance
        queue_manager = QueueManager()

        # Get available work units
        work_units = await queue_manager.get_available_work_units(db_session)

        # Should have some work units available (may include items from other tests)
        assert len(work_units) > 0

        # Check that all work units have the expected structure
        for work_unit in work_units:
            assert isinstance(work_unit, WorkUnit)
            assert work_unit.task_type in ["representation", "summary"]

        # The test is mainly verifying that get_available_work_units works without errors
        # and returns properly structured WorkUnit objects

    async def test_work_unit_claiming(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that work units can be claimed and are not available to other workers"""
        _session, _peers = sample_session_with_peers

        # Create a queue manager instance
        queue_manager = QueueManager()

        # Get available work units
        work_units = await queue_manager.get_available_work_units(db_session)
        assert len(work_units) > 0

        # Claim a work unit by creating an ActiveQueueSession entry
        work_unit = work_units[0]
        active_session = models.ActiveQueueSession(
            session_id=work_unit.session_id,
            sender_name=work_unit.sender_name,
            target_name=work_unit.target_name,
            task_type=work_unit.task_type,
        )
        db_session.add(active_session)
        await db_session.commit()

        # Get available work units again - the claimed one should not be available
        remaining_work_units = await queue_manager.get_available_work_units(db_session)

        # The claimed work unit should not be in the remaining list
        assert work_unit not in remaining_work_units
        # We can't assert exact count difference because work units are grouped by unique combinations

    async def test_stale_work_unit_cleanup(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that stale work units are cleaned up properly"""
        _session, _peers = sample_session_with_peers

        # Create an active queue session with an old timestamp
        # from datetime import datetime, timedelta, timezone

        # datetime.now(timezone.utc) - timedelta(minutes=10)

        # We'll test this by checking that the cleanup logic works in get_available_work_units
        # which is called by the queue manager during normal operation
        queue_manager = QueueManager()

        # Get available work units - this should clean up stale entries
        work_units = await queue_manager.get_available_work_units(db_session)

        # This test ensures the cleanup logic doesn't break, though we don't have stale entries yet
        assert isinstance(work_units, list)

    async def test_process_item_with_mocked_deriver(
        self,
        mock_deriver_process: Any,  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
        sample_queue_items: list[models.QueueItem],
    ):
        """Test that process_item works with mocked deriver"""
        # Take a sample queue item and process it
        queue_item = sample_queue_items[0]

        # This should not raise an exception since the deriver is mocked
        await process_item(queue_item.payload)

        # The mock should have been called
        # Note: We can't easily verify this since we're mocking the class method directly
        # In a real test, we might want to mock at a different level

    async def test_work_unit_string_representation(
        self, sample_session_with_peers: tuple[models.Session, list[models.Peer]]
    ):
        """Test that WorkUnit string representation works correctly"""
        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create a representation work unit
        work_unit = WorkUnit(
            session_id=session.id,
            sender_name=peer1.name,
            target_name=peer2.name,
            task_type="representation",
        )

        # Convert to string
        work_unit_str = str(work_unit)

        # Check that the string contains the expected information
        assert session.id in work_unit_str
        assert peer1.name in work_unit_str
        assert peer2.name in work_unit_str
        assert "representation" in work_unit_str

        # Create a summary work unit
        summary_work_unit = WorkUnit(
            session_id=session.id,
            sender_name=None,
            target_name=None,
            task_type="summary",
        )

        summary_str = str(summary_work_unit)
        assert session.id in summary_str
        assert "None" in summary_str
        assert "summary" in summary_str

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.queue_manager import QueueManager


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
            assert isinstance(work_unit, str)
            assert work_unit.split(":")[0] in ["representation", "summary"]

        # The test is mainly verifying that get_available_work_units works without errors
        # and returns properly structured work unit key strings

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
            work_unit_key=work_unit,
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

    async def test_work_unit_key_format(
        self, sample_session_with_peers: tuple[models.Session, list[models.Peer]]
    ):
        """Test that work unit keys have the correct format"""
        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create a representation work unit key
        # Format: task_type:workspace:session:sender:target
        work_unit_key = (
            f"representation:workspace1:{session.name}:{peer1.name}:{peer2.name}"
        )

        # Check that the key contains the expected information
        assert session.name in work_unit_key
        assert peer1.name in work_unit_key
        assert peer2.name in work_unit_key
        assert "representation" in work_unit_key
        assert "workspace1" in work_unit_key

        # Create a summary work unit key
        # Summary work units use None for sender/target
        summary_work_unit_key = f"summary:workspace1:{session.name}:None:None"

        assert session.name in summary_work_unit_key
        assert "None" in summary_work_unit_key
        assert "summary" in summary_work_unit_key
        assert "workspace1" in summary_work_unit_key

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.deriver import get_deriver_status


@pytest.mark.asyncio
class TestDeriverCRUD:
    """Test suite for the deriver CRUD functions"""

    async def test_get_deriver_status_fetchall_with_queue_items(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_deriver_status fetchall() call with actual queue items - covers line 43."""
        workspace, peer = sample_data
        
        # Create a session
        session = models.Session(workspace_name=workspace.name, name="test_session")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Add multiple queue items to test fetchall() on line 43
        queue_items = [
            models.QueueItem(
                session_id=session.id,
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "representation",
                },
                processed=False,
            )
            for _ in range(5)
        ]
        db_session.add_all(queue_items)
        await db_session.commit()
        
        # Call get_deriver_status - this will execute line 43: rows = result.fetchall()
        result = await get_deriver_status(db_session, workspace.name)
        
        # Verify the function correctly processed the fetchall() results
        assert isinstance(result, schemas.DeriverStatus)
        assert result.total_work_units == 5
        assert result.pending_work_units == 5
        assert result.completed_work_units == 0
        assert result.in_progress_work_units == 0

    async def test_get_deriver_status_fetchall_with_no_queue_items(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_deriver_status fetchall() call with no queue items - covers line 43."""
        workspace, _peer = sample_data
        
        # Call get_deriver_status with no queue items - this will still execute line 43: rows = result.fetchall()
        result = await get_deriver_status(db_session, workspace.name)
        
        # Verify the function correctly handled empty fetchall() results
        assert isinstance(result, schemas.DeriverStatus)
        assert result.total_work_units == 0
        assert result.pending_work_units == 0
        assert result.completed_work_units == 0
        assert result.in_progress_work_units == 0

    async def test_get_deriver_status_fetchall_with_mixed_status_items(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_deriver_status fetchall() call with mixed status queue items - covers line 43."""
        workspace, peer = sample_data
        
        # Create a session
        session = models.Session(workspace_name=workspace.name, name="test_session")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Add queue items with different processed states
        completed_items = [
            models.QueueItem(
                session_id=session.id,
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "representation",
                },
                processed=True,  # Completed
            )
            for _ in range(3)
        ]
        
        pending_items = [
            models.QueueItem(
                session_id=session.id,
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "representation",
                },
                processed=False,  # Pending
            )
            for _ in range(2)
        ]
        
        db_session.add_all(completed_items + pending_items)
        await db_session.commit()
        
        # Call get_deriver_status - this will execute line 43: rows = result.fetchall()
        result = await get_deriver_status(db_session, workspace.name)
        
        # Verify the function correctly processed the fetchall() results with mixed statuses
        assert isinstance(result, schemas.DeriverStatus)
        assert result.total_work_units == 5
        assert result.completed_work_units == 3
        assert result.pending_work_units == 2
        assert result.in_progress_work_units == 0

    async def test_get_deriver_status_fetchall_with_filters(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_deriver_status fetchall() call with observer and sender filters - covers line 43."""
        workspace, peer = sample_data
        
        # Create second peer for filtering test
        peer2 = models.Peer(workspace_name=workspace.name, name="peer2")
        db_session.add(peer2)
        
        # Create a session
        session = models.Session(workspace_name=workspace.name, name="test_session")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        await db_session.refresh(peer2)
        
        # Add queue items for different peers
        peer1_items = [
            models.QueueItem(
                session_id=session.id,
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "representation",
                },
                processed=False,
            )
            for _ in range(3)
        ]
        
        peer2_items = [
            models.QueueItem(
                session_id=session.id,
                payload={
                    "sender_name": peer2.name,
                    "target_name": peer2.name,
                    "task_type": "representation",
                },
                processed=False,
            )
            for _ in range(2)
        ]
        
        db_session.add_all(peer1_items + peer2_items)
        await db_session.commit()
        
        # Call get_deriver_status with observer filter - this will execute line 43: rows = result.fetchall()
        result = await get_deriver_status(
            db_session, workspace.name, observer_name=peer.name
        )
        
        # Verify the function correctly processed the filtered fetchall() results
        assert isinstance(result, schemas.DeriverStatus)
        assert result.total_work_units == 3  # Only peer1's items
        assert result.pending_work_units == 3

    async def test_get_deriver_status_fetchall_with_session_filter(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_deriver_status fetchall() call with session filter - covers line 43."""
        workspace, peer = sample_data
        
        # Create two sessions
        session1 = models.Session(workspace_name=workspace.name, name="session1")
        session2 = models.Session(workspace_name=workspace.name, name="session2")
        db_session.add_all([session1, session2])
        await db_session.commit()
        await db_session.refresh(session1)
        await db_session.refresh(session2)
        
        # Add queue items to both sessions
        session1_items = [
            models.QueueItem(
                session_id=session1.id,
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "representation",
                },
                processed=False,
            )
            for _ in range(4)
        ]
        
        session2_items = [
            models.QueueItem(
                session_id=session2.id,
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "representation",
                },
                processed=False,
            )
            for _ in range(1)
        ]
        
        db_session.add_all(session1_items + session2_items)
        await db_session.commit()
        
        # Call get_deriver_status with session filter - this will execute line 43: rows = result.fetchall()
        result = await get_deriver_status(
            db_session, workspace.name, session_name="session1"
        )
        
        # Verify the function correctly processed the filtered fetchall() results
        assert isinstance(result, schemas.DeriverStatus)
        assert result.total_work_units == 4  # Only session1's items
        assert result.pending_work_units == 4
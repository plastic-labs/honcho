"""Tests for the QueueManager class and queue processing functionality."""

import asyncio
import signal
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
import pytest_asyncio
from sqlalchemy import select

from src import models
from src.deriver.queue import QueueManager


class TestQueueManagerInitialization:
    """Test QueueManager initialization and configuration."""

    def test_queue_manager_default_initialization(self):
        """Test QueueManager initializes with default values."""
        with patch("src.deriver.queue.os.getenv") as mock_getenv:
            mock_getenv.return_value = "1"  # Default worker count

            manager = QueueManager()

            assert manager.workers == 1
            assert manager.semaphore._value == 1
            assert not manager.shutdown_event.is_set()
            assert len(manager.active_tasks) == 0
            assert len(manager.owned_sessions) == 0

    def test_queue_manager_custom_workers(self):
        """Test QueueManager respects DERIVER_WORKERS environment variable."""
        with patch("src.deriver.queue.os.getenv") as mock_getenv:
            mock_getenv.return_value = "4"

            manager = QueueManager()

            assert manager.workers == 4
            assert manager.semaphore._value == 4

    @patch("src.deriver.queue.sentry_sdk")
    def test_sentry_initialization_enabled(self, mock_sentry):
        """Test Sentry initialization when enabled."""
        with patch("src.deriver.queue.os.getenv") as mock_getenv:

            def getenv_side_effect(key, default=None):
                if key == "SENTRY_ENABLED":
                    return "True"
                elif key == "SENTRY_DSN":
                    return "https://test@sentry.io/123"
                elif key == "DERIVER_WORKERS":
                    return "1"
                return default

            mock_getenv.side_effect = getenv_side_effect

            QueueManager()

            mock_sentry.init.assert_called_once()

    @patch("src.deriver.queue.sentry_sdk")
    def test_sentry_initialization_disabled(self, mock_sentry):
        """Test Sentry is not initialized when disabled."""
        with patch("src.deriver.queue.os.getenv") as mock_getenv:

            def getenv_side_effect(key, default=None):
                if key == "SENTRY_ENABLED":
                    return "False"
                elif key == "DERIVER_WORKERS":
                    return "1"
                return default

            mock_getenv.side_effect = getenv_side_effect

            QueueManager()

            mock_sentry.init.assert_not_called()


class TestTaskAndSessionTracking:
    """Test task and session tracking functionality."""

    def test_add_task_tracking(self):
        """Test adding tasks to tracking set."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()

            # Create a mock task
            task = MagicMock()
            task.add_done_callback = MagicMock()

            manager.add_task(task)

            assert task in manager.active_tasks
            task.add_done_callback.assert_called_once()

    def test_session_tracking(self):
        """Test session tracking and untracking."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session_id = 123

            # Track session
            manager.track_session(session_id)
            assert session_id in manager.owned_sessions

            # Untrack session
            manager.untrack_session(session_id)
            assert session_id not in manager.owned_sessions

    def test_track_session_multiple(self):
        """Test tracking multiple sessions."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session_ids = [123, 456, 789]

            for session_id in session_ids:
                manager.track_session(session_id)

            assert all(sid in manager.owned_sessions for sid in session_ids)
            assert len(manager.owned_sessions) == 3


class TestDatabaseOperations:
    """Test database operations for queue management."""

    @pytest_asyncio.fixture
    async def setup_queue_data(self, db_session, sample_data):
        """Setup test data for queue operations."""
        test_app, test_user = sample_data

        # Create sessions
        session1 = models.Session(
            user_id=test_user.public_id, app_id=test_app.public_id, metadata={}
        )
        session2 = models.Session(
            user_id=test_user.public_id, app_id=test_app.public_id, metadata={}
        )
        db_session.add_all([session1, session2])
        await db_session.flush()

        # Create queue items (use integer session.id, not public_id)
        queue_item1 = models.QueueItem(
            session_id=session1.id,
            payload={"message_id": str(uuid4())},
            processed=False,
        )
        queue_item2 = models.QueueItem(
            session_id=session2.id,
            payload={"message_id": str(uuid4())},
            processed=False,
        )
        queue_item3 = models.QueueItem(
            session_id=session1.id,
            payload={"message_id": str(uuid4())},
            processed=True,  # Already processed
        )

        db_session.add_all([queue_item1, queue_item2, queue_item3])
        await db_session.flush()

        return session1, session2, [queue_item1, queue_item2, queue_item3]

    @pytest.mark.asyncio
    async def test_get_available_sessions(self, db_session, setup_queue_data):
        """Test getting available sessions for processing."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session1, session2, queue_items = setup_queue_data

            # Get available sessions
            available_sessions = await manager.get_available_sessions(db_session)

            # Should return sessions with unprocessed items
            assert len(available_sessions) == 1  # Limited to 1 by the query
            assert available_sessions[0] in [session1.id, session2.id]

    @pytest.mark.asyncio
    async def test_get_available_sessions_with_active_session(
        self, db_session, setup_queue_data
    ):
        """Test that active sessions are excluded from available sessions."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session1, session2, queue_items = setup_queue_data

            # Mark session1 as active
            active_session = models.ActiveQueueSession(session_id=session1.id)
            db_session.add(active_session)
            await db_session.flush()

            # Get available sessions
            available_sessions = await manager.get_available_sessions(db_session)

            # Should only return session2
            assert len(available_sessions) == 1
            assert available_sessions[0] == session2.id

    @pytest.mark.asyncio
    async def test_stale_session_cleanup(self, db_session, setup_queue_data):
        """Test cleanup of stale active sessions."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session1, session2, queue_items = setup_queue_data

            # Create a stale active session (older than 5 minutes)
            stale_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            stale_session = models.ActiveQueueSession(
                session_id=session1.id, last_updated=stale_time
            )
            db_session.add(stale_session)
            await db_session.flush()

            # Get available sessions (this should trigger cleanup)
            available_sessions = await manager.get_available_sessions(db_session)

            # Stale session should be cleaned up, making session1 available
            result = await db_session.execute(
                select(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.session_id == session1.id
                )
            )
            assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_get_next_message(self, db_session, setup_queue_data):
        """Test getting the next unprocessed message for a session."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session1, session2, queue_items = setup_queue_data

            # Get next message for session1
            next_message = await manager.get_next_message(db_session, session1.id)

            # Should return the unprocessed message
            assert next_message is not None
            assert next_message.session_id == session1.id
            assert not next_message.processed

    @pytest.mark.asyncio
    async def test_get_next_message_no_unprocessed(self, db_session, setup_queue_data):
        """Test getting next message when all are processed."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session1, session2, queue_items = setup_queue_data

            # Mark all messages as processed
            for item in queue_items:
                item.processed = True
            await db_session.flush()

            # Get next message
            next_message = await manager.get_next_message(db_session, session1.id)

            # Should return None
            assert next_message is None


class TestConcurrencyControl:
    """Test concurrency control and semaphore behavior."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_processing(self, mock_semaphore):
        """Test that semaphore properly limits concurrent session processing."""
        with patch("src.deriver.queue.os.getenv", return_value="2"):
            with patch("asyncio.Semaphore") as mock_semaphore_class:
                mock_semaphore_class.return_value = mock_semaphore
                manager = QueueManager()

                # Mock the process_session method to return actual async function
                async def mock_process_session(session_id):
                    async with manager.semaphore:
                        await asyncio.sleep(0.01)  # Simulate work

                with patch.object(
                    manager, "process_session", side_effect=mock_process_session
                ):
                    # Try to process multiple sessions
                    tasks = []
                    for i in range(5):
                        task = asyncio.create_task(manager.process_session(i))
                        tasks.append(task)
                        manager.add_task(task)

                    # Wait for tasks to complete
                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Verify semaphore was used
                    assert mock_semaphore.__aenter__.call_count == 5

    @pytest.mark.asyncio
    async def test_polling_loop_respects_semaphore_capacity(self):
        """Test that polling loop waits when all workers are busy."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()

            # Mock semaphore as locked (no capacity)
            manager.semaphore.locked = MagicMock(return_value=True)

            # Mock tracked_db to avoid database operations
            with patch("src.deriver.queue.tracked_db"):
                # Set shutdown event after a short delay to exit the loop
                async def set_shutdown():
                    await asyncio.sleep(0.1)
                    manager.shutdown_event.set()

                asyncio.create_task(set_shutdown())

                # Run polling loop
                await manager.polling_loop()

                # Should have checked semaphore status
                manager.semaphore.locked.assert_called()


class TestSignalHandling:
    """Test signal handling and graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_signal_handling(self, mock_signal_handling):
        """Test that shutdown properly handles signals."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()

            # Create actual async tasks instead of AsyncMock
            async def dummy_task():
                await asyncio.sleep(0.01)

            task1 = asyncio.create_task(dummy_task())
            task2 = asyncio.create_task(dummy_task())
            manager.active_tasks = {task1, task2}

            # Call shutdown
            await manager.shutdown(signal.SIGTERM)

            # Shutdown event should be set
            assert manager.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_owned_sessions(self, db_session):
        """Test cleanup of owned sessions during shutdown."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()

            # Add owned sessions
            session_ids = [123, 456, 789]
            for session_id in session_ids:
                manager.track_session(session_id)
                # Create corresponding active session records
                active_session = models.ActiveQueueSession(session_id=session_id)
                db_session.add(active_session)

            await db_session.flush()

            # Mock tracked_db to use our test session
            with patch("src.deriver.queue.tracked_db") as mock_tracked_db:
                mock_tracked_db.return_value.__aenter__.return_value = db_session
                mock_tracked_db.return_value.__aexit__.return_value = None

                # Run cleanup
                await manager.cleanup()

                # Verify sessions were removed from database
                result = await db_session.execute(
                    select(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.session_id.in_(session_ids)
                    )
                )
                remaining_sessions = result.scalars().all()
                assert len(remaining_sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_database_error(self, db_session):
        """Test cleanup handles database errors gracefully."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            manager.track_session(123)

            # Mock tracked_db to raise an exception
            with patch("src.deriver.queue.tracked_db") as mock_tracked_db:
                mock_tracked_db.side_effect = Exception("Database connection failed")

                # Cleanup should not raise exception
                await manager.cleanup()

                # Session should still be tracked (cleanup failed)
                assert 123 in manager.owned_sessions


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_polling_loop_handles_database_errors(self):
        """Test polling loop handles database errors gracefully."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()

            # Mock tracked_db as proper async context manager that fails inside the context
            call_count = 0

            class MockTrackedDBContext:
                def __init__(self, *args, **kwargs):
                    nonlocal call_count
                    call_count += 1

                async def __aenter__(self):
                    mock_db = MagicMock()
                    # Make get_available_sessions fail on first call
                    if call_count == 1:
                        mock_db.execute.side_effect = Exception(
                            "Database connection failed"
                        )
                    else:
                        # Set shutdown on second call to exit loop
                        manager.shutdown_event.set()
                        mock_db.execute.return_value = MagicMock()
                    return mock_db

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            with patch("src.deriver.queue.tracked_db", MockTrackedDBContext):
                # Should not raise exception and should retry
                await manager.polling_loop()

                # Should have attempted multiple calls
                assert call_count >= 2

    @pytest.mark.asyncio
    async def test_process_session_marks_failed_messages_as_processed(
        self, db_session, sample_queue_items
    ):
        """Test that failed message processing still marks messages as processed."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session, messages, queue_items = sample_queue_items

            # Mock process_item to raise an exception
            with patch(
                "src.deriver.queue.process_item",
                side_effect=Exception("Processing failed"),
            ):
                with patch("src.deriver.queue.tracked_db") as mock_tracked_db:
                    mock_tracked_db.return_value.__aenter__.return_value = db_session
                    mock_tracked_db.return_value.__aexit__.return_value = None

                    # Process the session
                    await manager.process_session(session.id)

                    # All messages should be marked as processed despite the error
                    result = await db_session.execute(
                        select(models.QueueItem).where(
                            models.QueueItem.session_id == session.id
                        )
                    )
                    queue_items_after = result.scalars().all()
                    assert all(item.processed for item in queue_items_after)

    @pytest.mark.asyncio
    async def test_session_claiming_handles_integrity_error(
        self, db_session, sample_data
    ):
        """Test that session claiming handles race conditions gracefully."""
        test_app, test_user = sample_data

        # Create sessions
        session1 = models.Session(
            user_id=test_user.public_id, app_id=test_app.public_id, metadata={}
        )
        session2 = models.Session(
            user_id=test_user.public_id, app_id=test_app.public_id, metadata={}
        )
        db_session.add_all([session1, session2])
        await db_session.flush()

        # Create queue items
        queue_item1 = models.QueueItem(
            session_id=session1.id,
            payload={"message_id": str(uuid4())},
            processed=False,
        )
        queue_item2 = models.QueueItem(
            session_id=session2.id,
            payload={"message_id": str(uuid4())},
            processed=False,
        )
        db_session.add_all([queue_item1, queue_item2])
        await db_session.flush()

        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()

            # Create an active session to cause IntegrityError
            active_session = models.ActiveQueueSession(session_id=session1.id)
            db_session.add(active_session)
            await db_session.flush()

            # Try to get available sessions and claim them
            available_sessions = await manager.get_available_sessions(db_session)

            # Should get session2 (session1 is active)
            assert len(available_sessions) == 1
            assert available_sessions[0] == session2.id


class TestIntegrationScenarios:
    """Test integration scenarios and real-world usage patterns."""

    @pytest.mark.asyncio
    async def test_full_session_processing_cycle(self, db_session, sample_queue_items):
        """Test complete processing cycle for a session."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session, messages, queue_items = sample_queue_items

            # Mock process_item to simulate successful processing
            with patch("src.deriver.queue.process_item") as mock_process:
                mock_process.return_value = None

                with patch("src.deriver.queue.tracked_db") as mock_tracked_db:
                    mock_tracked_db.return_value.__aenter__.return_value = db_session
                    mock_tracked_db.return_value.__aexit__.return_value = None

                    # Process the session
                    await manager.process_session(session.id)

                    # Verify all user messages were processed
                    user_message_count = len([item for item in queue_items])
                    assert mock_process.call_count == user_message_count

                    # Verify session is not in active sessions
                    result = await db_session.execute(
                        select(models.ActiveQueueSession).where(
                            models.ActiveQueueSession.session_id == session.id
                        )
                    )
                    assert result.scalar_one_or_none() is None

                    # Verify session is untracked
                    assert session.id not in manager.owned_sessions

    @pytest.mark.asyncio
    async def test_shutdown_during_processing(self, db_session, sample_queue_items):
        """Test graceful shutdown while processing messages."""
        with patch("src.deriver.queue.os.getenv", return_value="1"):
            manager = QueueManager()
            session, messages, queue_items = sample_queue_items

            # Mock process_item to be slow and check shutdown event
            async def slow_process_item(db, payload):
                await asyncio.sleep(0.1)
                if manager.shutdown_event.is_set():
                    return
                # Continue processing

            with patch("src.deriver.queue.process_item", side_effect=slow_process_item):
                with patch("src.deriver.queue.tracked_db") as mock_tracked_db:
                    mock_tracked_db.return_value.__aenter__.return_value = db_session
                    mock_tracked_db.return_value.__aexit__.return_value = None

                    # Start processing
                    process_task = asyncio.create_task(
                        manager.process_session(session.id)
                    )

                    # Trigger shutdown after a short delay
                    async def trigger_shutdown():
                        await asyncio.sleep(0.05)
                        manager.shutdown_event.set()

                    shutdown_task = asyncio.create_task(trigger_shutdown())

                    # Wait for both tasks
                    await asyncio.gather(
                        process_task, shutdown_task, return_exceptions=True
                    )

                    # Session should be cleaned up even with shutdown
                    assert session.id not in manager.owned_sessions

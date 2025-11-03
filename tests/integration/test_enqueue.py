from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.deriver import enqueue
from src.deriver.enqueue import generate_queue_records
from src.models import Peer, QueueItem, Workspace


@pytest.mark.asyncio
class TestEnqueueFunction:
    """Test suite for the enqueue function's internal logic"""

    # Helper methods
    async def create_sample_payload(
        self,
        db_session: AsyncSession,
        workspace_name: str = "test_workspace",
        session_name: str | None = "test_session",
        peer_name: str = "test_peer",
        count: int = 1,
    ) -> list[dict[str, Any]]:
        """Create real messages in database and return payload with actual IDs"""
        # Get the current max sequence number for this session
        result = await db_session.execute(
            select(func.max(models.Message.seq_in_session)).where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
        )
        current_max_seq = result.scalar() or 0

        messages: list[models.Message] = []
        for i in range(count):
            message = models.Message(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=peer_name,
                content=f"Test message {i}",
                public_id=generate_nanoid(),
                seq_in_session=current_max_seq + i + 1,
                token_count=10,
                h_metadata={"test": f"value_{i}"},
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Return payload with real message IDs
        return [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": msg.id,
                "content": msg.content,
                "metadata": msg.h_metadata,
                "peer_name": peer_name,
                "created_at": msg.created_at,
            }
            for msg in messages
        ]

    async def count_queue_items(self, db_session: AsyncSession):
        """Helper to count queue items in database"""
        result = await db_session.execute(select(QueueItem))
        return len(result.scalars().all())

    # Edge Cases & Input Validation Tests
    @pytest.mark.asyncio
    async def test_empty_payload_skips_enqueue(self, caplog: pytest.LogCaptureFixture):
        """Test that empty payload list is handled gracefully"""
        with caplog.at_level("DEBUG"):
            await enqueue([])

    @pytest.mark.asyncio
    async def test_malformed_payload_logs_error(self, caplog: pytest.LogCaptureFixture):
        """Test that malformed payload logs appropriate error"""
        malformed_payload = [{"incomplete": "data"}]

        with caplog.at_level("ERROR"):
            await enqueue(malformed_payload)  # Should not raise, but log error

    # SESSION MESSAGES
    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_session_with_deriver_disabled(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that deriver disabled sessions skip representation but allows summary"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        # Create session with deriver disabled
        test_session = models.Session(
            workspace_name=test_workspace.name,
            name=str(generate_nanoid()),
            configuration={"deriver_disabled": True},
        )
        db_session.add(test_session)
        await db_session.commit()

        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # When deriver is disabled, only summary records should be created (if applicable)
        # Since this is message 1, and 1 % 20 != 0 and 1 % 60 != 0, no summary should be created
        # No representation records should be created either (deriver disabled)
        assert (
            final_count == initial_count
        ), f"Expected no queue items, but got {final_count - initial_count}"

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_session_normal_processing_single_peer(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={test_peer.name: schemas.SessionPeerConfig(observe_me=True)},
            ),
            test_workspace.name,
        )
        await db_session.commit()

        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 1 queue items: 1 representation
        assert final_count - initial_count == 1

        # Verify queue items
        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        task_types = [item.payload["task_type"] for item in queue_items]
        assert "representation" in task_types

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_session_with_multiple_peers_none_observe_others(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test session processing with multiple peers where some observe others"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer1 = sample_data

        # Create second peer
        test_peer2 = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_peer2)

        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    test_peer1.name: schemas.SessionPeerConfig(),
                    test_peer2.name: schemas.SessionPeerConfig(),
                },
            ),
            test_workspace.name,
        )

        await db_session.commit()

        NUM_MESSAGES = 3
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer1.name,
            count=NUM_MESSAGES,  # Message from peer1
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create NUM_MESSAGES * queue items:
        # 1 representation per sender
        assert final_count - initial_count == NUM_MESSAGES

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        # Explicitly match up payloads by sender/target/task_type
        # For each message, we expect a representation
        expected_payloads: list[dict[str, Any]] = []
        for _ in range(NUM_MESSAGES):
            expected_payloads.append(
                {
                    "observed": test_peer1.name,
                    "observer": test_peer1.name,
                    "task_type": "representation",
                }
            )
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]
        assert len(actual_payloads) == len(expected_payloads)

        # Assert that all expected payloads are present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_session_with_multiple_peers_all_observe_others(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test session processing with multiple peers where some observe others"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer1 = sample_data

        # Create second peer
        test_peer2 = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_peer2)

        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    test_peer1.name: schemas.SessionPeerConfig(),
                    test_peer2.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        NUM_MESSAGES = 3
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer1.name,
            count=NUM_MESSAGES,  # Message from peer1
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create NUM_MESSAGES * 2 queue items:
        # 1 representation for sender, 1 local representation for observer
        assert final_count - initial_count == NUM_MESSAGES * 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        # Explicitly match up payloads by sender/target/task_type
        # For each message, we expect a representation and a local representation for observer
        expected_payloads: list[dict[str, Any]] = []
        for _ in range(NUM_MESSAGES):
            expected_payloads.append(
                {
                    "observed": test_peer1.name,
                    "observer": test_peer1.name,
                    "task_type": "representation",
                }
            )
            expected_payloads.append(
                {
                    "observed": test_peer1.name,
                    "observer": test_peer2.name,
                    "task_type": "representation",
                }
            )
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]
        assert len(actual_payloads) == len(expected_payloads)

        # Assert that all expected payloads are present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_session_with_multiple_peers_some_observe_others(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test session processing with multiple peers where some observe others"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer1 = sample_data

        # Create second peer
        observing_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(observing_peer)

        # Create third peer
        unobserving_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(unobserving_peer)

        # Create session
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    test_peer1.name: schemas.SessionPeerConfig(observe_me=True),
                    observing_peer.name: schemas.SessionPeerConfig(observe_others=True),
                    unobserving_peer.name: schemas.SessionPeerConfig(),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        NUM_MESSAGES = 3
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer1.name,
            count=NUM_MESSAGES,  # Message from peer1
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create NUM_MESSAGES * 2 queue items:
        # 1 representation for sender, 1 local representation for 1 peer observer
        assert final_count - initial_count == NUM_MESSAGES * 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        # Explicitly match up payloads by sender/target/task_type
        # For each message, we expect a representation and a local representation for observer
        expected_payloads: list[dict[str, Any]] = []
        for _ in range(NUM_MESSAGES):
            expected_payloads.append(
                {
                    "observed": test_peer1.name,
                    "observer": test_peer1.name,
                    "task_type": "representation",
                }
            )
            expected_payloads.append(
                {
                    "observed": test_peer1.name,
                    "observer": observing_peer.name,
                    "task_type": "representation",
                }
            )
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]
        assert len(actual_payloads) == len(expected_payloads)

        # Assert that all expected payloads are present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

        assert unobserving_peer.name not in [
            item.payload.get("observer") for item in queue_items
        ]

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_session_peer_config_overrides_peer_config(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that session peer config overrides peer config"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        # Set peer configuration to observe_me=True
        test_peer.configuration = {"observe_me": True}

        # Create session
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={test_peer.name: schemas.SessionPeerConfig(observe_me=False)},
            ),
            test_workspace.name,
        )

        await db_session.commit()

        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 0 queue items
        assert final_count - initial_count == 0

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_multi_sender_scenario(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test complex scenario with multiple peers and mixed configurations"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer1 = sample_data

        # Create additional peers
        additional_sender_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )

        observer_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )

        db_session.add_all([additional_sender_peer, observer_peer])

        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    test_peer1.name: schemas.SessionPeerConfig(),
                    additional_sender_peer.name: schemas.SessionPeerConfig(),
                    observer_peer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        payload1 = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer1.name,
        )

        payload2 = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=additional_sender_peer.name,
        )

        payload = payload1 + payload2

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 4 queue items:
        # 1 representation for test_peer1 (sender) and observer_peer (target)
        # 1 representation for additional_sender_peer (sender) and observer_peer (target)
        assert final_count - initial_count == 4

        # Verify the correct representations were created
        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.all()

        # Build expected payloads for this scenario
        expected_payloads: list[dict[str, Any]] = []
        # 2 messages: one from test_peer1, one from additional_sender_peer
        for sender in [test_peer1.name, additional_sender_peer.name]:
            # representation for sender (self)
            expected_payloads.append(
                {
                    "task_type": "representation",
                    "observed": sender,
                    "observer": sender,
                }
            )
            # representation for observer_peer (observe_others=True)
            expected_payloads.append(
                {
                    "task_type": "representation",
                    "observed": sender,
                    "observer": observer_peer.name,
                }
            )

        # Extract actual payloads (task_type, observed, observer) from queue_items
        actual_payloads = [
            {
                "task_type": item[0].payload.get("task_type"),
                "observed": item[0].payload.get("observed"),
                "observer": item[0].payload.get("observer"),
            }
            for item in queue_items
        ]

        assert len(actual_payloads) == len(expected_payloads)

        # For each expected payload, assert it is present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

    # RACE CONDITION TESTS - Testing the new logic for peers that have left
    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_sender_left_session_after_message_sent(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that messages from senders who left the session still get processed with default config"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, sender_peer = sample_data

        # Create an observer peer
        observer_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(observer_peer)

        # Create session with both peers
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    sender_peer.name: schemas.SessionPeerConfig(observe_me=True),
                    observer_peer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Simulate sender leaving the session by setting left_at

        session_peer_result = await db_session.execute(
            select(models.SessionPeer).where(
                models.SessionPeer.session_name == test_session.name,
                models.SessionPeer.peer_name == sender_peer.name,
                models.SessionPeer.workspace_name == test_workspace.name,
            )
        )
        session_peer = session_peer_result.scalar_one()
        session_peer.left_at = datetime.now(timezone.utc)
        await db_session.commit()

        # Create message payload from the peer who left
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=sender_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 2 queue items:
        # 1 representation for sender (using default config since they left)
        # 1 representation for observer (still in session and observing others)
        assert final_count - initial_count == 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        expected_payloads = [
            {
                "observed": sender_peer.name,
                "observer": sender_peer.name,
                "task_type": "representation",
            },
            {
                "observed": sender_peer.name,
                "observer": observer_peer.name,
                "task_type": "representation",
            },
        ]
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]

        assert len(actual_payloads) == len(expected_payloads)
        for expected in expected_payloads:
            assert expected in actual_payloads

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_observer_left_session_no_queue_items_generated(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that peers who left the session don't get representation tasks enqueued"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, sender_peer = sample_data

        # Create observer peers - one will leave, one will stay
        observer_who_left = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        observer_who_stayed = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add_all([observer_who_left, observer_who_stayed])

        # Create session with all peers
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    sender_peer.name: schemas.SessionPeerConfig(observe_me=True),
                    observer_who_left.name: schemas.SessionPeerConfig(
                        observe_others=True
                    ),
                    observer_who_stayed.name: schemas.SessionPeerConfig(
                        observe_others=True
                    ),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Simulate one observer leaving the session

        session_peer_result = await db_session.execute(
            select(models.SessionPeer).where(
                models.SessionPeer.session_name == test_session.name,
                models.SessionPeer.peer_name == observer_who_left.name,
                models.SessionPeer.workspace_name == test_workspace.name,
            )
        )
        session_peer = session_peer_result.scalar_one()
        session_peer.left_at = datetime.now(timezone.utc)
        await db_session.commit()

        # Create message payload
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=sender_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 2 queue items:
        # 1 representation for sender
        # 1 representation for observer_who_stayed (observer_who_left should be skipped)
        assert final_count - initial_count == 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        # Verify observer_who_left is NOT in the target names
        observers = [item.payload.get("observer") for item in queue_items]
        assert observer_who_left.name not in observers
        assert observer_who_stayed.name in observers
        assert sender_peer.name in observers

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_sender_not_in_peer_configuration_uses_defaults(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test get_effective_observe_me handles missing sender configuration gracefully"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, existing_peer = sample_data

        # Create observer peer
        observer_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(observer_peer)

        # Create session with only observer (sender not in peers_with_configuration)
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    observer_peer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Create message from peer NOT in the session configuration
        # This simulates the race condition where a peer left after sending
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=existing_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 2 queue items:
        # 1 representation for unknown sender (using default observe_me=True)
        # 1 representation for observer (observing others)
        assert final_count - initial_count == 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        expected_payloads = [
            {
                "observed": existing_peer.name,
                "observer": existing_peer.name,
                "task_type": "representation",
            },
            {
                "observed": existing_peer.name,
                "observer": observer_peer.name,
                "task_type": "representation",
            },
        ]
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]

        assert len(actual_payloads) == len(expected_payloads)
        for expected in expected_payloads:
            assert expected in actual_payloads

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_mixed_active_inactive_peers_complex_scenario(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test complex scenario with mix of active/inactive peers and different configurations"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, sender_peer = sample_data

        # Create multiple peers with different roles
        active_observer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        inactive_observer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        active_non_observer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        inactive_non_observer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )

        db_session.add_all(
            [
                active_observer,
                inactive_observer,
                active_non_observer,
                inactive_non_observer,
            ]
        )

        # Create session with all peers having different configurations
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    sender_peer.name: schemas.SessionPeerConfig(observe_me=True),
                    active_observer.name: schemas.SessionPeerConfig(
                        observe_others=True
                    ),
                    inactive_observer.name: schemas.SessionPeerConfig(
                        observe_others=True
                    ),
                    active_non_observer.name: schemas.SessionPeerConfig(
                        observe_others=False
                    ),
                    inactive_non_observer.name: schemas.SessionPeerConfig(
                        observe_others=False
                    ),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Mark some peers as having left the session

        for peer_name in [inactive_observer.name, inactive_non_observer.name]:
            session_peer_result = await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.session_name == test_session.name,
                    models.SessionPeer.peer_name == peer_name,
                    models.SessionPeer.workspace_name == test_workspace.name,
                )
            )
            session_peer = session_peer_result.scalar_one()
            session_peer.left_at = datetime.now(timezone.utc)
        await db_session.commit()

        # Create message payload from sender
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=sender_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 2 queue items:
        # 1 representation for sender (observe_me=True)
        # 1 representation for active_observer (observe_others=True and still active)
        # inactive_observer should be skipped (left session)
        # active_non_observer should be skipped (observe_others=False)
        # inactive_non_observer should be skipped (left session)
        assert final_count - initial_count == 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        expected_payloads = [
            {
                "observed": sender_peer.name,
                "observer": sender_peer.name,
                "task_type": "representation",
            },
            {
                "observed": sender_peer.name,
                "observer": active_observer.name,
                "task_type": "representation",
            },
        ]
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]

        assert len(actual_payloads) == len(expected_payloads)
        for expected in expected_payloads:
            assert expected in actual_payloads

        # Verify inactive peers are not in target names
        observers = [item.payload.get("observer") for item in queue_items]
        assert inactive_observer.name not in observers
        assert inactive_non_observer.name not in observers
        assert active_non_observer.name not in observers


class TestGetEffectiveObserveMeFunction:
    """Unit tests for the get_effective_observe_me function specifically testing race condition handling"""

    def test_sender_missing_from_configuration_uses_default(self):
        """Test that missing sender uses default observe_me=True"""
        from src.deriver.enqueue import get_effective_observe_me

        # Empty peer configuration dict simulates sender who left after sending message
        peers_with_configuration: dict[str, list[dict[str, Any]]] = {}

        result = get_effective_observe_me("missing_sender", peers_with_configuration)

        # Should use default PeerConfig() which has observe_me=True
        assert result is True

    def test_sender_with_empty_configurations_uses_default(self):
        """Test that sender with empty peer and session configs uses default"""
        from src.deriver.enqueue import get_effective_observe_me

        # Sender present but with empty configurations
        peers_with_configuration: dict[str, list[dict[str, Any]]] = {
            "sender": [{}, {}]  # Empty peer config, empty session config
        }

        result = get_effective_observe_me("sender", peers_with_configuration)

        # Should use default PeerConfig() which has observe_me=True
        assert result is True

    def test_sender_with_peer_config_observe_me_false(self):
        """Test that peer config observe_me=False is respected"""
        from src.deriver.enqueue import get_effective_observe_me

        peers_with_configuration = {
            "sender": [{"observe_me": False}, {}]  # Peer config with observe_me=False
        }

        result = get_effective_observe_me("sender", peers_with_configuration)

        assert result is False

    def test_session_config_overrides_peer_config(self):
        """Test that session peer config takes precedence over peer config"""
        from src.deriver.enqueue import get_effective_observe_me

        peers_with_configuration = {
            "sender": [
                {"observe_me": True},  # Peer config says True
                {"observe_me": False},  # Session config says False - should win
            ]
        }

        result = get_effective_observe_me("sender", peers_with_configuration)

        assert result is False

    def test_session_config_none_falls_back_to_peer_config(self):
        """Test that session config with None observe_me falls back to peer config"""
        from src.deriver.enqueue import get_effective_observe_me

        peers_with_configuration = {
            "sender": [
                {"observe_me": False},  # Peer config says False
                {"observe_me": None},  # Session config is None - should fall back
            ]
        }

        result = get_effective_observe_me("sender", peers_with_configuration)

        assert result is False

    def test_mixed_configurations_with_active_status(self):
        """Test various configuration combinations with active status"""
        from src.deriver.enqueue import get_effective_observe_me

        # Test cases: (peer_config, session_config, expected_result)
        test_cases: list[tuple[dict[str, Any] | None, dict[str, Any] | None, bool]] = [
            # Default case - missing sender
            (None, None, True),
            # Empty configs
            ({}, {}, True),
            # Peer config only
            ({"observe_me": False}, {}, False),
            ({"observe_me": True}, {}, True),
            # Session config overrides
            ({"observe_me": True}, {"observe_me": False}, False),
            ({"observe_me": False}, {"observe_me": True}, True),
            # Session config None falls back to peer
            ({"observe_me": False}, {"observe_me": None}, False),
            ({"observe_me": True}, {"observe_me": None}, True),
        ]

        for i, (peer_config, session_config, expected) in enumerate(test_cases):
            if peer_config is None:
                # Test missing sender
                peers_with_configuration = {}
                observed = "missing_sender"
            else:
                peers_with_configuration = {
                    f"sender_{i}": [peer_config or {}, session_config or {}]
                }
                observed = f"sender_{i}"

            result = get_effective_observe_me(observed, peers_with_configuration)
            assert (
                result == expected
            ), f"Test case {i} failed: peer_config={peer_config}, session_config={session_config}, expected={expected}, got={result}"


@pytest.mark.asyncio
class TestAdvancedEnqueueEdgeCases:
    """Test advanced edge cases for the enqueue system with race conditions"""

    # Helper methods
    async def create_sample_payload(
        self,
        db_session: AsyncSession,
        workspace_name: str = "test_workspace",
        session_name: str | None = "test_session",
        peer_name: str = "test_peer",
        count: int = 1,
    ) -> list[dict[str, Any]]:
        """Create real messages in database and return payload with actual IDs"""
        # Get the current max sequence number for this session
        result = await db_session.execute(
            select(func.max(models.Message.seq_in_session)).where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
        )
        current_max_seq = result.scalar() or 0

        messages: list[models.Message] = []
        for i in range(count):
            message = models.Message(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=peer_name,
                content=f"Test message {i}",
                public_id=generate_nanoid(),
                seq_in_session=current_max_seq + i + 1,
                token_count=10,
                h_metadata={"test": f"value_{i}"},
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Return payload with real message IDs
        return [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": msg.id,
                "content": msg.content,
                "metadata": msg.h_metadata,
                "peer_name": peer_name,
                "created_at": msg.created_at,
            }
            for msg in messages
        ]

    async def count_queue_items(self, db_session: AsyncSession):
        """Helper to count queue items in database"""
        result = await db_session.execute(select(QueueItem))
        return len(result.scalars().all())

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_edge_case_all_peers_left_except_sender(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test edge case where all observer peers have left the session"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, sender_peer = sample_data

        # Create multiple observer peers
        observer1 = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        observer2 = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add_all([observer1, observer2])

        # Create session with all peers
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    sender_peer.name: schemas.SessionPeerConfig(observe_me=True),
                    observer1.name: schemas.SessionPeerConfig(observe_others=True),
                    observer2.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Mark all observers as having left

        for peer_name in [observer1.name, observer2.name]:
            session_peer_result = await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.session_name == test_session.name,
                    models.SessionPeer.peer_name == peer_name,
                    models.SessionPeer.workspace_name == test_workspace.name,
                )
            )
            session_peer = session_peer_result.scalar_one()
            session_peer.left_at = datetime.now(timezone.utc)
        await db_session.commit()

        # Create message payload
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=sender_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create only 1 queue item: representation for sender only
        assert final_count - initial_count == 1

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        assert len(queue_items) == 1
        assert queue_items[0].payload["observed"] == sender_peer.name
        assert queue_items[0].payload["observer"] == sender_peer.name
        assert queue_items[0].payload["task_type"] == "representation"

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_edge_case_sender_and_observer_both_left_different_times(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test race condition where both sender and observer left at different times"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, sender_peer = sample_data

        observer_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(observer_peer)

        # Create session
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    sender_peer.name: schemas.SessionPeerConfig(observe_me=True),
                    observer_peer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Mark both as having left (observer left first, then sender)

        base_time = datetime.now(timezone.utc)

        # Observer left first
        observer_session_peer_result = await db_session.execute(
            select(models.SessionPeer).where(
                models.SessionPeer.session_name == test_session.name,
                models.SessionPeer.peer_name == observer_peer.name,
                models.SessionPeer.workspace_name == test_workspace.name,
            )
        )
        observer_session_peer = observer_session_peer_result.scalar_one()
        observer_session_peer.left_at = base_time

        # Sender left later
        sender_session_peer_result = await db_session.execute(
            select(models.SessionPeer).where(
                models.SessionPeer.session_name == test_session.name,
                models.SessionPeer.peer_name == sender_peer.name,
                models.SessionPeer.workspace_name == test_workspace.name,
            )
        )
        sender_session_peer = sender_session_peer_result.scalar_one()
        sender_session_peer.left_at = base_time

        await db_session.commit()

        # Create message payload from sender who left
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=sender_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create only 1 queue item: representation for sender (using default config)
        # Observer should be skipped because they left the session
        assert final_count - initial_count == 1

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        assert len(queue_items) == 1
        assert queue_items[0].payload["observed"] == sender_peer.name
        assert queue_items[0].payload["observer"] == sender_peer.name
        assert queue_items[0].payload["task_type"] == "representation"

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.tracked_db")
    async def test_edge_case_message_from_never_joined_peer(
        self,
        mock_tracked_db: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test handling message from peer who was never in the session"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, existing_peer = sample_data

        observer_peer = models.Peer(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(observer_peer)

        # Create session with only observer peer
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={
                    observer_peer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        # Create message from peer who was NEVER in the session
        payload = await self.create_sample_payload(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=existing_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 2 queue items:
        # 1 for never_joined_peer (using default config)
        # 1 for observer (observe_others=True)
        assert final_count - initial_count == 2

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()

        expected_payloads = [
            {
                "observed": existing_peer.name,
                "observer": existing_peer.name,
                "task_type": "representation",
            },
            {
                "observed": existing_peer.name,
                "observer": observer_peer.name,
                "task_type": "representation",
            },
        ]
        actual_payloads = [
            {
                "observed": item.payload.get("observed"),
                "observer": item.payload.get("observer"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]

        assert len(actual_payloads) == len(expected_payloads)
        for expected in expected_payloads:
            assert expected in actual_payloads


@pytest.mark.asyncio
class TestGenerateQueueRecordsSeqInSession:
    """Unit tests for generate_queue_records function focusing on seq_in_session handling"""

    async def test_generate_queue_records_uses_seq_from_payload_not_crud(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """
        Test that generate_queue_records uses seq_in_session from payload
        instead of making a CRUD call to get_message_seq_in_session.
        """

        test_workspace, test_peer = sample_data

        # Create a test session
        test_session = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create a message payload with seq_in_session included
        message_payload = {
            "message_id": 12345,
            "peer_name": test_peer.name,
            "workspace_name": test_workspace.name,
            "session_name": test_session.name,
            "content": "Test message",
            "seq_in_session": 42,  # This should be used, not queried
            "created_at": datetime.now(timezone.utc),  # Required by create_payload
        }

        # Mock the CRUD function to track if it's called
        with patch("src.deriver.enqueue.crud.get_message_seq_in_session") as mock_crud:
            mock_db_session = AsyncMock()

            peers_config: dict[str, list[Any]] = {
                test_peer.name: [
                    {"observe_me": True},
                    {"observe_others": True},
                    True,
                ]
            }
            records = await generate_queue_records(
                db_session=mock_db_session,
                message=message_payload,
                peers_with_configuration=peers_config,
                session_id=test_session.id,
                deriver_disabled=False,
            )

            mock_crud.assert_not_called()

            assert len(records) > 0

            summary_records = [r for r in records if r["task_type"] == "summary"]
            for record in summary_records:
                assert record["payload"]["message_seq_in_session"] == 42

    async def test_generate_queue_records_falls_back_to_crud_when_seq_missing(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """
        Test that generate_queue_records falls back to CRUD call
        when seq_in_session is missing from payload.

        This is the fallback behavior for backward compatibility.
        """

        test_workspace, test_peer = sample_data

        # Create a test session
        test_session = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create a message payload WITHOUT seq_in_session
        message_payload = {
            "message_id": 12345,
            "peer_name": test_peer.name,
            "workspace_name": test_workspace.name,
            "session_name": test_session.name,
            "content": "Test message",
            "created_at": datetime.now(timezone.utc),
            # seq_in_session is MISSING
        }

        # Mock the CRUD function
        with patch("src.deriver.enqueue.crud.get_message_seq_in_session") as mock_crud:
            mock_crud.return_value = 50  # This should be used as fallback

            mock_db_session = AsyncMock()

            peers_config: dict[str, list[Any]] = {
                test_peer.name: [
                    {"observe_me": True},
                    {"observe_others": True},
                    True,
                ]
            }
            records = await generate_queue_records(
                db_session=mock_db_session,
                message=message_payload,
                peers_with_configuration=peers_config,
                session_id=test_session.id,
                deriver_disabled=False,
            )

            # The CRUD function SHOULD have been called as fallback
            mock_crud.assert_called_once_with(
                mock_db_session,
                workspace_name=test_workspace.name,
                session_name=test_session.name,
                message_id=12345,
            )

            # Verify that records were created with the fallback value
            summary_records = [r for r in records if r["task_type"] == "summary"]
            for record in summary_records:
                # Should use the value from CRUD fallback (50)
                assert record["payload"]["message_seq_in_session"] == 50

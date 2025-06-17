from typing import Optional
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select

from src import crud, models, schemas
from src.models import QueueItem
from src.routers.messages import enqueue


@pytest.mark.asyncio
class TestEnqueueFunction:
    """Test suite for the enqueue function's internal logic"""

    # Helper methods
    def create_sample_payload(
        self,
        workspace_name="test_workspace",
        session_name: Optional[str] = "test_session",
        peer_name="test_peer",
        count=1,
    ):
        """Create sample payload for testing"""
        return [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": f"msg_{i}",
                "content": f"Test message {i}",
                "metadata": {"test": f"value_{i}"},
                "peer_name": peer_name,
            }
            for i in range(count)
        ]

    async def count_queue_items(self, db_session):
        """Helper to count queue items in database"""
        result = await db_session.execute(select(QueueItem))
        return len(result.scalars().all())

    # Edge Cases & Input Validation Tests
    @pytest.mark.asyncio
    async def test_empty_payload_skips_enqueue(self, caplog):
        """Test that empty payload list is handled gracefully"""
        with caplog.at_level("DEBUG"):
            await enqueue([])

        assert "Empty payload list, skipping enqueue" in caplog.text

    @pytest.mark.asyncio
    async def test_malformed_payload_logs_error(self, caplog):
        """Test that malformed payload logs appropriate error"""
        malformed_payload = [{"incomplete": "data"}]

        with caplog.at_level("ERROR"):
            await enqueue(malformed_payload)  # Should not raise, but log error

        assert "Failed to enqueue messages: 'workspace_name'" in caplog.text

    # PEER MESSAGE TESTS

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_none_session_with_observe_me_true(
        self, mock_tracked_db, db_session, sample_data
    ):
        """Test peer-only processing when observe_me=True"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session
        test_workspace, test_peer = sample_data

        payload = self.create_sample_payload(
            workspace_name=test_workspace.name,
            session_name=None,
            peer_name=test_peer.name,
            count=1,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 1 queue item (representation)
        assert final_count - initial_count == 1

        # Verify queue items have correct structure
        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id.is_(None))
        )
        queue_items = result.scalars().all()

        for item in queue_items:
            payload_data = item.payload
            assert payload_data["task_type"] == "representation"
            assert payload_data["sender_name"] == test_peer.name
            assert payload_data["target_name"] == test_peer.name

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_none_session_with_observe_me_false(
        self, mock_tracked_db, db_session, sample_data, caplog
    ):
        """Test peer-only processing when observe_me=False"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        test_peer.feature_flags = {"observe_me": False}
        await db_session.commit()

        payload = self.create_sample_payload(
            workspace_name=test_workspace.name,
            session_name=None,
            peer_name=test_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        with caplog.at_level("INFO"):
            await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should not create any queue items
        assert final_count == initial_count
        assert (
            f"Peer {test_peer.name} has observe_me=False, skipping enqueue"
            in caplog.text
        )

    # SESSION MESSAGES
    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_session_with_deriver_disabled(
        self, mock_tracked_db, db_session, sample_data, caplog
    ):
        """Test that deriver disabled sessions skip enqueue"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        # Create session with deriver disabled
        test_session = models.Session(
            workspace_name=test_workspace.name,
            name=str(generate_nanoid()),
            h_metadata={"deriver_disabled": True},
        )
        db_session.add(test_session)
        await db_session.commit()

        payload = self.create_sample_payload(
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # we expect 1 queue item for session level summarization
        assert final_count == initial_count + 1

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_session_normal_processing_single_peer(
        self, mock_tracked_db, db_session, sample_data
    ):
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peer_names={test_peer.name: schemas.SessionPeerConfig(observe_me=True)},
            ),
            test_workspace.name,
        )
        await db_session.commit()

        payload = self.create_sample_payload(
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
    @patch("src.routers.messages.tracked_db")
    async def test_session_with_multiple_peers_none_observe_others(
        self, mock_tracked_db, db_session, sample_data
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
                peer_names={
                    test_peer1.name: schemas.SessionPeerConfig(),
                    test_peer2.name: schemas.SessionPeerConfig(),
                },
            ),
            test_workspace.name,
        )

        await db_session.commit()

        NUM_MESSAGES = 3
        payload = self.create_sample_payload(
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
        # For each message, we expect a representation and a summary queue item
        expected_payloads = []
        for _ in range(NUM_MESSAGES):
            expected_payloads.append(
                {
                    "sender_name": test_peer1.name,
                    "target_name": test_peer1.name,
                    "task_type": "representation",
                }
            )
        actual_payloads = [
            {
                "sender_name": item.payload.get("sender_name"),
                "target_name": item.payload.get("target_name"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]
        assert len(actual_payloads) == len(expected_payloads)

        # Assert that all expected payloads are present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_session_with_multiple_peers_all_observe_others(
        self, mock_tracked_db, db_session, sample_data
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
                peer_names={
                    test_peer1.name: schemas.SessionPeerConfig(),
                    test_peer2.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        NUM_MESSAGES = 3
        payload = self.create_sample_payload(
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
        # For each message, we expect a representation and a summary queue item
        expected_payloads = []
        for _ in range(NUM_MESSAGES):
            expected_payloads.append(
                {
                    "sender_name": test_peer1.name,
                    "target_name": test_peer1.name,
                    "task_type": "representation",
                }
            )
            expected_payloads.append(
                {
                    "sender_name": test_peer1.name,
                    "target_name": test_peer2.name,
                    "task_type": "representation",
                }
            )
        actual_payloads = [
            {
                "sender_name": item.payload.get("sender_name"),
                "target_name": item.payload.get("target_name"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]
        assert len(actual_payloads) == len(expected_payloads)

        # Assert that all expected payloads are present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_session_with_multiple_peers_some_observe_others(
        self, mock_tracked_db, db_session, sample_data
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
                peer_names={
                    test_peer1.name: schemas.SessionPeerConfig(observe_me=True),
                    observing_peer.name: schemas.SessionPeerConfig(observe_others=True),
                    unobserving_peer.name: schemas.SessionPeerConfig(),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        NUM_MESSAGES = 3
        payload = self.create_sample_payload(
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
        # For each message, we expect a representation and a summary queue item
        expected_payloads = []
        for _ in range(NUM_MESSAGES):
            expected_payloads.append(
                {
                    "sender_name": test_peer1.name,
                    "target_name": test_peer1.name,
                    "task_type": "representation",
                }
            )
            expected_payloads.append(
                {
                    "sender_name": test_peer1.name,
                    "target_name": observing_peer.name,
                    "task_type": "representation",
                }
            )
        actual_payloads = [
            {
                "sender_name": item.payload.get("sender_name"),
                "target_name": item.payload.get("target_name"),
                "task_type": item.payload.get("task_type"),
            }
            for item in queue_items
        ]
        assert len(actual_payloads) == len(expected_payloads)

        # Assert that all expected payloads are present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

        assert unobserving_peer.name not in [
            item.payload.get("target_name") for item in queue_items
        ]

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_session_peer_config_overrides_peer_config(
        self, mock_tracked_db, db_session, sample_data
    ):
        """Test that session peer config overrides peer config"""
        mock_tracked_db.return_value.__aenter__.return_value = db_session

        test_workspace, test_peer = sample_data

        # Set peer feature flags to observe_me=False
        test_peer.feature_flags = {"observe_me": True}

        # Create session
        test_session = await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peer_names={
                    test_peer.name: schemas.SessionPeerConfig(observe_me=False)
                },
            ),
            test_workspace.name,
        )

        await db_session.commit()

        payload = self.create_sample_payload(
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer.name,
        )

        initial_count = await self.count_queue_items(db_session)
        await enqueue(payload)
        final_count = await self.count_queue_items(db_session)

        # Should create 0 queue items
        assert final_count - initial_count == 0

        result = await db_session.execute(
            select(QueueItem).where(QueueItem.session_id == test_session.id)
        )
        queue_items = result.scalars().all()
        assert len(queue_items) == 0

    @pytest.mark.asyncio
    @patch("src.routers.messages.tracked_db")
    async def test_multi_sender_scenario(
        self, mock_tracked_db, db_session, sample_data
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
                peer_names={
                    test_peer1.name: schemas.SessionPeerConfig(),
                    additional_sender_peer.name: schemas.SessionPeerConfig(),
                    observer_peer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            ),
            test_workspace.name,
        )
        await db_session.commit()

        payload1 = self.create_sample_payload(
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_name=test_peer1.name,
        )

        payload2 = self.create_sample_payload(
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
        expected_payloads = []
        # 2 messages: one from test_peer1, one from additional_sender_peer
        for sender in [test_peer1.name, additional_sender_peer.name]:
            # representation for sender (self)
            expected_payloads.append(
                {
                    "task_type": "representation",
                    "sender_name": sender,
                    "target_name": sender,
                }
            )
            # representation for observer_peer (observe_others=True)
            expected_payloads.append(
                {
                    "task_type": "representation",
                    "sender_name": sender,
                    "target_name": observer_peer.name,
                }
            )

        # Extract actual payloads (task_type, sender_name, target_name) from queue_items
        actual_payloads = [
            {
                "task_type": item[0].payload.get("task_type"),
                "sender_name": item[0].payload.get("sender_name"),
                "target_name": item[0].payload.get("target_name"),
            }
            for item in queue_items
        ]

        assert len(actual_payloads) == len(expected_payloads)

        # For each expected payload, assert it is present in actual_payloads
        for expected in expected_payloads:
            assert expected in actual_payloads

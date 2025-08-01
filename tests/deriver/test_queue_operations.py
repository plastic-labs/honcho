from collections.abc import Callable
from typing import Any

import pytest

from src import models


@pytest.mark.asyncio
class TestQueueOperations:
    """Test suite for queue operations using the deriver conftest fixtures"""

    async def test_sample_queue_items_created(
        self,
        sample_queue_items: list[models.QueueItem],
    ):
        """Test that sample queue items are created correctly"""
        # Should have 9 items: 3 messages * 2 representations + 3 summaries
        assert len(sample_queue_items) == 9

        # Check that we have the right mix of task types
        representation_items = [
            item
            for item in sample_queue_items
            if item.payload.get("task_type") == "representation"
        ]
        summary_items = [
            item
            for item in sample_queue_items
            if item.payload.get("task_type") == "summary"
        ]

        assert len(representation_items) == 6
        assert len(summary_items) == 3

        # Check that all items are unprocessed
        assert all(not item.processed for item in sample_queue_items)

    async def test_sample_session_with_peers(
        self,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that sample session with peers is created correctly"""
        session, peers = sample_session_with_peers
        assert session is not None
        assert len(peers) == 3

        # Check that all peers have the same workspace
        assert all(peer.workspace_name == session.workspace_name for peer in peers)

    async def test_sample_messages(
        self,
        sample_messages: list[models.Message],
    ):
        """Test that sample messages are created correctly"""
        assert len(sample_messages) == 3

        # Check that all messages have content
        assert all(message.content for message in sample_messages)

        # Check that all messages have peer names
        assert all(message.peer_name for message in sample_messages)

    async def test_create_active_queue_session(
        self,
        create_active_queue_session: Callable[..., Any],
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that we can create active queue sessions"""
        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create an active queue session
        active_session = await create_active_queue_session(
            session_id=session.id,
            sender_name=peer1.name,
            target_name=peer2.name,
            task_type="representation",
        )

        assert active_session is not None
        assert active_session.session_id == session.id
        assert active_session.sender_name == peer1.name
        assert active_session.target_name == peer2.name
        assert active_session.task_type == "representation"

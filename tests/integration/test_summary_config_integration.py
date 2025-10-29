"""Integration tests for configurable summary thresholds."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.deriver.enqueue import handle_session
from src.models import Peer, Workspace


@pytest.mark.asyncio
class TestSummaryConfigIntegration:
    """Integration tests for summary configuration across the stack."""

    # Helper methods
    async def create_sample_payload(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        session_name: str,
        peer_name: str,
        count: int = 1,
        start_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        """Create real messages in database and return payload with actual IDs."""
        # Get the current max sequence number for this session
        result = await db_session.execute(
            select(func.max(models.Message.seq_in_session)).where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
        )
        current_max_seq = result.scalar() or 0

        # If start_seq is provided, use it; otherwise increment from current max
        starting_sequence = start_seq if start_seq is not None else current_max_seq + 1

        messages: list[models.Message] = []
        for i in range(count):
            message = models.Message(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=peer_name,
                content=f"Test message {starting_sequence + i}",
                public_id=generate_nanoid(),
                seq_in_session=starting_sequence + i,
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
                "seq_in_session": msg.seq_in_session,
            }
            for msg in messages
        ]

    async def create_messages_up_to_sequence(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        session_name: str,
        peer_name: str,
        target_seq: int,
    ) -> list[dict[str, Any]]:
        """Create messages from sequence 1 up to target_seq and return last message payload."""
        # Get current max sequence
        result = await db_session.execute(
            select(func.max(models.Message.seq_in_session)).where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
        )
        current_max_seq = result.scalar() or 0

        # Create messages from current_max_seq + 1 to target_seq
        messages_needed = target_seq - current_max_seq

        if messages_needed <= 0:
            # Already at or past target, just return payload for target_seq message
            result = await db_session.execute(
                select(models.Message).where(
                    models.Message.workspace_name == workspace_name,
                    models.Message.session_name == session_name,
                    models.Message.seq_in_session == target_seq,
                )
            )
            msg = result.scalar_one()
            return [
                {
                    "workspace_name": workspace_name,
                    "session_name": session_name,
                    "message_id": msg.id,
                    "content": msg.content,
                    "metadata": msg.h_metadata,
                    "peer_name": peer_name,
                    "created_at": msg.created_at,
                    "seq_in_session": msg.seq_in_session,
                }
            ]

        # Create all needed messages
        messages: list[models.Message] = []
        for seq in range(current_max_seq + 1, target_seq + 1):
            message = models.Message(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=peer_name,
                content=f"Test message {seq}",
                public_id=generate_nanoid(),
                seq_in_session=seq,
                token_count=10,
                h_metadata={"test": f"value_{seq}"},
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Return only the last message as payload
        last_msg = messages[-1]
        return [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": last_msg.id,
                "content": last_msg.content,
                "metadata": last_msg.h_metadata,
                "peer_name": peer_name,
                "created_at": last_msg.created_at,
                "seq_in_session": last_msg.seq_in_session,
            }
        ]

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.settings")
    async def test_workspace_level_config_affects_enqueue(
        self,
        mock_settings: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """Test that workspace-level summary configuration affects queue item creation."""
        mock_settings.SUMMARY.ENABLED = True
        workspace, peer = sample_data

        # Create a session
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )

        # Update workspace with custom summary thresholds
        workspace = await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                messages_per_short_summary=10,
                messages_per_long_summary=20,
            ),
        )
        await db_session.commit()

        # Create messages up to the 10th message (should trigger short summary)
        payload = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=10,
        )

        queue_records = await handle_session(
            db_session,
            payload,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        # Should create a summary queue item because message 10 hits the threshold
        summary_records = [r for r in queue_records if r["task_type"] == "summary"]
        assert len(summary_records) == 1

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.settings")
    async def test_session_level_config_overrides_workspace(
        self,
        mock_settings: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """Test that session-level configuration overrides workspace configuration."""
        mock_settings.SUMMARY.ENABLED = True
        workspace, peer = sample_data

        # Create a session
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )

        # Set workspace config
        await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                messages_per_short_summary=10,
                messages_per_long_summary=20,
            ),
        )

        # Set session config with different thresholds
        session = await crud.update_session(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            session=schemas.SessionUpdate(
                messages_per_short_summary=15,
                messages_per_long_summary=30,
            ),
        )
        await db_session.commit()

        # Message at position 10 should NOT trigger summary (session threshold is 15)
        payload_10 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=10,
        )

        queue_records_10 = await handle_session(
            db_session,
            payload_10,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        summary_records_10 = [
            r for r in queue_records_10 if r["task_type"] == "summary"
        ]
        assert len(summary_records_10) == 0  # Should not trigger at 10

        # Message at position 15 SHOULD trigger summary (session threshold is 15)
        payload_15 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=15,
        )

        queue_records_15 = await handle_session(
            db_session,
            payload_15,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        summary_records_15 = [
            r for r in queue_records_15 if r["task_type"] == "summary"
        ]
        assert len(summary_records_15) == 1  # Should trigger at 15

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.settings")
    async def test_global_defaults_when_no_config(
        self,
        mock_settings: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """Test that global defaults are used when no configuration is set."""
        from src.config import settings

        mock_settings.SUMMARY.ENABLED = True
        mock_settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY = (
            settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
        )
        workspace, peer = sample_data

        # Create a session without custom configuration
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )
        await db_session.commit()

        # Get global default threshold
        default_short = settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY

        # Create messages up to default threshold
        payload = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=default_short,
        )

        queue_records = await handle_session(
            db_session,
            payload,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        summary_records = [r for r in queue_records if r["task_type"] == "summary"]
        assert len(summary_records) == 1

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.settings")
    async def test_long_summary_threshold_respected(
        self,
        mock_settings: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """Test that long summary threshold is correctly applied."""
        mock_settings.SUMMARY.ENABLED = True
        workspace, peer = sample_data

        # Create a session
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )

        # Set custom thresholds
        await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                messages_per_short_summary=10,
                messages_per_long_summary=25,
            ),
        )
        await db_session.commit()

        # Message at position 24 should NOT trigger summary
        payload_24 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=24,
        )

        queue_records_24 = await handle_session(
            db_session,
            payload_24,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        summary_records_24 = [
            r for r in queue_records_24 if r["task_type"] == "summary"
        ]
        assert len(summary_records_24) == 0

        # Message at position 25 should trigger summary
        payload_25 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=25,
        )

        queue_records_25 = await handle_session(
            db_session,
            payload_25,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        summary_records_25 = [
            r for r in queue_records_25 if r["task_type"] == "summary"
        ]
        assert len(summary_records_25) == 1

    @pytest.mark.asyncio
    @patch("src.deriver.enqueue.settings")
    async def test_multiple_messages_trigger_summaries_correctly(
        self,
        mock_settings: AsyncMock,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """Test that summaries are triggered at correct intervals for multiple messages."""
        mock_settings.SUMMARY.ENABLED = True
        workspace, peer = sample_data

        # Create a session
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )

        # Set thresholds: short=10, long=30
        await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                messages_per_short_summary=10,
                messages_per_long_summary=30,
            ),
        )
        await db_session.commit()

        # Test message 10 (short summary)
        payload_10 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=10,
        )
        records_10 = await handle_session(
            db_session, payload_10, workspace.name, session.name
        )
        summary_count_10 = sum(1 for r in records_10 if r["task_type"] == "summary")
        assert summary_count_10 == 1

        # Test message 20 (short summary only, not long)
        payload_20 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=20,
        )
        records_20 = await handle_session(
            db_session, payload_20, workspace.name, session.name
        )
        summary_count_20 = sum(1 for r in records_20 if r["task_type"] == "summary")
        assert summary_count_20 == 1

        # Test message 30 (both short and long summary threshold met)
        payload_30 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=30,
        )
        records_30 = await handle_session(
            db_session, payload_30, workspace.name, session.name
        )
        summary_count_30 = sum(1 for r in records_30 if r["task_type"] == "summary")
        # Should create 1 summary item (both thresholds met but only one queue item created)
        assert summary_count_30 == 1

        # Test message 25 (no summary)
        payload_25 = await self.create_messages_up_to_sequence(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            target_seq=25,
        )
        records_25 = await handle_session(
            db_session, payload_25, workspace.name, session.name
        )
        summary_count_25 = sum(1 for r in records_25 if r["task_type"] == "summary")
        assert summary_count_25 == 0

    @pytest.mark.asyncio
    async def test_configuration_persists_across_updates(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ) -> None:
        """Test that configuration updates merge properly without losing data."""
        workspace, _peer = sample_data

        # Set initial configuration
        workspace = await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                configuration={
                    "messages_per_short_summary": 10,
                    "some_other_config": "value",
                }
            ),
        )
        await db_session.commit()

        # Update with only long summary threshold
        workspace = await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                configuration={
                    "messages_per_long_summary": 30,
                }
            ),
        )
        await db_session.commit()

        # Verify both configs are present
        assert workspace.configuration["messages_per_short_summary"] == 10
        assert workspace.configuration["messages_per_long_summary"] == 30
        assert workspace.configuration["some_other_config"] == "value"

    @pytest.mark.asyncio
    async def test_session_config_persists_across_updates(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ) -> None:
        """Test that session configuration updates merge properly."""
        workspace, _peer = sample_data

        # Create a session
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )

        # Set initial configuration
        session = await crud.update_session(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            session=schemas.SessionUpdate(
                configuration={
                    "messages_per_short_summary": 15,
                    "deriver_disabled": False,
                }
            ),
        )
        await db_session.commit()

        # Update with only long summary threshold
        session = await crud.update_session(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            session=schemas.SessionUpdate(
                configuration={
                    "messages_per_long_summary": 45,
                }
            ),
        )
        await db_session.commit()

        # Verify both configs are present
        assert session.configuration["messages_per_short_summary"] == 15
        assert session.configuration["messages_per_long_summary"] == 45
        assert session.configuration["deriver_disabled"] is False

    @pytest.mark.asyncio
    async def test_create_workspace_with_summary_config(
        self, db_session: AsyncSession
    ) -> None:
        """Test creating a workspace with summary configuration via API schema."""
        workspace_create = schemas.WorkspaceCreate(
            name="workspace-with-config",
            messages_per_short_summary=12,
            messages_per_long_summary=36,
        )

        workspace = await crud.get_or_create_workspace(
            db_session, workspace=workspace_create
        )
        await db_session.commit()

        assert workspace.configuration["messages_per_short_summary"] == 12
        assert workspace.configuration["messages_per_long_summary"] == 36

    @pytest.mark.asyncio
    async def test_create_session_with_summary_config(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ) -> None:
        """Test creating a session with summary configuration via API schema."""
        workspace, _peer = sample_data

        session_create = schemas.SessionCreate(
            name="session-with-config",
            messages_per_short_summary=18,
            messages_per_long_summary=54,
        )

        session = await crud.get_or_create_session(
            db_session, workspace_name=workspace.name, session=session_create
        )
        await db_session.commit()

        assert session.configuration["messages_per_short_summary"] == 18
        assert session.configuration["messages_per_long_summary"] == 54

    @pytest.mark.asyncio
    async def test_partial_config_update_at_workspace_level(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ) -> None:
        """Test updating only one threshold at workspace level."""
        workspace, _peer = sample_data

        # Update only short summary
        workspace = await crud.update_workspace(
            db_session,
            workspace_name=workspace.name,
            workspace=schemas.WorkspaceUpdate(
                messages_per_short_summary=14,
            ),
        )
        await db_session.commit()

        assert workspace.configuration["messages_per_short_summary"] == 14
        assert "messages_per_long_summary" not in workspace.configuration

    @pytest.mark.asyncio
    async def test_partial_config_update_at_session_level(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ) -> None:
        """Test updating only one threshold at session level."""
        workspace, _peer = sample_data

        # Create a session
        session = await crud.get_or_create_session(
            db_session,
            workspace_name=workspace.name,
            session=schemas.SessionCreate(name="test-session"),
        )

        # Update only long summary
        session = await crud.update_session(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            session=schemas.SessionUpdate(
                messages_per_long_summary=42,
            ),
        )
        await db_session.commit()

        assert session.configuration["messages_per_long_summary"] == 42
        assert "messages_per_short_summary" not in session.configuration

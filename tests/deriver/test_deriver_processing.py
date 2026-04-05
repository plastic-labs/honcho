import signal
from typing import Any

import pytest

from src import models
from src.utils.representation import Representation
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key


@pytest.mark.asyncio
class TestDeriverProcessing:
    """Test suite for deriver processing using the conftest fixtures"""

    async def test_work_unit_key_generation(
        self,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ):
        """Test that work unit keys are generated correctly"""

        session, peers = sample_session_with_peers
        peer1 = peers[0]

        # Create a payload for representation task
        # Note: observer is no longer part of the work_unit_key for representation tasks
        representation_payload = {
            "session_name": session.name,
            "observed": peer1.name,
            "task_type": "representation",
        }

        # Generate work unit key for representation
        work_unit_key = construct_work_unit_key(
            session.workspace_name, representation_payload
        )
        # Representation keys no longer include observer (deduplication change)
        expected_key = (
            f"representation:{session.workspace_name}:{session.name}:{peer1.name}"
        )
        assert work_unit_key == expected_key

        # Create a payload for summary task
        summary_payload = {
            "session_name": session.name,
            "task_type": "summary",
        }

        # Generate work unit key for summary
        summary_work_unit_key = construct_work_unit_key(
            session.workspace_name, summary_payload
        )
        expected_summary_key = (
            f"summary:{session.workspace_name}:{session.name}:None:None"
        )
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

    async def test_mock_representation_manager(
        self,
        mock_representation_manager: Any,  # AsyncMock object
    ):
        """Test that the representation manager is properly mocked"""
        assert mock_representation_manager is not None

        # Verify we can call the mocked methods
        await mock_representation_manager.save_representation(
            Representation(explicit=[], deductive=[])
        )

        # Verify the methods were called
        assert mock_representation_manager.save_representation.called  # type: ignore[attr-defined]


class TestBackwardsCompatibility:
    """Test backwards compatibility for queue items created before the deduplication change."""

    def test_parse_legacy_representation_work_unit_key(self):
        """Test that legacy 5-part representation work unit keys are parsed correctly.

        Before the deduplication change, representation keys had the format:
        representation:{workspace}:{session}:{observer}:{observed}

        After the change, the format is:
        representation:{workspace}:{session}:{observed}

        We need to support both for backwards compatibility with existing queue items.
        """
        legacy_key = (
            "representation:workspace_123:session_456:observer_peer:observed_peer"
        )
        parsed = parse_work_unit_key(legacy_key)

        assert parsed.task_type == "representation"
        assert parsed.workspace_name == "workspace_123"
        assert parsed.session_name == "session_456"
        assert parsed.observer == "observer_peer"
        assert parsed.observed == "observed_peer"

    def test_parse_new_representation_work_unit_key(self):
        """Test that new 4-part representation work unit keys are parsed correctly."""
        new_key = "representation:workspace_123:session_456:observed_peer"
        parsed = parse_work_unit_key(new_key)

        assert parsed.task_type == "representation"
        assert parsed.workspace_name == "workspace_123"
        assert parsed.session_name == "session_456"
        assert parsed.observer is None
        assert parsed.observed == "observed_peer"

    def test_parse_invalid_representation_work_unit_key_raises(self):
        """Test that invalid representation keys raise ValueError."""
        with pytest.raises(ValueError):
            parse_work_unit_key("representation:workspace:session")

        with pytest.raises(ValueError):
            parse_work_unit_key("representation:a:b:c:d:e")

    def test_legacy_payload_observer_converted_to_observers_list(self):
        """Test that legacy payloads with singular 'observer' are handled correctly."""
        legacy_payload: dict[str, Any] = {
            "observer": "peer_observer",
            "observed": "peer_observed",
            "task_type": "representation",
        }

        # This mirrors the logic in queue_manager.py process_work_unit
        observers = legacy_payload.get("observers")
        if observers is None:
            legacy_observer = legacy_payload.get("observer")
            observers = [legacy_observer] if legacy_observer else []

        assert observers == ["peer_observer"]

    def test_new_payload_observers_list_used_directly(self):
        """Test that new payloads with 'observers' list are used directly."""
        new_payload: dict[str, Any] = {
            "observers": ["peer1", "peer2"],
            "observed": "peer3",
            "task_type": "representation",
        }

        observers = new_payload.get("observers")
        if observers is None:
            legacy_observer = new_payload.get("observer")
            observers = [legacy_observer] if legacy_observer else []

        assert observers == ["peer1", "peer2"]

    def test_empty_payload_results_in_empty_observers_list(self):
        """Test that payloads with neither observer nor observers return empty list."""
        empty_payload: dict[str, Any] = {
            "observed": "peer_observed",
            "task_type": "representation",
        }

        observers = empty_payload.get("observers")
        if observers is None:
            legacy_observer = empty_payload.get("observer")
            observers = [legacy_observer] if legacy_observer else []

        assert observers == []

    # async def test_representation_batch_uses_earliest_cutoff(
    #     self,
    #     db_session: AsyncSession,
    #     sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    #     monkeypatch: pytest.MonkeyPatch,
    # ) -> None:
    #     """Ensure batching history cutoff uses the earliest payload in the batch."""
    #     captured_cutoffs: list[int] = []

    #     async def fake_get_session_context_formatted(*_args: Any, **kwargs: Any) -> str:
    #         captured_cutoffs.append(kwargs["cutoff"])
    #         return "formatted-history"

    #     # Mock only the function we need to inspect for the test assertion
    #     monkeypatch.setattr(
    #         "src.deriver.deriver.summarizer.get_session_context_formatted",
    #         fake_get_session_context_formatted,
    #     )

    #     # Provide a stub working representation so embedding lookups are skipped.
    #     monkeypatch.setattr(
    #         "src.crud.get_working_representation",
    #         AsyncMock(
    #             return_value=Representation(
    #                 explicit=[],
    #                 deductive=[],
    #             )
    #         ),
    #     )

    #     # Avoid executing the full reasoning pipeline; we only care about cutoff behavior.
    #     monkeypatch.setattr(
    #         "src.deriver.deriver.CertaintyReasoner.reason",
    #         AsyncMock(return_value=Representation(explicit=[], deductive=[])),
    #     )

    #     # Use the real session and workspace from fixtures
    #     session, peers = sample_session_with_peers
    #     alice = peers[0]

    #     # Create test messages with different IDs in the database
    #     now = datetime.now(timezone.utc)
    #     messages: list[models.Message] = []
    #     for i in range(8):
    #         message = models.Message(
    #             workspace_name=session.workspace_name,
    #             session_name=session.name,
    #             peer_name=alice.name,
    #             content=f"message {i}",
    #             seq_in_session=i + 1,
    #             token_count=10,
    #             created_at=now - timedelta(minutes=7 - i),
    #         )
    #         db_session.add(message)
    #         messages.append(message)

    #     await db_session.commit()

    #     # Refresh messages to get their IDs
    #     for message in messages:
    #         await db_session.refresh(message)

    #     await process_representation_tasks_batch(
    #         observer=alice.name,
    #         message_level_configuration=None,
    #         observed=alice.name,
    #         messages=messages,
    #     )

    #     # Verify that the earliest message ID was used as the cutoff
    #     assert captured_cutoffs == [messages[0].id]

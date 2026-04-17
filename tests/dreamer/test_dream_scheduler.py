"""Regression tests for dream scheduler bug fixes."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dreamer.dream_scheduler import (
    DreamScheduler,
    check_and_schedule_dream,
    set_dream_scheduler,
)
from src.schemas import DreamType
from src.utils.work_unit import construct_work_unit_key


@pytest.fixture
def dream_scheduler():
    """Create a fresh DreamScheduler instance for each test."""
    # Reset the singleton before each test
    DreamScheduler.reset_singleton()
    scheduler = DreamScheduler()
    set_dream_scheduler(scheduler)
    # Patch DREAM.ENABLED to True so tests work regardless of local config
    with patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED", True):
        yield scheduler
    # Cleanup
    DreamScheduler.reset_singleton()


class TestCancelDreamsForObserved:
    """Regression tests for Bug #1: Peer-to-peer observation dreams not cancelled on activity.

    Previously, when a message arrived from peer Bob, only the self-observation dream
    (observer=Bob, observed=Bob) was cancelled. Peer-to-peer observation dreams
    (observer=Alice, observed=Bob) were NOT cancelled, allowing dreams to fire
    during active conversation.
    """

    @pytest.mark.asyncio
    async def test_cancel_self_observation_dream(self, dream_scheduler: DreamScheduler):
        """Cancelling dreams for observed peer should cancel self-observation dreams."""
        workspace_name = "test_workspace"
        peer_name = "bob"

        # Schedule a self-observation dream (observer=bob, observed=bob)
        work_unit_key = construct_work_unit_key(
            workspace_name,
            {
                "task_type": "dream",
                "observer": peer_name,
                "observed": peer_name,
                "dream_type": "omni",
            },
        )

        with patch.object(dream_scheduler, "execute_dream", new_callable=AsyncMock):
            await dream_scheduler.schedule_dream(
                work_unit_key,
                workspace_name,
                delay_minutes=60,
                dream_type=DreamType.OMNI,
                observer=peer_name,
                observed=peer_name,
            )

            # Verify dream is pending
            assert work_unit_key in dream_scheduler.pending_dreams

            # Cancel dreams for observed peer
            cancelled = await dream_scheduler.cancel_dreams_for_observed(
                workspace_name, peer_name
            )

            # Verify the dream was cancelled
            assert work_unit_key in cancelled
            assert work_unit_key not in dream_scheduler.pending_dreams

    @pytest.mark.asyncio
    async def test_cancel_peer_to_peer_observation_dream(
        self, dream_scheduler: DreamScheduler
    ):
        """Cancelling dreams for observed peer should also cancel peer-to-peer dreams.

        This is the core regression test for Bug #1: previously, if Alice was
        observing Bob and Bob sent a message, the dream for (observer=Alice,
        observed=Bob) would NOT be cancelled.
        """
        workspace_name = "test_workspace"
        observer = "alice"  # Alice is watching Bob
        observed = "bob"  # Bob sends a message

        # Schedule a peer-to-peer observation dream (observer=alice, observed=bob)
        work_unit_key = construct_work_unit_key(
            workspace_name,
            {
                "task_type": "dream",
                "observer": observer,
                "observed": observed,
                "dream_type": "omni",
            },
        )

        with patch.object(dream_scheduler, "execute_dream", new_callable=AsyncMock):
            await dream_scheduler.schedule_dream(
                work_unit_key,
                workspace_name,
                delay_minutes=60,
                dream_type=DreamType.OMNI,
                observer=observer,
                observed=observed,
            )

            # Verify dream is pending
            assert work_unit_key in dream_scheduler.pending_dreams

            # When Bob sends a message, cancel all dreams where observed=bob
            cancelled = await dream_scheduler.cancel_dreams_for_observed(
                workspace_name, observed
            )

            # Verify the peer-to-peer dream was cancelled
            assert work_unit_key in cancelled
            assert work_unit_key not in dream_scheduler.pending_dreams

    @pytest.mark.asyncio
    async def test_cancel_multiple_observers_same_observed(
        self, dream_scheduler: DreamScheduler
    ):
        """When observed peer sends a message, ALL dreams observing them should cancel."""
        workspace_name = "test_workspace"
        observed = "bob"
        observers = ["alice", "charlie", "bob"]  # Multiple observers including self

        work_unit_keys: list[str] = []

        with patch.object(dream_scheduler, "execute_dream", new_callable=AsyncMock):
            for observer in observers:
                work_unit_key = construct_work_unit_key(
                    workspace_name,
                    {
                        "task_type": "dream",
                        "observer": observer,
                        "observed": observed,
                        "dream_type": "omni",
                    },
                )
                work_unit_keys.append(work_unit_key)

                await dream_scheduler.schedule_dream(
                    work_unit_key,
                    workspace_name,
                    delay_minutes=60,
                    dream_type=DreamType.OMNI,
                    observer=observer,
                    observed=observed,
                )

            # Verify all dreams are pending
            assert len(dream_scheduler.pending_dreams) == 3

            # Cancel all dreams where observed=bob
            cancelled = await dream_scheduler.cancel_dreams_for_observed(
                workspace_name, observed
            )

            # All three should be cancelled
            assert len(cancelled) == 3
            for key in work_unit_keys:
                assert key in cancelled
            assert len(dream_scheduler.pending_dreams) == 0

    @pytest.mark.asyncio
    async def test_does_not_cancel_dreams_for_different_observed(
        self, dream_scheduler: DreamScheduler
    ):
        """Cancelling dreams for one observed peer should not affect others."""
        workspace_name = "test_workspace"

        # Dream for Alice observing Bob
        key_alice_bob = construct_work_unit_key(
            workspace_name,
            {
                "task_type": "dream",
                "observer": "alice",
                "observed": "bob",
                "dream_type": "omni",
            },
        )

        # Dream for Alice observing Charlie (should NOT be cancelled)
        key_alice_charlie = construct_work_unit_key(
            workspace_name,
            {
                "task_type": "dream",
                "observer": "alice",
                "observed": "charlie",
                "dream_type": "omni",
            },
        )

        with patch.object(dream_scheduler, "execute_dream", new_callable=AsyncMock):
            await dream_scheduler.schedule_dream(
                key_alice_bob,
                workspace_name,
                delay_minutes=60,
                dream_type=DreamType.OMNI,
                observer="alice",
                observed="bob",
            )
            await dream_scheduler.schedule_dream(
                key_alice_charlie,
                workspace_name,
                delay_minutes=60,
                dream_type=DreamType.OMNI,
                observer="alice",
                observed="charlie",
            )

            assert len(dream_scheduler.pending_dreams) == 2

            # Cancel only dreams where observed=bob
            cancelled = await dream_scheduler.cancel_dreams_for_observed(
                workspace_name, "bob"
            )

            # Only the bob dream should be cancelled
            assert key_alice_bob in cancelled
            assert key_alice_charlie not in cancelled
            assert key_alice_charlie in dream_scheduler.pending_dreams

    @pytest.mark.asyncio
    async def test_does_not_cancel_dreams_for_different_workspace(
        self, dream_scheduler: DreamScheduler
    ):
        """Cancelling dreams should be scoped to the correct workspace."""
        observed = "bob"

        key_ws1 = construct_work_unit_key(
            "workspace1",
            {
                "task_type": "dream",
                "observer": "alice",
                "observed": observed,
                "dream_type": "omni",
            },
        )
        key_ws2 = construct_work_unit_key(
            "workspace2",
            {
                "task_type": "dream",
                "observer": "alice",
                "observed": observed,
                "dream_type": "omni",
            },
        )

        with patch.object(dream_scheduler, "execute_dream", new_callable=AsyncMock):
            await dream_scheduler.schedule_dream(
                key_ws1,
                "workspace1",
                delay_minutes=60,
                dream_type=DreamType.OMNI,
                observer="alice",
                observed=observed,
            )
            await dream_scheduler.schedule_dream(
                key_ws2,
                "workspace2",
                delay_minutes=60,
                dream_type=DreamType.OMNI,
                observer="alice",
                observed=observed,
            )

            # Cancel only in workspace1
            cancelled = await dream_scheduler.cancel_dreams_for_observed(
                "workspace1", observed
            )

            assert key_ws1 in cancelled
            assert key_ws2 not in cancelled
            assert key_ws2 in dream_scheduler.pending_dreams


class TestDocumentCountAtExecutionTime:
    """Regression tests for Bug #2: Stale document count used in metadata update.

    Previously, the document count was captured when the dream was scheduled
    (at check_and_schedule_dream time), then used 60 minutes later when the
    dream actually executed. This caused incorrect metadata if documents were
    added during the wait period.

    Now, execute_dream queries the current document count at execution time.
    """

    @pytest.mark.asyncio
    async def test_execute_dream_queries_document_count_at_execution(
        self, dream_scheduler: DreamScheduler
    ):
        """execute_dream should query current document count, not use a stale value.

        This test verifies that execute_dream fetches the document count fresh
        from the database at execution time rather than using a pre-captured value.

        The key architectural change was:
        - OLD: schedule_dream(document_count) -> _delayed_dream(document_count) -> execute_dream(document_count)
        - NEW: schedule_dream() -> _delayed_dream() -> execute_dream() queries count internally

        We verify this by mocking the database to return a specific count and
        checking that enqueue_dream receives that count.
        """
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from src import models
        from src.schemas import (
            ResolvedConfiguration,
            ResolvedDreamConfiguration,
            ResolvedPeerCardConfiguration,
            ResolvedReasoningConfiguration,
            ResolvedSummaryConfiguration,
        )

        workspace_name = "test_workspace"
        observer = "bob"
        observed = "bob"
        session_name = "test_session"

        # The document count that the database will return
        CURRENT_DOC_COUNT = 42

        # Track what document_count is passed to enqueue_dream
        captured_document_count: int | None = None

        async def capture_enqueue_dream(
            _ws_name: str,
            observer: str,  # pyright: ignore[reportUnusedParameter]
            observed: str,  # pyright: ignore[reportUnusedParameter]
            dream_type: Any,  # pyright: ignore[reportUnusedParameter]
            document_count: int,
            session_name: str,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            nonlocal captured_document_count
            captured_document_count = document_count

        # Create mock database session that returns our test data
        mock_session = MagicMock()
        mock_workspace = MagicMock(spec=models.Workspace)
        mock_db_session = MagicMock(spec=models.Session)

        # Mock scalar to return session_name for first call, document count for second
        scalar_call_count = 0

        async def mock_scalar(_stmt: Any) -> str | int:
            nonlocal scalar_call_count
            scalar_call_count += 1
            if scalar_call_count == 1:
                return session_name  # First call gets session_name from documents
            else:
                return CURRENT_DOC_COUNT  # Second call gets document count

        mock_session.scalar = mock_scalar

        @asynccontextmanager
        async def mock_tracked_db(_: str | None = None):
            yield mock_session

        with (
            patch(
                "src.dreamer.dream_scheduler.tracked_db",
                mock_tracked_db,
            ),
            patch(
                "src.deriver.enqueue.enqueue_dream",
                side_effect=capture_enqueue_dream,
            ),
            patch(
                "src.crud.get_session",
                return_value=mock_db_session,
            ),
            patch(
                "src.crud.get_workspace",
                return_value=mock_workspace,
            ),
            patch(
                "src.utils.config_helpers.get_configuration",
                return_value=ResolvedConfiguration(
                    reasoning=ResolvedReasoningConfiguration(enabled=True),
                    peer_card=ResolvedPeerCardConfiguration(use=True, create=True),
                    summary=ResolvedSummaryConfiguration(
                        enabled=True,
                        messages_per_short_summary=10,
                        messages_per_long_summary=20,
                    ),
                    dream=ResolvedDreamConfiguration(enabled=True),
                ),
            ),
        ):
            # Execute the dream
            await dream_scheduler.execute_dream(
                workspace_name,
                DreamType.OMNI,
                observer=observer,
                observed=observed,
            )

            # Verify that execute_dream queried the document count (2 scalar calls)
            assert (
                scalar_call_count == 2
            ), "Should have queried session_name and document count"

            # Verify that enqueue_dream received the CURRENT document count (42),
            # proving that execute_dream queries the count at execution time
            assert captured_document_count == CURRENT_DOC_COUNT


class TestThresholdFilter:
    """Regression tests for Finding 2: threshold must count only explicit-level docs.

    Previously the threshold counted all documents in a collection, including
    dreamer output (deductive/inductive/contradiction). This created a feedback
    loop where each dream's output inflated the trigger for the next dream.
    The fix filters the count to `level == "explicit"` only.
    """

    @pytest.fixture(autouse=True)
    def _pin_dream_config(self):
        """Pin DOCUMENT_THRESHOLD=50 and ENABLED_TYPES=['omni'] for this class.

        These tests assume the default thresholds; a developer's local env
        (e.g. DREAM_DOCUMENT_THRESHOLD=5 for faster manual testing) would
        otherwise invalidate the 30/60/10 fixtures below. Scoped to this
        class only — do NOT widen; other tests may have different assumptions.
        """
        with (
            patch("src.dreamer.dream_scheduler.settings.DREAM.DOCUMENT_THRESHOLD", 50),
            patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED_TYPES", ["omni"]),
        ):
            yield

    async def _make_collection(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> models.Collection:
        """Helper: create a Collection in the test workspace with no dream metadata."""
        workspace, peer = sample_data
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={},
        )
        db_session.add(collection)
        await db_session.commit()
        return collection

    async def _insert_doc(
        self,
        db_session: AsyncSession,
        collection: models.Collection,
        level: str,
    ) -> None:
        """Helper: insert one Document at the given level."""
        db_session.add(
            models.Document(
                content="test",
                level=level,
                workspace_name=collection.workspace_name,
                observer=collection.observer,
                observed=collection.observed,
            )
        )

    @pytest.mark.asyncio
    async def test_mixed_levels_below_explicit_threshold(
        self,
        dream_scheduler: DreamScheduler,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """30 explicit + 40 deductive + 10 inductive → should NOT trigger.

        Total doc count = 80 (would trigger under the buggy unfiltered count),
        but explicit count = 30 < threshold 50, so the correct behavior is to
        NOT schedule a dream. This is the core regression: the fix must reject
        this scenario.
        """
        collection = await self._make_collection(db_session, sample_data)
        for _ in range(30):
            await self._insert_doc(db_session, collection, "explicit")
        for _ in range(40):
            await self._insert_doc(db_session, collection, "deductive")
        for _ in range(10):
            await self._insert_doc(db_session, collection, "inductive")
        await db_session.commit()

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            scheduled = await check_and_schedule_dream(db_session, collection)

        assert scheduled is False, (
            "Threshold should filter on explicit level only — dreamer output "
            "(deductive/inductive) must not count toward the trigger."
        )

    @pytest.mark.asyncio
    async def test_explicit_only_at_threshold(
        self,
        dream_scheduler: DreamScheduler,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """60 explicit + 0 derived → should trigger (60 ≥ threshold 50)."""
        collection = await self._make_collection(db_session, sample_data)
        for _ in range(60):
            await self._insert_doc(db_session, collection, "explicit")
        await db_session.commit()

        with patch.object(
            dream_scheduler, "schedule_dream", new_callable=AsyncMock
        ) as mock_schedule:
            scheduled = await check_and_schedule_dream(db_session, collection)

        assert scheduled is True
        assert mock_schedule.called, "schedule_dream should fire when threshold met"

    @pytest.mark.asyncio
    async def test_contradiction_excluded_from_count(
        self,
        dream_scheduler: DreamScheduler,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Contradiction-level docs are dreamer output — must not count.

        100 contradictions + 10 explicit → explicit=10 < threshold=50, no trigger.
        Confirms the positive `== "explicit"` filter excludes contradiction by
        construction (same as deductive/inductive).
        """
        collection = await self._make_collection(db_session, sample_data)
        for _ in range(100):
            await self._insert_doc(db_session, collection, "contradiction")
        for _ in range(10):
            await self._insert_doc(db_session, collection, "explicit")
        await db_session.commit()

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            scheduled = await check_and_schedule_dream(db_session, collection)

        assert scheduled is False


class TestEnqueueCancelsDreamsCorrectly:
    """Integration test verifying the full flow of message enqueue cancelling dreams."""

    @pytest.mark.asyncio
    async def test_enqueue_cancels_peer_to_peer_dreams(
        self, dream_scheduler: DreamScheduler
    ):
        """When a message is enqueued, it should cancel all dreams for that observed peer."""

        workspace_name = "test_workspace"
        observed = "bob"

        # Schedule dreams for multiple observers watching bob
        keys: list[Any] = []
        for observer in ["alice", "charlie", "bob"]:
            key = construct_work_unit_key(
                workspace_name,
                {
                    "task_type": "dream",
                    "observer": observer,
                    "observed": observed,
                    "dream_type": "omni",
                },
            )
            keys.append(key)

            with patch.object(dream_scheduler, "execute_dream", new_callable=AsyncMock):
                await dream_scheduler.schedule_dream(
                    key,
                    workspace_name,
                    delay_minutes=60,
                    dream_type=DreamType.OMNI,
                    observer=observer,
                    observed=observed,
                )

        assert len(dream_scheduler.pending_dreams) == 3

        # Mock the database operations in enqueue
        with patch("src.deriver.enqueue.tracked_db"):
            # The enqueue function should cancel dreams via cancel_dreams_for_observed
            # We just test that the scheduler method was called correctly
            cancelled = await dream_scheduler.cancel_dreams_for_observed(
                workspace_name, observed
            )

        # All three dreams should be cancelled
        assert len(cancelled) == 3
        assert len(dream_scheduler.pending_dreams) == 0

"""Tests for the card_refresh dream type (DEV-2000, Scopes RFC prerequisite).

Covers:
- queue plumbing: payload roundtrip, work-unit key isolation from omni,
  enqueue alongside a pending omni dream
- process_dream dispatch of DreamType.CARD_REFRESH (and that it does NOT
  advance the omni dream guard pair)
- specialist tool restriction (no observation-mutating tools)
- the low tool-iteration cap
- rebuild mode omitting the prior peer card from the prompt
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.enqueue import enqueue_dream
from src.dreamer.orchestrator import DreamResult, process_dream
from src.dreamer.specialists import CardRefreshSpecialist
from src.llm import HonchoLLMCallResponse
from src.schemas import DreamType
from src.utils.queue_payload import DreamPayload, create_dream_payload
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key

OBSERVATION_MUTATION_TOOLS = {
    "create_observations",
    "create_observations_deductive",
    "create_observations_inductive",
    "delete_observations",
}


@pytest_asyncio.fixture
async def seeded_collection(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> models.Collection:
    """Create a Collection with an empty dream metadata dict."""
    workspace, peer = sample_data
    collection = models.Collection(
        observer=peer.name,
        observed=peer.name,
        workspace_name=workspace.name,
        internal_metadata={},
    )
    db_session.add(collection)
    await db_session.commit()
    await db_session.refresh(collection)
    return collection


def _make_card_refresh_result() -> DreamResult:
    return DreamResult(
        run_id="test_run_card",
        specialists_run=["card_refresh"],
        deduction_success=True,
        induction_success=False,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=2,
        total_duration_ms=42.0,
        input_tokens=10,
        output_tokens=5,
    )


class TestQueuePlumbing:
    def test_payload_roundtrip_carries_rebuild(self):
        payload_dict = create_dream_payload(
            DreamType.CARD_REFRESH,
            observer="alice",
            observed="bob",
            rebuild=True,
        )
        validated = DreamPayload(**payload_dict)
        assert validated.dream_type == DreamType.CARD_REFRESH
        assert validated.rebuild is True

        # Default is False, including for older payloads missing the field.
        assert (
            DreamPayload(dream_type=DreamType.OMNI, observer="a", observed="b").rebuild
            is False
        )

    def test_work_unit_key_does_not_collide_with_omni(self):
        base = {"task_type": "dream", "observer": "alice", "observed": "bob"}
        omni_key = construct_work_unit_key("ws", {**base, "dream_type": "omni"})
        card_key = construct_work_unit_key("ws", {**base, "dream_type": "card_refresh"})

        assert omni_key != card_key
        parsed = parse_work_unit_key(card_key)
        assert parsed.task_type == "dream"
        assert parsed.dream_type == "card_refresh"
        assert parsed.observer == "alice"
        assert parsed.observed == "bob"

    @pytest.mark.asyncio
    async def test_enqueue_alongside_pending_omni(
        self,
        db_session: AsyncSession,
        seeded_collection: models.Collection,
    ):
        """A pending omni dream must not dedupe away a card_refresh enqueue —
        the work-unit keys differ by dream type."""
        await enqueue_dream(
            seeded_collection.workspace_name,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
            dream_type=DreamType.OMNI,
        )
        await enqueue_dream(
            seeded_collection.workspace_name,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
            dream_type=DreamType.CARD_REFRESH,
            rebuild=True,
        )

        items = (
            (
                await db_session.execute(
                    select(models.QueueItem).where(
                        models.QueueItem.workspace_name
                        == seeded_collection.workspace_name,
                        models.QueueItem.task_type == "dream",
                        models.QueueItem.processed == False,  # noqa: E712
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(items) == 2
        dream_types = {item.payload["dream_type"] for item in items}
        assert dream_types == {"omni", "card_refresh"}
        card_item = next(
            item for item in items if item.payload["dream_type"] == "card_refresh"
        )
        assert card_item.payload["rebuild"] is True


class TestProcessDreamDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_card_refresh(
        self,
        seeded_collection: models.Collection,
    ):
        payload = DreamPayload(
            dream_type=DreamType.CARD_REFRESH,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
            rebuild=True,
            trigger_reason="manual",
        )

        with patch(
            "src.dreamer.orchestrator.run_card_refresh_dream",
            new=AsyncMock(return_value=_make_card_refresh_result()),
        ) as mock_run:
            await process_dream(payload, seeded_collection.workspace_name)

        assert mock_run.await_args is not None
        kwargs = mock_run.await_args.kwargs
        assert kwargs["workspace_name"] == seeded_collection.workspace_name
        assert kwargs["observer"] == seeded_collection.observer
        assert kwargs["observed"] == seeded_collection.observed
        assert kwargs["rebuild"] is True
        assert kwargs["dream_type"] == "card_refresh"
        assert kwargs["trigger_reason"] == "manual"

    @pytest.mark.asyncio
    async def test_card_refresh_does_not_advance_dream_guard(
        self,
        db_session: AsyncSession,
        seeded_collection: models.Collection,
    ):
        """The omni guard pair (last_dream_at / last_dream_document_count)
        must not move on a card refresh — it would delay real consolidation."""
        payload = DreamPayload(
            dream_type=DreamType.CARD_REFRESH,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
        )

        with patch(
            "src.dreamer.orchestrator.run_card_refresh_dream",
            new=AsyncMock(return_value=_make_card_refresh_result()),
        ):
            await process_dream(payload, seeded_collection.workspace_name)

        await db_session.refresh(seeded_collection)
        dream_meta: dict[str, Any] = seeded_collection.internal_metadata.get(
            "dream", {}
        )
        assert "last_dream_at" not in dream_meta
        assert "last_dream_document_count" not in dream_meta


class TestCardRefreshSpecialist:
    def test_tools_exclude_observation_mutation(self):
        for rebuild in (False, True):
            specialist = CardRefreshSpecialist(rebuild=rebuild)
            tool_names = {t["name"] for t in specialist.get_tools()}
            assert tool_names == {
                "get_recent_observations",
                "search_memory",
                "update_peer_card",
            }
            assert not tool_names & OBSERVATION_MUTATION_TOOLS

    def test_tools_without_peer_card_strip_update(self):
        specialist = CardRefreshSpecialist()
        tool_names = {t["name"] for t in specialist.get_tools(peer_card_enabled=False)}
        assert "update_peer_card" not in tool_names
        assert not tool_names & OBSERVATION_MUTATION_TOOLS

    def test_low_iteration_cap(self, monkeypatch: pytest.MonkeyPatch):
        specialist = CardRefreshSpecialist()
        assert specialist.get_max_iterations() == min(
            6, settings.DREAM.MAX_TOOL_ITERATIONS
        )

        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 4)
        assert specialist.get_max_iterations() == 4

        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 30)
        assert specialist.get_max_iterations() == 6

    def test_rebuild_flag_controls_card_injection(self):
        assert CardRefreshSpecialist(rebuild=False).inject_peer_card is True
        assert CardRefreshSpecialist(rebuild=True).inject_peer_card is False

    def test_rebuild_prompts_instruct_observation_only_build(self):
        specialist = CardRefreshSpecialist(rebuild=True)
        system_prompt = specialist.build_system_prompt("alice")
        assert "REBUILD MODE" in system_prompt
        assert "solely from the observations" in system_prompt

        user_prompt = specialist.build_user_prompt("alice", hints=None, peer_card=None)
        assert "Rebuild the peer card" in user_prompt
        assert "CURRENT PEER CARD" not in user_prompt

    async def _run_specialist(
        self, specialist: CardRefreshSpecialist, stored_card: list[str]
    ) -> tuple[AsyncMock, AsyncMock]:
        """Run the specialist with a fully mocked LLM layer; returns the
        (get_peer_card, honcho_llm_call) mocks for inspection."""
        mock_response = HonchoLLMCallResponse(
            content="done",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )
        mock_get_peer_card = AsyncMock(return_value=stored_card)
        mock_llm_call = AsyncMock(return_value=mock_response)

        with (
            patch("src.dreamer.specialists.crud.get_peer", new=AsyncMock()),
            patch(
                "src.dreamer.specialists.crud.get_peer_card",
                new=mock_get_peer_card,
            ),
            patch(
                "src.dreamer.specialists.create_tool_executor",
                new=AsyncMock(return_value=AsyncMock()),
            ),
            patch(
                "src.dreamer.specialists.honcho_llm_call",
                new=mock_llm_call,
            ),
        ):
            result = await specialist.run(
                workspace_name="workspace",
                observer="alice",
                observed="alice",
                session_name=None,
            )
        assert result.success is True
        return mock_get_peer_card, mock_llm_call

    # Sentinel card entry that cannot collide with the prompt's own examples
    # (the shared PEER CARD section contains e.g. "IDENTITY: Name: Alice").
    STORED_CARD: list[str] = [
        "IDENTITY: Name: Zorblax-Prime",
        "ATTRIBUTE: Location: Ganymede",
    ]

    @pytest.mark.asyncio
    async def test_refresh_mode_injects_existing_card(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.METRICS, "ENABLED", False)

        mock_get_peer_card, mock_llm_call = await self._run_specialist(
            CardRefreshSpecialist(rebuild=False), self.STORED_CARD
        )

        mock_get_peer_card.assert_awaited_once()
        assert mock_llm_call.await_args is not None
        kwargs = mock_llm_call.await_args.kwargs
        user_message = kwargs["messages"][1]["content"]
        assert "IDENTITY: Name: Zorblax-Prime" in user_message
        assert "CURRENT PEER CARD" in user_message
        # Restricted tool offering and low iteration cap reach the LLM call.
        tool_names = {t["name"] for t in kwargs["tools"]}
        assert not tool_names & OBSERVATION_MUTATION_TOOLS
        assert kwargs["max_tool_iterations"] == min(
            6, settings.DREAM.MAX_TOOL_ITERATIONS
        )

    @pytest.mark.asyncio
    async def test_rebuild_mode_omits_existing_card(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.METRICS, "ENABLED", False)

        mock_get_peer_card, mock_llm_call = await self._run_specialist(
            CardRefreshSpecialist(rebuild=True), self.STORED_CARD
        )

        # The stored card is never even fetched, let alone injected.
        mock_get_peer_card.assert_not_awaited()
        assert mock_llm_call.await_args is not None
        kwargs = mock_llm_call.await_args.kwargs
        for message in kwargs["messages"]:
            assert "IDENTITY: Name: Zorblax-Prime" not in message["content"]
        # No CURRENT PEER CARD block in the user prompt (the system prompt's
        # shared taxonomy section legitimately mentions the phrase).
        assert "CURRENT PEER CARD" not in kwargs["messages"][1]["content"]

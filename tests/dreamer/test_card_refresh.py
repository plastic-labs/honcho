"""Tests for the card_refresh dream type.

Covers:
- queue plumbing: payload roundtrip, work-unit key isolation from omni,
  enqueue alongside a pending omni dream
- process_dream dispatch of DreamType.CARD_REFRESH (and that it does NOT
  advance the omni dream guard pair)
- specialist tool restriction (no observation-mutating tools)
- the low tool-iteration cap, and the higher one for rebuilds
- rebuild mode omitting the prior peer card from the prompt
- the get_session_summaries tool: membership, ordering, paging, fallback
  from long to short summary, deleted sessions, and session-scoped refusal
"""

import datetime
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
from src.utils.agent_tools import (
    ToolContext,
    _handle_get_session_summaries,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.queue_payload import DreamPayload, create_dream_payload
from src.utils.summarizer import SUMMARIES_KEY, SummaryType
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
        read_tools = {
            "get_recent_observations",
            "search_memory",
            "get_session_summaries",
        }
        refresh = {t["name"] for t in CardRefreshSpecialist().get_tools()}
        assert refresh == read_tools | {"edit_peer_card", "update_peer_card"}
        # A rebuild cannot see the card it would be editing; it rewrites it.
        rebuild = {t["name"] for t in CardRefreshSpecialist(rebuild=True).get_tools()}
        assert rebuild == read_tools | {"update_peer_card"}
        assert not (refresh | rebuild) & OBSERVATION_MUTATION_TOOLS

    def test_tools_without_peer_card_strip_card_writes(self):
        for rebuild in (False, True):
            specialist = CardRefreshSpecialist(rebuild=rebuild)
            tool_names = {
                t["name"] for t in specialist.get_tools(peer_card_enabled=False)
            }
            assert not tool_names & {"update_peer_card", "edit_peer_card"}
            assert not tool_names & OBSERVATION_MUTATION_TOOLS

    def test_prompt_matches_the_card_tools(self):
        refresh = CardRefreshSpecialist().build_system_prompt("alice")
        assert "edit_peer_card" in refresh
        rebuild = CardRefreshSpecialist(rebuild=True).build_system_prompt("alice")
        assert "edit_peer_card" not in rebuild
        assert "complete deduplicated list" in rebuild

    def test_low_iteration_cap(self, monkeypatch: pytest.MonkeyPatch):
        specialist = CardRefreshSpecialist()
        assert specialist.get_max_iterations() == min(
            6, settings.DREAM.MAX_TOOL_ITERATIONS
        )

        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 4)
        assert specialist.get_max_iterations() == 4

        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 30)
        assert specialist.get_max_iterations() == 6

    def test_rebuild_iteration_cap(self, monkeypatch: pytest.MonkeyPatch):
        specialist = CardRefreshSpecialist(rebuild=True)
        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 50)
        assert specialist.get_max_iterations() == 20

        monkeypatch.setattr(settings.DREAM, "MAX_TOOL_ITERATIONS", 8)
        assert specialist.get_max_iterations() == 8

    def test_rebuild_flag_controls_card_injection(self):
        assert CardRefreshSpecialist(rebuild=False).inject_peer_card is True
        assert CardRefreshSpecialist(rebuild=True).inject_peer_card is False

    def test_rebuild_prompts_instruct_a_survey_before_writing(self):
        specialist = CardRefreshSpecialist(rebuild=True)
        system_prompt = specialist.build_system_prompt("alice")
        assert "REBUILD MODE" in system_prompt
        assert "only from evidence you find now" in system_prompt
        assert "get_session_summaries" in system_prompt
        assert "Keep it short" not in system_prompt

        user_prompt = specialist.build_user_prompt("alice", hints=None, peer_card=None)
        assert "Rebuild the peer card" in user_prompt
        assert "CURRENT PEER CARD" not in user_prompt

    def test_refresh_prompt_stays_short(self):
        system_prompt = CardRefreshSpecialist().build_system_prompt("alice")
        assert "REBUILD MODE" not in system_prompt
        assert "Keep it short" in system_prompt

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


async def _add_session(
    db_session: AsyncSession,
    workspace_name: str,
    name: str,
    members: list[str],
    summaries: dict[SummaryType, tuple[str, str]],
    *,
    deleted: bool = False,
) -> None:
    """Add a session with its members and stored summaries.

    `summaries` maps a summary type to its (content, created_at) pair.
    """
    db_session.add(
        models.Session(
            name=name,
            workspace_name=workspace_name,
            internal_metadata={
                SUMMARIES_KEY: {
                    summary_type.value: {
                        "content": content,
                        "message_id": 1,
                        "summary_type": summary_type.value,
                        "created_at": created_at,
                        "token_count": 1,
                        "message_public_id": "m",
                    }
                    for summary_type, (content, created_at) in summaries.items()
                }
            }
            if summaries
            else {},
            deleted_at=datetime.datetime.now(datetime.UTC) if deleted else None,
        )
    )
    await db_session.flush()
    for member in members:
        await db_session.execute(
            models.session_peers_table.insert().values(
                workspace_name=workspace_name,
                session_name=name,
                peer_name=member,
                joined_at=datetime.datetime.now(datetime.UTC),
                left_at=None,
            )
        )
    await db_session.flush()


class TestSessionSummariesTool:
    @pytest_asyncio.fixture
    async def summarized_sessions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> tuple[str, str, str]:
        workspace, peer = sample_data
        other = models.Peer(name="other-peer", workspace_name=workspace.name)
        db_session.add(other)
        await db_session.flush()

        long_and_short = {
            SummaryType.LONG: ("oldest long", "2026-11-01T00:00:00Z"),
            SummaryType.SHORT: ("oldest short", "2026-11-01T01:00:00Z"),
        }
        await _add_session(
            db_session, workspace.name, "oldest", [peer.name], long_and_short
        )
        await _add_session(
            db_session,
            workspace.name,
            "newest",
            [peer.name, other.name],
            {SummaryType.SHORT: ("newest short", "2026-11-03T00:00:00Z")},
        )
        await _add_session(
            db_session,
            workspace.name,
            "middle",
            [peer.name],
            {SummaryType.SHORT: ("middle short", "2026-11-02T00:00:00Z")},
        )
        await _add_session(
            db_session,
            workspace.name,
            "not-a-member",
            [other.name],
            {SummaryType.SHORT: ("not a member", "2026-11-04T00:00:00Z")},
        )
        await _add_session(
            db_session,
            workspace.name,
            "deleted",
            [peer.name],
            {SummaryType.SHORT: ("deleted", "2026-11-05T00:00:00Z")},
            deleted=True,
        )
        await _add_session(db_session, workspace.name, "unsummarized", [peer.name], {})
        await db_session.commit()
        return workspace.name, peer.name, other.name

    @staticmethod
    def _context(
        workspace_name: str,
        observer: str,
        observed: str,
        session_allowlist: list[str] | None = None,
    ) -> ToolContext:
        return ToolContext(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=None,
            current_messages=None,
            include_observation_ids=True,
            db_lock=AsyncMock(),
            session_allowlist=session_allowlist,
        )

    @pytest.mark.asyncio
    async def test_reads_the_peers_sessions_newest_first(
        self, summarized_sessions: tuple[str, str, str]
    ):
        workspace_name, peer, _ = summarized_sessions
        result = await _handle_get_session_summaries(
            self._context(workspace_name, peer, peer), {}
        )

        order = [
            result.index(f"## Session {name} ")
            for name in ("newest", "middle", "oldest")
        ]
        assert order == sorted(order)
        # A session with a long summary shows it; its short summary is left out.
        assert "oldest long" in result
        assert "oldest short" not in result
        for absent in ("not-a-member", "deleted", "unsummarized"):
            assert f"## Session {absent} " not in result

    @pytest.mark.asyncio
    async def test_pages_back_with_offset(
        self, summarized_sessions: tuple[str, str, str]
    ):
        workspace_name, peer, _ = summarized_sessions
        context = self._context(workspace_name, peer, peer)

        first = await _handle_get_session_summaries(context, {"limit": 2})
        assert "## Session newest " in first
        assert "## Session middle " in first
        assert "## Session oldest " not in first

        second = await _handle_get_session_summaries(context, {"limit": 2, "offset": 2})
        assert second.startswith("Summaries of sessions 3-3")
        assert "## Session oldest " in second

        past_the_end = await _handle_get_session_summaries(context, {"offset": 3})
        assert past_the_end.startswith("No session summaries found")

    @pytest.mark.asyncio
    async def test_cross_peer_view_needs_both_peers_in_the_session(
        self, summarized_sessions: tuple[str, str, str]
    ):
        workspace_name, peer, other = summarized_sessions
        result = await _handle_get_session_summaries(
            self._context(workspace_name, other, peer), {}
        )

        assert "## Session newest " in result
        assert "## Session middle " not in result
        assert "## Session not-a-member " not in result

    @pytest.mark.asyncio
    async def test_refused_for_session_scoped_queries(
        self, summarized_sessions: tuple[str, str, str]
    ):
        workspace_name, peer, _ = summarized_sessions
        result = await _handle_get_session_summaries(
            self._context(workspace_name, peer, peer, session_allowlist=["newest"]),
            {},
        )

        assert "unavailable for session-scoped queries" in result

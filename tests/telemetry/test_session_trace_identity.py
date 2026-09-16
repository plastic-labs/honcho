"""Canonical session identity from background producers through trace export."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from tenacity import wait_none

from src import crud, dependencies, models
from src.config import ConfiguredModelSettings, FallbackModelSettings, settings
from src.dependencies import tracked_db
from src.deriver import deriver
from src.deriver.consumer import process_item, process_representation_batch
from src.dreamer import orchestrator
from src.dreamer.specialists import (
    SPECIALISTS,
    CardRefreshSpecialist,
    SpecialistResult,
)
from src.llm import api, capture, executor, runtime
from src.llm.backend import CompletionResult, ProviderBackend
from src.telemetry import trace_exporter
from src.telemetry.events.base import BaseEvent
from src.telemetry.events.trace import LLMCallTracedEvent, TraceContentEvent
from src.utils import summarizer
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import SummaryPayload
from src.utils.representation import PromptRepresentation


@pytest.fixture(autouse=True)
def mock_llm_call_functions() -> None:
    """Override the suite's summary mocks to exercise the real producers."""


@pytest.fixture
def trace_events(monkeypatch: pytest.MonkeyPatch) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", True)
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PURPOSES", [])
    monkeypatch.setattr(trace_exporter, "emit_trace", events.append)
    capture.clear_exporters()
    capture.register_exporter(trace_exporter.TraceExporter())
    return events


@pytest.fixture(params=[False, True], ids=["primary", "retry-and-fallback"])
def backend(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    """Run real retry/planning/capture code with deterministic provider responses."""
    retry = request.param
    db_open = ContextVar("session_trace_db_open", default=False)
    tracked_db = dependencies.tracked_db

    @asynccontextmanager
    async def track_connection(
        operation: str, *, read_only: bool = False
    ) -> AsyncGenerator[AsyncSession]:
        token = db_open.set(True)
        try:
            async with tracked_db(operation, read_only=read_only) as db:
                yield db
        finally:
            db_open.reset(token)

    monkeypatch.setattr(summarizer, "tracked_db", track_connection)
    monkeypatch.setattr(deriver, "tracked_db", track_connection)

    async def complete(**kwargs: Any) -> CompletionResult:
        assert not db_open.get(), "DB connection held during inference"
        if retry and runtime.current_attempt.get() < 3:
            raise TimeoutError("synthetic provider timeout")
        content = (
            PromptRepresentation(explicit=[])
            if kwargs.get("response_format") is PromptRepresentation
            else "Synthetic conversation summary."
        )
        return CompletionResult(
            content=content, input_tokens=500, output_tokens=10, finish_reason="stop"
        )

    fake = AsyncMock(spec=ProviderBackend)
    fake.complete.side_effect = complete
    config = ConfiguredModelSettings(
        model="synthetic-primary",
        transport="openai",
        fallback=FallbackModelSettings(model="synthetic-fallback", transport="openai"),
    )
    monkeypatch.setattr(settings.DERIVER, "MODEL_CONFIG", config)
    monkeypatch.setattr(settings.SUMMARY, "MODEL_CONFIG", config)
    monkeypatch.setattr(runtime, "client_for_model_config", Mock(return_value=Mock()))
    monkeypatch.setattr(executor, "backend_for_provider", Mock(return_value=fake))
    monkeypatch.setattr(api, "wait_exponential", Mock(return_value=wait_none()))
    return fake


@pytest.fixture
async def conversations(
    db_session: AsyncSession,
) -> list[tuple[models.Session, list[models.Message]]]:
    """Two names in one workspace, plus the same name in another workspace."""
    conversations: list[tuple[models.Session, list[models.Message]]] = []
    for names in (("same-name", "other-name"), ("same-name",)):
        workspace = models.Workspace(name=generate_nanoid())
        db_session.add(workspace)
        await db_session.flush()
        peer = models.Peer(name="alice", workspace_name=workspace.name)
        db_session.add(peer)
        await db_session.flush()
        for name in names:
            session = models.Session(
                name=name,
                workspace_name=workspace.name,
                configuration={
                    "reasoning": {"enabled": True},
                    "summary": {
                        "enabled": True,
                        "messages_per_short_summary": 2,
                        "messages_per_long_summary": 3,
                    },
                },
            )
            db_session.add(session)
            await db_session.flush()
            messages = [
                models.Message(
                    workspace_name=workspace.name,
                    session_name=name,
                    peer_name=peer.name,
                    content=f"Synthetic message {seq}",
                    seq_in_session=seq,
                    token_count=5,
                )
                for seq in range(1, 7)
            ]
            db_session.add_all(messages)
            conversations.append((session, messages))
    await db_session.commit()
    return conversations


@pytest.mark.parametrize("card_refresh", [False, True])
@pytest.mark.parametrize("with_session", [False, True])
async def test_dream_resolves_session_id_before_db_cleanup(
    card_refresh: bool,
    with_session: bool,
    conversations: list[tuple[models.Session, list[models.Message]]],
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = conversations[0]
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    monkeypatch.setattr(settings.DREAM.SURPRISAL, "ENABLED", False)
    monkeypatch.setattr(
        dependencies,
        "SessionLocal",
        async_sessionmaker(db_engine, expire_on_commit=False),
    )
    monkeypatch.setattr(orchestrator, "tracked_db", tracked_db)
    run = AsyncMock(
        return_value=SpecialistResult(
            run_id="specialist",
            specialist_type="card_refresh" if card_refresh else "deduction",
            iterations=1,
            tool_calls_count=0,
            input_tokens=1,
            output_tokens=1,
            duration_ms=1,
            success=True,
            content="done",
        )
    )
    if card_refresh:
        monkeypatch.setattr(CardRefreshSpecialist, "run", run)
        dream = orchestrator.run_card_refresh_dream
    else:
        for specialist in SPECIALISTS.values():
            monkeypatch.setattr(specialist, "run", run)
        dream = orchestrator.run_dream

    result = await dream(
        workspace_name=session.workspace_name,
        observer="alice",
        observed="alice",
        session_name=session.name if with_session else None,
    )

    assert result is not None and result.deduction_success
    assert run.await_count == (1 if card_refresh else 2)
    for call in run.await_args_list:
        assert call.kwargs["session_id"] == (session.id if with_session else None)


@pytest.mark.parametrize("session_source", ["queue", "lookup", "configuration"])
async def test_background_session_identity(
    session_source: str,
    conversations: list[tuple[models.Session, list[models.Message]]],
    backend: AsyncMock,
    trace_events: list[BaseEvent],
) -> None:
    session_ids = {session.id for session, _ in conversations}
    assert len(session_ids) == 3

    for session, messages in conversations:
        start = len(trace_events)
        configuration = get_configuration(None, session)
        await process_representation_batch(
            messages=messages,
            message_level_configuration=(
                None if session_source == "configuration" else configuration
            ),
            observers=["alice"],
            observed="alice",
            queue_item_message_ids=[m.id for m in messages],
            session_id=session.id if session_source == "queue" else None,
            queue_item_ids=[101, 102],
        )

        # Short only, long only, then concurrent incremental short + long.
        for seq in (2, 3, 6):
            message = messages[seq - 1]
            await process_item(
                models.QueueItem(
                    id=200 + seq,
                    task_type="summary",
                    workspace_name=session.workspace_name,
                    session_id=session.id,
                    message_id=message.id,
                    payload=SummaryPayload(
                        session_name=session.name,
                        message_seq_in_session=seq,
                        message_public_id=message.public_id,
                        configuration=configuration,
                    ).model_dump(),
                )
            )

        traced = [
            event
            for event in trace_events[start:]
            if isinstance(event, LLMCallTracedEvent)
        ]
        assert {event.call_purpose for event in traced} == {
            "deriver.representation",
            "summary.short",
            "summary.long",
        }
        assert {event.session_id for event in traced} == {session.id}
        assert {event.workspace_name for event in traced} == {session.workspace_name}
        public_ids = {m.public_id for m in messages}
        for event in traced:
            assert event.source_message_ids
            assert set(event.source_message_ids) <= public_ids
            assert event.duration_ms is not None and event.duration_ms >= 0
            assert event.was_stream is False
            assert event.retry_attempts == 3
            assert event.is_final_attempt == (event.attempt == 3)
            assert event.effective_max_output_tokens
            assert event.outcome == (
                "error" if event.finish_reason == "error" else "success"
            )
            assert event.error_class == (
                "TimeoutError" if event.outcome == "error" else None
            )
            if event.call_purpose == "deriver.representation":
                assert set(event.source_message_ids) == public_ids
                assert event.queue_item_ids == [101, 102]
                assert event.observers == ["alice"]
                assert event.observed == "alice"
                assert event.agent_type == "deriver"
            else:
                assert len(event.queue_item_ids) == 1
                assert event.queue_item_ids[0] in {202, 203, 206}
                assert event.agent_type == "summarizer"
        assert all(event.trace_id == event.span_id for event in traced)
        assert all(event.trace_id != session.id for event in traced)
        successes = [event for event in traced if event.finish_reason == "stop"]
        assert len(successes) == 5
        assert sum(event.call_purpose == "summary.short" for event in successes) == 2
        assert sum(event.call_purpose == "summary.long" for event in successes) == 2
        if any(event.was_fallback for event in traced):
            assert len(traced) == 15
            for event in successes:
                attempts = [e for e in traced if e.trace_id == event.trace_id]
                assert [e.attempt for e in attempts] == [1, 2, 3]
                assert [e.was_fallback for e in attempts] == [False, False, True]
                assert event.model == "synthetic-fallback"
        else:
            assert len(traced) == 5

        content_hashes = {
            event.content_hash
            for event in trace_events[start:]
            if isinstance(event, TraceContentEvent)
        }
        for event in traced:
            assert event.input_message_refs
            assert set(event.input_message_refs) <= content_hashes
            if event.output_content_ref:
                assert event.output_content_ref in content_hashes
            restored = LLMCallTracedEvent.model_validate_json(event.model_dump_json())
            assert restored.session_id == session.id

    assert backend.complete.await_count == sum(
        isinstance(event, LLMCallTracedEvent) for event in trace_events
    )


async def test_queue_session_id_avoids_lookup(
    conversations: list[tuple[models.Session, list[models.Message]]],
    backend: AsyncMock,
    trace_events: list[BaseEvent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, messages = conversations[0]
    lookup = AsyncMock(side_effect=AssertionError("unnecessary session lookup"))
    monkeypatch.setattr(crud, "get_session", lookup)
    await process_representation_batch(
        messages,
        get_configuration(None, session),
        observers=["alice", "bob"],
        observed="alice",
        queue_item_message_ids=[messages[-1].id],
        session_id=session.id,
    )
    lookup.assert_not_awaited()
    assert backend.complete.await_count > 0
    assert {
        e.session_id for e in trace_events if isinstance(e, LLMCallTracedEvent)
    } == {session.id}
    for event in trace_events:
        if isinstance(event, LLMCallTracedEvent):
            assert event.observers == ["alice", "bob"]
            assert event.source_message_ids == [messages[-1].public_id]


async def test_sessionless_summaries_remain_nullable(
    backend: AsyncMock, trace_events: list[BaseEvent]
) -> None:
    await summarizer.create_short_summary("Synthetic conversation", input_tokens=5)
    await summarizer.create_long_summary("Synthetic conversation")
    traced = [e for e in trace_events if isinstance(e, LLMCallTracedEvent)]
    assert backend.complete.await_count == len(traced)
    assert {e.call_purpose for e in traced} == {"summary.short", "summary.long"}
    for event in traced:
        assert event.session_id is None
        assert (
            LLMCallTracedEvent.model_validate_json(event.model_dump_json()).session_id
            is None
        )
        legacy = event.model_dump()
        del legacy["session_id"]
        assert LLMCallTracedEvent.model_validate(legacy).session_id is None

"""Deriver and summarizer telemetry share one run id per batch and per summary."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import wait_none

import src.telemetry.events as telemetry_events
from src import models
from src.config import ConfiguredModelSettings, FallbackModelSettings, settings
from src.deriver import deriver
from src.deriver.consumer import process_item, process_representation_batch
from src.llm import api, capture, executor, runtime
from src.llm.backend import CompletionResult, ProviderBackend
from src.telemetry import trace_exporter
from src.telemetry.events.agent import AgentToolSummaryCreatedEvent
from src.telemetry.events.base import BaseEvent
from src.telemetry.events.llm import LLMCallCompletedEvent
from src.telemetry.events.representation import RepresentationCompletedEvent
from src.telemetry.events.trace import LLMCallTracedEvent
from src.utils import summarizer
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import SummaryPayload
from src.utils.representation import PromptRepresentation


@pytest.fixture(autouse=True)
def mock_llm_call_functions() -> None:
    """Override the suite's summary mocks to exercise the real producers."""


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[BaseEvent]:
    """Every event the deriver, summarizer and LLM executor emit."""
    emitted: list[BaseEvent] = []
    monkeypatch.setattr(telemetry_events, "emit", emitted.append)
    monkeypatch.setattr(deriver, "emit", emitted.append)
    monkeypatch.setattr(summarizer, "emit", emitted.append)
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", True)
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PURPOSES", [])
    monkeypatch.setattr(trace_exporter, "emit_trace", emitted.append)
    capture.clear_exporters()
    capture.register_exporter(trace_exporter.TraceExporter())
    return emitted


@pytest.fixture(params=[False, True], ids=["primary", "retry-and-fallback"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    retry = request.param

    async def complete(**kwargs: Any) -> CompletionResult:
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


@pytest.fixture
async def conversation(
    db_session: AsyncSession,
) -> tuple[models.Session, list[models.Message]]:
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()
    db_session.add(models.Peer(name="alice", workspace_name=workspace.name))
    await db_session.flush()
    session = models.Session(
        name=generate_nanoid(),
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
            session_name=session.name,
            peer_name="alice",
            content=f"Synthetic message {seq}",
            seq_in_session=seq,
            token_count=5,
        )
        for seq in range(1, 7)
    ]
    db_session.add_all(messages)
    await db_session.commit()
    return session, messages


def _of[E: BaseEvent](events: list[BaseEvent], kind: type[E]) -> list[E]:
    return [event for event in events if isinstance(event, kind)]


@pytest.mark.usefixtures("backend")
async def test_deriver_batch_events_share_one_run_id(
    conversation: tuple[models.Session, list[models.Message]],
    events: list[BaseEvent],
) -> None:
    session, messages = conversation
    configuration = get_configuration(None, session)
    run_ids: list[str] = []
    for batch in (messages[:3], messages[3:]):
        start = len(events)
        await process_representation_batch(
            messages=batch,
            message_level_configuration=configuration,
            observers=["alice"],
            observed="alice",
            queue_item_message_ids=[m.id for m in batch],
            session_id=session.id,
        )
        emitted = events[start:]

        (completed,) = _of(emitted, RepresentationCompletedEvent)
        assert completed.run_id
        calls = _of(emitted, LLMCallCompletedEvent)
        assert calls
        assert {call.run_id for call in calls} == {completed.run_id}
        traced = _of(emitted, LLMCallTracedEvent)
        assert len(traced) == len(calls)
        assert {event.trace_id for event in traced} == {completed.run_id}
        # The run id rides on the completion event only: the call's context
        # still has no agent run_id, so trace exports keep their shape.
        assert {event.run_id for event in traced} == {None}
        run_ids.append(completed.run_id)

    assert len(set(run_ids)) == 2


@pytest.mark.usefixtures("backend")
async def test_each_summary_has_its_own_run_id(
    conversation: tuple[models.Session, list[models.Message]],
    events: list[BaseEvent],
) -> None:
    session, messages = conversation
    configuration = get_configuration(None, session)
    # Message 6 is due for both a short and a long summary, run concurrently.
    message = messages[5]
    await process_item(
        models.QueueItem(
            id=206,
            task_type="summary",
            workspace_name=session.workspace_name,
            session_id=session.id,
            message_id=message.id,
            payload=SummaryPayload(
                session_name=session.name,
                message_seq_in_session=6,
                message_public_id=message.public_id,
                configuration=configuration,
            ).model_dump(),
        )
    )

    created = {e.summary_type: e for e in _of(events, AgentToolSummaryCreatedEvent)}
    assert set(created) == {"short", "long"}
    assert created["short"].run_id and created["long"].run_id
    assert created["short"].run_id != created["long"].run_id

    calls = _of(events, LLMCallCompletedEvent)
    traced = _of(events, LLMCallTracedEvent)
    for summary_type, purpose in (("short", "summary.short"), ("long", "summary.long")):
        run_id = created[summary_type].run_id
        assert {c.run_id for c in calls if c.call_purpose == purpose} == {run_id}
        assert {t.trace_id for t in traced if t.call_purpose == purpose} == {run_id}
    assert {event.run_id for event in traced} == {None}


@pytest.mark.usefixtures("backend")
async def test_standalone_summary_calls_still_get_a_run_id(
    events: list[BaseEvent],
) -> None:
    await summarizer.create_short_summary("Synthetic conversation", input_tokens=5)
    await summarizer.create_long_summary("Synthetic conversation")
    calls = _of(events, LLMCallCompletedEvent)
    traced = _of(events, LLMCallTracedEvent)
    by_purpose = {
        purpose: {c.run_id for c in calls if c.call_purpose == purpose}
        for purpose in ("summary.short", "summary.long")
    }
    (short_run_id,) = by_purpose["summary.short"]
    (long_run_id,) = by_purpose["summary.long"]
    assert short_run_id and long_run_id and short_run_id != long_run_id
    assert {t.trace_id for t in traced} == {short_run_id, long_run_id}

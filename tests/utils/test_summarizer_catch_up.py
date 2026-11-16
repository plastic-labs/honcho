"""Auto workspaces summarize queued history once, catching up in chunks.

The ingest enqueuer queues only the last summaries of each batch of history, so
a summary in an auto workspace can be far behind its message. It reads the
whole gap, oldest first, in chunks, instead of only its interval.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import UpstreamLLMError
from src.llm import HonchoLLMCallResponse
from src.models import Peer, Workspace
from src.schemas.configuration import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils import summarizer
from src.utils.summarizer import (
    Summary,
    SummaryCatchUpIncomplete,
    SummaryType,
    _create_and_save_summary,  # pyright: ignore[reportPrivateUsage]
    _save_summary,  # pyright: ignore[reportPrivateUsage]
    create_long_summary,
    create_short_summary,
    summarize_if_needed,
)


def _configuration() -> ResolvedConfiguration:
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
        summary=ResolvedSummaryConfiguration(
            enabled=True, messages_per_short_summary=20, messages_per_long_summary=60
        ),
        dream=ResolvedDreamConfiguration(enabled=False),
    )


async def _session(
    db: AsyncSession, workspace: Workspace, peer: Peer, count: int
) -> tuple[models.Session, list[models.Message]]:
    session = (
        await crud.get_or_create_session(
            db,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={peer.name: schemas.SessionPeerConfig()},
            ),
            workspace.name,
        )
    ).resource
    messages = [
        models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            content=f"message-{i}",
            public_id=generate_nanoid(),
            seq_in_session=i,
            token_count=5,
        )
        for i in range(1, count + 1)
    ]
    db.add_all(messages)
    await db.commit()
    for message in messages:
        await db.refresh(message)
    return session, messages


class _Model:
    """A fake summary model: numbered summaries, every prompt kept."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        self.prompts.append(kwargs["prompt"])
        return HonchoLLMCallResponse(
            content=f"summary-{len(self.prompts)}",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )


def _model() -> tuple[_Model, Any]:
    model = _Model()
    # The conftest replaces the summary builders; restore the real ones.
    return model, (
        patch("src.utils.summarizer.create_short_summary", new=create_short_summary),
        patch("src.utils.summarizer.create_long_summary", new=create_long_summary),
        patch("src.utils.summarizer.honcho_llm_call", new=model),
    )


async def _saved(db: AsyncSession, session: models.Session) -> dict[str, Any]:
    session_id = session.id
    db.expire_all()
    metadata = await db.scalar(
        select(models.Session.internal_metadata).where(models.Session.id == session_id)
    )
    assert metadata is not None
    return metadata.get("summaries", {})


async def _summarize(
    session: models.Session,
    message: models.Message,
    summary_type: SummaryType,
    *,
    catch_up: bool,
) -> _Model:
    model, patches = _model()
    with patches[0], patches[1], patches[2]:
        await _create_and_save_summary(
            workspace_name=session.workspace_name,
            session_name=session.name,
            message_id=message.id,
            message_seq_in_session=message.seq_in_session,
            message_public_id=message.public_id,
            summary_type=summary_type,
            configuration=_configuration(),
            catch_up=catch_up,
        )
    return model


def _named(prompt: str, messages: list[models.Message]) -> list[int]:
    return [m.seq_in_session for m in messages if f"{m.content}\n" in prompt + "\n"]


async def _save(
    db: AsyncSession,
    session: models.Session,
    message: models.Message,
    summary_type: SummaryType,
    content: str,
) -> None:
    """Save and commit a summary, as the summarizer's own save does."""
    write = await _save_summary(
        db,
        Summary(
            content=content,
            message_id=message.id,
            summary_type=summary_type.value,
            created_at="2026-11-13T00:00:00Z",
            token_count=4,
            message_public_id=message.public_id,
        ),
        session.workspace_name,
        session.name,
    )
    await db.commit()
    await write.post_commit()


@pytest.mark.asyncio
async def test_catch_up_reads_the_whole_gap_in_chunks(
    db_session: AsyncSession,
    auto_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 45)
    # Room for about 15 short messages per chunk.
    monkeypatch.setattr(summarizer, "_CATCH_UP_CHUNK_TOKENS", 150)

    model = await _summarize(session, messages[39], SummaryType.SHORT, catch_up=True)

    assert len(model.prompts) >= 3
    covered = [_named(prompt, messages) for prompt in model.prompts]
    # Every message up to the 40th, each once, in order; nothing after it.
    assert [seq for chunk in covered for seq in chunk] == list(range(1, 41))
    # Each chunk builds on the summary of the one before.
    for n, prompt in enumerate(model.prompts[1:], start=1):
        assert f"summary-{n}" in prompt
    fortieth = messages[39].id
    saved = (await _saved(db_session, session))[SummaryType.SHORT.value]
    assert saved["message_id"] == fortieth
    assert saved["content"] == f"summary-{len(model.prompts)}"


@pytest.mark.asyncio
async def test_without_catch_up_a_summary_reads_its_interval(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    workspace, peer = sample_data
    session, messages = await _session(db_session, workspace, peer, 45)

    model = await _summarize(session, messages[39], SummaryType.SHORT, catch_up=False)

    [prompt] = model.prompts
    assert _named(prompt, messages) == list(range(21, 41))


@pytest.mark.asyncio
async def test_a_short_summary_catches_up_from_the_long_one(
    db_session: AsyncSession, auto_data: tuple[Workspace, Peer]
) -> None:
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 60)
    await _save(db_session, session, messages[39], SummaryType.LONG, "the-long-summary")

    model = await _summarize(session, messages[59], SummaryType.SHORT, catch_up=True)

    [prompt] = model.prompts
    assert "the-long-summary" in prompt
    assert _named(prompt, messages) == list(range(41, 61))


@pytest.mark.asyncio
async def test_a_gap_within_the_interval_reads_as_before(
    db_session: AsyncSession, auto_data: tuple[Workspace, Peer]
) -> None:
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 40)
    await _save(db_session, session, messages[19], SummaryType.SHORT, "the-last-short")

    model = await _summarize(session, messages[39], SummaryType.SHORT, catch_up=True)

    [prompt] = model.prompts
    assert "the-last-short" in prompt
    assert _named(prompt, messages) == list(range(21, 41))


@pytest.mark.asyncio
async def test_catch_up_resumes_after_the_last_saved_chunk(
    db_session: AsyncSession, auto_data: tuple[Workspace, Peer]
) -> None:
    """A retry continues from its checkpoint, not from the interval's start."""
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 40)
    await _save(db_session, session, messages[29], SummaryType.SHORT, "checkpoint")

    model = await _summarize(session, messages[39], SummaryType.SHORT, catch_up=True)

    [prompt] = model.prompts
    assert "checkpoint" in prompt
    assert _named(prompt, messages) == list(range(31, 41))


@pytest.mark.asyncio
async def test_a_short_summary_starts_right_after_the_long_one(
    db_session: AsyncSession, auto_data: tuple[Workspace, Peer]
) -> None:
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 60)
    await _save(db_session, session, messages[49], SummaryType.LONG, "long-to-50")

    model = await _summarize(session, messages[59], SummaryType.SHORT, catch_up=True)

    [prompt] = model.prompts
    assert "long-to-50" in prompt
    assert _named(prompt, messages) == list(range(51, 61))


@pytest.mark.asyncio
async def test_catch_up_reads_past_a_long_run_of_deleted_messages(
    db_session: AsyncSession, auto_data: tuple[Workspace, Peer]
) -> None:
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 2_100)
    # Everything from 2 to 2,080 is gone: more sequence numbers than one read.
    gone = [m.id for m in messages[1:2_080]]
    await db_session.execute(delete(models.Message).where(models.Message.id.in_(gone)))
    await db_session.commit()

    model = await _summarize(session, messages[2_099], SummaryType.SHORT, catch_up=True)

    [prompt] = model.prompts
    survivors = [messages[0], *messages[2_080:]]
    assert _named(prompt, survivors) == [1, *range(2_081, 2_101)]


@pytest.mark.asyncio
async def test_one_long_message_is_read_whole(
    db_session: AsyncSession,
    auto_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message over the chunk size but within the input budget isn't clipped."""
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 25)
    messages[0].content = "message-1 " + "word " * 400
    await db_session.commit()
    monkeypatch.setattr(summarizer, "_CATCH_UP_CHUNK_TOKENS", 150)

    model = await _summarize(session, messages[24], SummaryType.SHORT, catch_up=True)

    assert "word " * 400 in model.prompts[0]
    assert "[truncated]" not in model.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [TimeoutError(), UpstreamLLMError("provider 503"), RuntimeError("provider down")],
    ids=["timeout", "upstream", "other"],
)
async def test_a_failed_call_is_raised_and_keeps_the_checkpoint(
    db_session: AsyncSession,
    auto_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """Raised as is, never skipped: the queue reports it, and a retry or the
    next summary resumes after the last saved chunk."""
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 45)
    monkeypatch.setattr(summarizer, "_CATCH_UP_CHUNK_TOKENS", 150)
    model = _Model()

    async def then_fail(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        if model.prompts:
            raise error
        return await model(**kwargs)

    with (
        patch("src.utils.summarizer.create_short_summary", new=create_short_summary),
        patch("src.utils.summarizer.honcho_llm_call", new=then_fail),
        pytest.raises(type(error)),
    ):
        await _create_and_save_summary(
            workspace_name=session.workspace_name,
            session_name=session.name,
            message_id=messages[39].id,
            message_seq_in_session=40,
            message_public_id=messages[39].public_id,
            summary_type=SummaryType.SHORT,
            configuration=_configuration(),
            catch_up=True,
        )

    [first] = model.prompts
    expected = messages[_named(first, messages)[-1] - 1].id
    saved = (await _saved(db_session, session))[SummaryType.SHORT.value]
    assert saved["content"] == "summary-1"
    assert saved["message_id"] == expected


@pytest.mark.asyncio
async def test_a_chunk_with_a_bad_answer_is_skipped_not_blocking(
    db_session: AsyncSession,
    auto_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty answer skips its chunk; the rest is summarized, then reported."""
    workspace, peer = auto_data
    session, messages = await _session(db_session, workspace, peer, 45)
    monkeypatch.setattr(summarizer, "_CATCH_UP_CHUNK_TOKENS", 150)
    fortieth = messages[39].id
    model = _Model()
    calls = 0

    async def empty_second(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            return HonchoLLMCallResponse(
                content="", input_tokens=10, output_tokens=0, finish_reasons=["stop"]
            )
        return await model(**kwargs)

    with (
        patch("src.utils.summarizer.create_short_summary", new=create_short_summary),
        patch("src.utils.summarizer.honcho_llm_call", new=empty_second),
        pytest.raises(SummaryCatchUpIncomplete, match="skipped messages"),
    ):
        await _create_and_save_summary(
            workspace_name=session.workspace_name,
            session_name=session.name,
            message_id=fortieth,
            message_seq_in_session=40,
            message_public_id=messages[39].public_id,
            summary_type=SummaryType.SHORT,
            configuration=_configuration(),
            catch_up=True,
        )

    assert calls >= 3
    saved = (await _saved(db_session, session))[SummaryType.SHORT.value]
    # Past the skipped chunk, all the way to the target.
    assert saved["message_id"] == fortieth


@pytest.mark.asyncio
@pytest.mark.parametrize("auto", [True, False])
async def test_only_auto_workspaces_catch_up_long_then_short(
    db_session: AsyncSession,
    auto_data: tuple[Workspace, Peer],
    sample_data: tuple[Workspace, Peer],
    auto: bool,
) -> None:
    """Catching up, short waits for long; otherwise both run at once."""
    workspace, peer = auto_data if auto else sample_data
    if not auto:
        workspace.auto_enabled_at = None
        await db_session.commit()
    session, messages = await _session(db_session, workspace, peer, 60)
    events: list[str] = []
    both_started = asyncio.Event()
    started: set[SummaryType] = set()

    async def record(*_args: Any, **kwargs: Any) -> None:
        summary_type: SummaryType = kwargs["summary_type"]
        assert kwargs["catch_up"] is auto
        events.append(f"start {summary_type.name}")
        started.add(summary_type)
        if len(started) == 2:
            both_started.set()
        if not auto:
            # Only returns once the other has started too: they run at once.
            await asyncio.wait_for(both_started.wait(), timeout=5)
        await asyncio.sleep(0)
        events.append(f"end {summary_type.name}")

    with patch(
        "src.utils.summarizer._create_and_save_summary",
        new=AsyncMock(side_effect=record),
    ):
        await summarize_if_needed(
            workspace.name,
            session.name,
            messages[59].id,
            60,
            messages[59].public_id,
            _configuration(),
        )

    if auto:
        assert events == ["start LONG", "end LONG", "start SHORT", "end SHORT"]
    else:
        assert events[:2] in (
            ["start LONG", "start SHORT"],
            ["start SHORT", "start LONG"],
        )

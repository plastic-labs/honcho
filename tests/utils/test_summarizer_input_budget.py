"""The summarizer keeps its prompt under SUMMARY_MAX_INPUT_TOKENS (DEV-3043).

The summary window is chosen by message count, so long messages could make the
prompt outgrow the model's context window. The oldest messages are dropped to
fit, and a single oversized message is clipped.
"""

import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.llm import HonchoLLMCallResponse
from src.models import Peer, Workspace
from src.schemas.configuration import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils.summarizer import (
    SUMMARY_TARGET_RATIO,
    Summary,
    SummaryType,
    _create_and_save_summary,  # pyright: ignore[reportPrivateUsage]
    _fit_messages_to_budget,  # pyright: ignore[reportPrivateUsage]
    _format_messages,  # pyright: ignore[reportPrivateUsage]
    _save_summary,  # pyright: ignore[reportPrivateUsage]
    create_long_summary,
    create_short_summary,
)
from src.utils.tokens import estimate_tokens, tokenizer

# About one token per repetition under o200k.
_WORD = " alpha"


def _actual_tokens(text: str) -> int:
    """Exact o200k count, counting control-token look-alikes as plain text.

    `estimate_tokens` falls back to a length estimate on such text, which would
    hide an over-budget prompt.
    """
    return len(tokenizer.encode(text, disallowed_special=()))


def _sent_tokens(text: str) -> int:
    """Tokens of `text` as it appears in a prompt, where tabs are expanded."""
    return _actual_tokens(text.expandtabs())


def _message(content: str, peer_name: str = "alice") -> models.Message:
    """An unsaved message; fitting only reads `peer_name` and `content`."""
    return models.Message(peer_name=peer_name, content=content)


def _text_of_tokens(count: int) -> str:
    """Text of roughly `count` tokens."""
    return (_WORD * count).strip()


# --- fitting the window ------------------------------------------------------


def test_window_under_budget_is_unchanged() -> None:
    """A window that fits keeps every message and formats as before."""
    messages = [_message("hello"), _message("hi there", peer_name="bob")]

    kept, text, clipped = _fit_messages_to_budget(messages, budget_tokens=1000)

    assert kept == messages
    assert text == _format_messages(messages)
    assert not clipped


def test_oldest_messages_are_dropped_first() -> None:
    """Over budget, the newest messages are kept and the oldest dropped."""
    messages = [_message(_text_of_tokens(100)) for _ in range(10)]

    kept, text, clipped = _fit_messages_to_budget(messages, budget_tokens=350)

    assert 0 < len(kept) < len(messages)
    assert kept == messages[-len(kept) :]
    assert text == _format_messages(kept)
    assert _sent_tokens(text) <= 350
    assert not clipped


def test_tabs_are_counted_as_the_prompt_expands_them() -> None:
    """The prompt builders expand tabs, so a tab-heavy line costs more than its
    raw text suggests; the budget counts it expanded."""
    messages = [_message("\tx" * 200) for _ in range(5)]

    kept, text, _ = _fit_messages_to_budget(messages, budget_tokens=900)

    assert _sent_tokens(text) <= 900
    # Counting the raw text would have let more messages in.
    raw_cost = _actual_tokens(f"alice: {messages[0].content}") + 1
    assert len(kept) < 900 // raw_cost


def test_oversized_newest_message_is_clipped_without_modifying_it() -> None:
    """A newest message that alone exceeds the budget is clipped and marked."""
    content = _text_of_tokens(5000)
    newest = _message(content)

    kept, text, clipped = _fit_messages_to_budget([_message("older"), newest], 200)

    assert kept == [newest]
    assert clipped
    assert text.startswith("alice: alpha")
    assert text.endswith("[truncated]")
    assert _sent_tokens(text) <= 200
    assert newest.content == content


def test_clipping_tab_heavy_text_keeps_most_of_the_budget() -> None:
    """Tabs expand to many more tokens than the raw text has; clipping still
    keeps close to a full budget of content instead of discarding most of it."""
    content = "\tx" * 12_000

    kept, text, clipped = _fit_messages_to_budget([_message(content)], 2000)

    assert kept and clipped
    assert _sent_tokens(text) <= 2000
    assert _sent_tokens(text) >= 1900


def test_special_token_text_is_counted_exactly() -> None:
    """Text that looks like a control token is counted as text, not estimated
    from its length, which undercounts dense text such as CJK."""
    content = "<|endoftext|>" + "猫" * 2000
    assert estimate_tokens(f"alice: {content}") < _actual_tokens(f"alice: {content}")

    kept, text, clipped = _fit_messages_to_budget([_message(content)], 1000)

    assert kept and clipped
    assert _sent_tokens(text) <= 1000


def test_nothing_fits_when_the_budget_is_spent() -> None:
    """With no room for even a clipped line, nothing is kept."""
    kept, text, clipped = _fit_messages_to_budget([_message(_text_of_tokens(100))], 2)

    assert (kept, text, clipped) == ([], "", False)


# --- the summary call --------------------------------------------------------


def _configuration(custom_instructions: str | None = None) -> ResolvedConfiguration:
    """A resolved configuration with summaries on, so no DB lookup is needed."""
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
        summary=ResolvedSummaryConfiguration(
            enabled=True,
            messages_per_short_summary=20,
            messages_per_long_summary=60,
            custom_instructions=custom_instructions,
        ),
        dream=ResolvedDreamConfiguration(enabled=False),
    )


async def _session_with_messages(
    db_session: AsyncSession,
    workspace: Workspace,
    peer: Peer,
    contents: list[str],
    previous_summary: Summary | None = None,
) -> tuple[models.Session, list[models.Message]]:
    """Create a session holding `contents` as messages, oldest first."""
    session = (
        await crud.get_or_create_session(
            db_session,
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
            content=content,
            public_id=generate_nanoid(),
            seq_in_session=i + 1,
            token_count=_actual_tokens(content),
        )
        for i, content in enumerate(contents)
    ]
    db_session.add_all(messages)
    await db_session.commit()
    for message in messages:
        await db_session.refresh(message)
    if previous_summary is not None:
        # Through the summarizer's own save, which also refreshes the session
        # cache that the summary run reads from.
        await _save_summary(db_session, previous_summary, workspace.name, session.name)
        await db_session.commit()
    return session, messages


async def _summarize(
    session: models.Session,
    last_message: models.Message,
    summary_type: SummaryType,
    custom_instructions: str | None = None,
) -> Any:
    """Run `_create_and_save_summary` with the LLM call and save patched out.

    The conftest's runtime mocks replace the summary builders; the real ones
    are restored here so the prompt that would be sent can be measured.

    Returns the mocked `honcho_llm_call`, or None if it was never called.
    """
    response = HonchoLLMCallResponse(
        content="summary", input_tokens=10, output_tokens=5, finish_reasons=["stop"]
    )
    with (
        patch("src.utils.summarizer.create_short_summary", new=create_short_summary),
        patch("src.utils.summarizer.create_long_summary", new=create_long_summary),
        patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=response,
        ) as llm_call,
        patch("src.utils.summarizer._save_summary", new=AsyncMock()),
    ):
        await _create_and_save_summary(
            workspace_name=session.workspace_name,
            session_name=session.name,
            message_id=last_message.id,
            message_seq_in_session=last_message.seq_in_session,
            message_public_id=last_message.public_id,
            summary_type=summary_type,
            configuration=_configuration(custom_instructions),
        )
    return llm_call if llm_call.await_count else None


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_type", [SummaryType.SHORT, SummaryType.LONG])
async def test_summary_prompt_stays_under_the_input_budget(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    summary_type: SummaryType,
) -> None:
    """A window far over budget is trimmed to the newest messages that fit."""
    budget = 3000
    monkeypatch.setattr(settings.SUMMARY, "MAX_INPUT_TOKENS", budget)
    workspace, peer = sample_data
    contents = [f"message {i}:" + _WORD * 500 for i in range(12)]
    session, messages = await _session_with_messages(
        db_session, workspace, peer, contents
    )

    llm_call = await _summarize(session, messages[-1], summary_type)

    assert llm_call is not None
    kwargs = llm_call.await_args.kwargs
    assert _actual_tokens(kwargs["prompt"]) <= budget
    assert "message 11:" in kwargs["prompt"]
    assert "message 0:" not in kwargs["prompt"]
    # Telemetry names only the messages actually summarized, newest last.
    source_ids = kwargs["telemetry"].source_message_ids
    assert 0 < len(source_ids) < len(messages)
    assert source_ids == [m.public_id for m in messages[-len(source_ids) :]]


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_type", [SummaryType.SHORT, SummaryType.LONG])
async def test_tabs_and_control_token_text_stay_under_the_budget(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    summary_type: SummaryType,
) -> None:
    """Tab-heavy messages and custom instructions holding control-token text
    and dense CJK still fit, counted as the provider will see them."""
    budget = 6000
    monkeypatch.setattr(settings.SUMMARY, "MAX_INPUT_TOKENS", budget)
    workspace, peer = sample_data
    contents = [f"message {i}:" + "\tx" * 400 for i in range(12)]
    session, messages = await _session_with_messages(
        db_session, workspace, peer, contents
    )
    instructions = "<|endoftext|>" + "猫" * 1500

    llm_call = await _summarize(
        session, messages[-1], summary_type, custom_instructions=instructions
    )

    assert llm_call is not None
    prompt = llm_call.await_args.kwargs["prompt"]
    assert "猫" * 1500 in prompt
    assert _actual_tokens(prompt) <= budget
    assert "message 11:" in prompt


@pytest.mark.asyncio
async def test_previous_summary_counts_against_the_budget(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The previous summary is sent in full, so fewer messages fit beside it."""
    budget = 3000
    monkeypatch.setattr(settings.SUMMARY, "MAX_INPUT_TOKENS", budget)
    workspace, peer = sample_data
    previous: Summary = {
        "content": "Earlier:" + _WORD * 1500,
        "message_id": 0,
        "summary_type": SummaryType.LONG.value,
        "created_at": "2026-01-01T00:00:00Z",
        "token_count": 1500,
        "message_public_id": "",
    }
    contents = [f"message {i}:" + _WORD * 500 for i in range(12)]
    session, messages = await _session_with_messages(
        db_session, workspace, peer, contents, previous_summary=previous
    )

    llm_call = await _summarize(session, messages[-1], SummaryType.LONG)

    assert llm_call is not None
    prompt = llm_call.await_args.kwargs["prompt"]
    assert previous["content"] in prompt
    assert _actual_tokens(prompt) <= budget
    assert "message 11:" in prompt


@pytest.mark.asyncio
async def test_clipped_message_is_accounted_at_its_sent_size(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clipped message's stored token count would overstate the input; the
    short-summary length target is set from what is actually sent."""
    budget = 7000
    monkeypatch.setattr(settings.SUMMARY, "MAX_INPUT_TOKENS", budget)
    monkeypatch.setattr(settings.SUMMARY, "MAX_TOKENS_SHORT", 10_000)
    workspace, peer = sample_data
    # Dense text: about one token per character, within the column's limit.
    content = "猫" * 20_000
    session, messages = await _session_with_messages(
        db_session, workspace, peer, [content]
    )
    assert messages[0].token_count > budget

    llm_call = await _summarize(session, messages[-1], SummaryType.SHORT)

    assert llm_call is not None
    prompt = llm_call.await_args.kwargs["prompt"]
    assert "[truncated]" in prompt
    target = re.search(r"Aim for about (\d+) tokens", prompt)
    assert target is not None
    assert int(target.group(1)) <= budget * SUMMARY_TARGET_RATIO


@pytest.mark.asyncio
async def test_window_within_budget_sends_every_message(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    """Under the default budget, an ordinary window is summarized in full."""
    workspace, peer = sample_data
    session, messages = await _session_with_messages(
        db_session, workspace, peer, [f"message {i}" for i in range(5)]
    )

    llm_call = await _summarize(session, messages[-1], SummaryType.SHORT)

    assert llm_call is not None
    kwargs = llm_call.await_args.kwargs
    assert kwargs["telemetry"].source_message_ids == [m.public_id for m in messages]
    assert all(f"message {i}" in kwargs["prompt"] for i in range(5))


# --- settings ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("short_cap", "long_cap", "budget"),
    [
        pytest.param(1500, 6000, 6000, id="at-the-long-cap"),
        pytest.param(1500, 6000, 11_999, id="under-twice-the-long-cap"),
        pytest.param(10_000, 6000, 15_000, id="short-cap-is-the-larger"),
    ],
)
def test_input_budget_below_twice_the_larger_cap_is_rejected(
    short_cap: int, long_cap: int, budget: int
) -> None:
    """A previous summary can be as long as either cap and is sent in full, so
    the budget must leave at least that much again for everything else."""
    with pytest.raises(ValueError, match="MAX_INPUT_TOKENS"):
        type(settings.SUMMARY)(
            MAX_TOKENS_SHORT=short_cap,
            MAX_TOKENS_LONG=long_cap,
            MAX_INPUT_TOKENS=budget,
        )


def test_input_budget_at_twice_the_larger_cap_is_accepted() -> None:
    """The boundary itself is allowed."""
    summary_settings = type(settings.SUMMARY)(
        MAX_TOKENS_SHORT=10_000, MAX_TOKENS_LONG=6000, MAX_INPUT_TOKENS=20_000
    )

    assert summary_settings.MAX_INPUT_TOKENS == 20_000

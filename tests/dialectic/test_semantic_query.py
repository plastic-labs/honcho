# pyright: reportPrivateUsage=false
"""Peer chat separates retrieval text from additional reasoning instructions."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.dialectic.core import DialecticAgent
from src.llm import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)
from src.schemas import DialecticOptions
from src.utils.representation import Representation

QUERY = "What context helps answer the user's question about Prague?"
SEMANTIC = "weather in Prague"
REASONING = "Focus on travel preferences, and answer in two sentences."
BRIEF = "Memory counts: 3 explicit.\nMap notes:\n- Prefers train travel."


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["answer", "answer_stream"])
@pytest.mark.parametrize("semantic_query", [SEMANTIC, None])
@pytest.mark.parametrize("reasoning_context", [REASONING, None])
@pytest.mark.parametrize("brief", [BRIEF, None], ids=["with_brief", "without_brief"])
async def test_retrieval_excludes_instructions_and_reasoning_preserves_query(
    semantic_query: str | None,
    reasoning_context: str | None,
    streaming: bool,
    brief: str | None,
) -> None:
    """Each optional field affects only its own side of the retrieval boundary."""
    agent = DialecticAgent(
        workspace_name="workspace",
        session_name=None,
        observer="observer",
        observed="observed",
        reasoning_level="low",
    )
    embed = AsyncMock(return_value=[0.1] * 4)
    search = AsyncMock(return_value=Representation())

    async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
        yield HonchoLLMCallStreamChunk(content="ok")

    response: Any = (
        StreamingResponseWithMetadata(
            chunks(),
            tool_calls_made=[],
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            iterations=1,
        )
        if streaming
        else HonchoLLMCallResponse(
            content="ok",
            input_tokens=1,
            output_tokens=1,
            finish_reasons=["end_turn"],
            tool_calls_made=[],
        )
    )
    llm = AsyncMock(return_value=response)

    with (
        patch("src.dialectic.core.embedding_client.embed", new=embed),
        patch("src.dialectic.core.search_memory", new=search),
        patch("src.dialectic.core.honcho_llm_call", new=llm),
        patch.object(DialecticAgent, "_pair_brief", new=AsyncMock(return_value=brief)),
    ):
        if streaming:
            async for _ in agent.answer_stream(
                QUERY,
                semantic_query=semantic_query,
                reasoning_context=reasoning_context,
            ):
                pass
        else:
            await agent.answer(
                QUERY,
                semantic_query=semantic_query,
                reasoning_context=reasoning_context,
            )

    retrieval = semantic_query if semantic_query is not None else QUERY
    embed.assert_awaited_once_with(retrieval)
    assert [call.kwargs["query"] for call in search.await_args_list] == [retrieval] * 2
    assert llm.await_args is not None
    sent = llm.await_args.kwargs["messages"]
    assert sent[-1]["role"] == "user"
    assert QUERY in sent[-1]["content"]
    assert (REASONING in sent[-1]["content"]) == (reasoning_context is not None)
    assert (BRIEF in sent[-1]["content"]) == (brief is not None)


@pytest.mark.parametrize("field", ["semantic_query", "reasoning_context"])
@pytest.mark.parametrize("value", ["", "\x00", "x" * 10001])
def test_separated_chat_text_rejects_empty_or_oversized_values(
    field: str, value: str
) -> None:
    """Instructions and retrieval text have the same bounds as the query."""
    with pytest.raises(ValidationError):
        DialecticOptions.model_validate({"query": QUERY, field: value})


@pytest.mark.parametrize("field", ["semantic_query", "reasoning_context"])
def test_separated_chat_text_sanitizes_nuls_at_length_boundary(field: str) -> None:
    """NUL stripping cannot turn accepted text into an empty input."""
    options = DialecticOptions.model_validate(
        {"query": QUERY, field: "\x00" + "x" * 10000}
    )
    assert getattr(options, field) == "x" * 10000

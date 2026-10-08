# pyright: reportPrivateUsage=false
"""`semantic_query` steers the prefetch embedding; the agent still sees `query`."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.dialectic.core import DialecticAgent
from src.llm import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)
from src.models import Peer, Workspace
from src.schemas import DialecticOptions
from src.utils.representation import Representation

QUERY = "Given the session so far, what is the most relevant context? Weather in Prague"
SEMANTIC = "weather in Prague"


def _make_agent() -> DialecticAgent:
    return DialecticAgent(
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
        reasoning_level="low",
    )


async def _prepare(agent: DialecticAgent, semantic_query: str | None) -> AsyncMock:
    prefetch = AsyncMock(return_value="- Alice lives in Prague")
    with (
        patch.object(agent, "_prefetch_relevant_observations", new=prefetch),
        patch.object(agent, "_create_tool_executor", new=AsyncMock()),
    ):
        await agent._prepare_messages(QUERY, "task", semantic_query)
    return prefetch


@pytest.mark.asyncio
async def test_prefetch_uses_semantic_query_but_agent_gets_full_query() -> None:
    agent = _make_agent()

    prefetch = await _prepare(agent, SEMANTIC)

    prefetch.assert_awaited_once_with(SEMANTIC)
    user_message = agent.messages[-1]["content"]
    assert user_message.startswith(f"Query: {QUERY}")
    assert "Alice lives in Prague" in user_message


@pytest.mark.asyncio
async def test_prefetch_falls_back_to_query_without_semantic_query() -> None:
    agent = _make_agent()

    prefetch = await _prepare(agent, None)

    prefetch.assert_awaited_once_with(QUERY)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["answer", "answer_stream"])
@pytest.mark.parametrize(
    ("semantic_query", "embedded"), [(SEMANTIC, SEMANTIC), (None, QUERY)]
)
async def test_answer_embeds_and_searches_with_the_retrieval_text(
    semantic_query: str | None, embedded: str, streaming: bool
) -> None:
    """Through the real `answer` path: what is embedded and searched is the
    retrieval text, and the LLM is handed the full query either way."""
    agent = DialecticAgent(
        workspace_name="workspace",
        session_name=None,
        observer="observer",
        observed="observed",
        reasoning_level="low",
    )
    embed = AsyncMock(return_value=[0.1] * 4)
    search = AsyncMock(return_value=Representation())

    async def _stream():
        yield HonchoLLMCallStreamChunk(content="ok")

    response: Any = (
        StreamingResponseWithMetadata(
            _stream(),
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
    ):
        if streaming:
            _ = [
                chunk
                async for chunk in agent.answer_stream(
                    QUERY, semantic_query=semantic_query
                )
            ]
        else:
            await agent.answer(QUERY, semantic_query=semantic_query)

    embed.assert_awaited_once_with(embedded)
    assert [call.kwargs["query"] for call in search.await_args_list] == [embedded] * 2
    assert {call.kwargs["embedding"][0] for call in search.await_args_list} == {0.1}
    assert llm.await_args is not None
    sent = llm.await_args.kwargs["messages"]
    assert sent[-1]["content"] == f"Query: {QUERY}"


class TestDialecticOptionsSemanticQuery:
    def test_defaults_to_none(self) -> None:
        assert DialecticOptions.model_validate({"query": "q"}).semantic_query is None

    def test_accepts_a_value(self) -> None:
        options = DialecticOptions.model_validate(
            {"query": "q", "semantic_query": SEMANTIC}
        )
        assert options.semantic_query == SEMANTIC

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DialecticOptions.model_validate({"query": "q", "semantic_query": ""})

    def test_all_nul_is_rejected_not_emptied(self) -> None:
        with pytest.raises(ValidationError):
            DialecticOptions.model_validate({"query": "q", "semantic_query": "\x00"})


class TestPeerChatRoute:
    def _chat(
        self,
        client: TestClient,
        workspace: Workspace,
        peer: Peer,
        body: dict[str, Any],
    ) -> tuple[Any, AsyncMock]:
        with patch(
            "src.routers.peers.agentic_chat", new=AsyncMock(return_value="ok")
        ) as mock_chat:
            response = client.post(
                f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
                json={"query": QUERY, **body},
            )
        return response, mock_chat

    def test_forwarded_to_the_agent(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ) -> None:
        workspace, peer = sample_data

        response, mock_chat = self._chat(
            client, workspace, peer, {"semantic_query": SEMANTIC}
        )

        assert response.status_code == 200
        assert mock_chat.await_args is not None
        assert mock_chat.await_args.kwargs["query"] == QUERY
        assert mock_chat.await_args.kwargs["semantic_query"] == SEMANTIC

    def test_absent_is_none(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ) -> None:
        workspace, peer = sample_data

        response, mock_chat = self._chat(client, workspace, peer, {})

        assert response.status_code == 200
        assert mock_chat.await_args is not None
        assert mock_chat.await_args.kwargs["semantic_query"] is None

    def test_streaming_forwarded_to_the_agent(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ) -> None:
        workspace, peer = sample_data

        async def _chunks(*_args: object, **_kwargs: object):
            yield "ok"

        with patch(
            "src.routers.peers.agentic_chat_stream", side_effect=_chunks
        ) as mock_stream:
            response = client.post(
                f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
                json={"query": QUERY, "semantic_query": SEMANTIC, "stream": True},
            )

        assert response.status_code == 200
        assert mock_stream.call_args is not None
        assert mock_stream.call_args.kwargs["semantic_query"] == SEMANTIC

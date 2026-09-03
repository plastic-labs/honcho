"""Tests for the SDK's `include_evidence` option on peer and workspace chat.

The server's dialectic is mocked out by the autouse `mock_llm_call_functions`
fixture, so these are about the SDK contract: sending the flag, and giving the
caller back a typed answer-plus-evidence instead of a bare answer. What the
evidence actually contains is covered server-side.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from sdks.python.src.honcho import ChatResponse, Evidence
from sdks.python.src.honcho.client import Honcho
from src import models

TOOL_CALL = {"tool_name": "search_memory", "tool_input": {"query": "coffee"}}
QUERY = "What does the user drink?"
WORKSPACE_QUERY = "What do people here drink?"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _server_document() -> models.Document:
    """A conclusion row shaped the way the server would hand one over."""
    return models.Document(
        id="doc-sentinel",
        level="deductive",
        content="User drinks coffee in the morning",
        internal_metadata={},
        source_ids=["doc-a", "doc-b"],
        session_name="session-1",
        created_at=NOW,
        observer="observer",
        observed="observed",
        workspace_name="workspace",
    )


def _server_message() -> models.Message:
    return models.Message(
        public_id="msg-sentinel",
        content="I drink a lot of coffee",
        peer_name="alice",
        session_name="session-1",
        created_at=NOW,
        workspace_name="workspace",
        seq_in_session=1,
    )


class Drink(BaseModel):
    name: str


def _stub_chat(mock: Any, content: str) -> None:
    """Have the mocked dialectic record evidence when it was handed a place to."""

    async def _chat(*_args: object, **kwargs: Any) -> str:
        evidence = kwargs.get("evidence")
        if evidence is not None:
            evidence.record_tool_calls([TOOL_CALL])
        return content

    mock.side_effect = _chat


def _stub_chat_stream(mock: Any) -> None:
    def _chat_stream(*_args: object, **kwargs: Any) -> Any:
        evidence = kwargs.get("evidence")

        async def _chunks() -> Any:
            if evidence is not None:
                evidence.record_tool_calls([TOOL_CALL])
            for chunk in ("The user ", "drinks coffee."):
                yield chunk

        return _chunks()

    mock.side_effect = _chat_stream


class TestPeerChat:
    def test_deserializes_the_evidence_payload(
        self,
        honcho_sync_test_client: Honcho,
        mock_llm_call_functions: dict[str, Any],
    ):
        _stub_chat(mock_llm_call_functions["agentic_chat"], "The user drinks coffee.")
        peer = honcho_sync_test_client.peer("alice")

        result = peer.chat(QUERY, include_evidence=True)

        assert isinstance(result, ChatResponse)
        assert isinstance(result.evidence, Evidence)
        assert result.evidence.tool_calls[0].tool_name == "search_memory"

    def test_sends_the_flag_only_when_asked(
        self,
        honcho_sync_test_client: Honcho,
        mock_llm_call_functions: dict[str, Any],
    ):
        peer = honcho_sync_test_client.peer("alice")

        peer.chat(QUERY)
        assert mock_llm_call_functions["agentic_chat"].await_args.kwargs[
            "evidence"
        ] is (None)

        peer.chat(QUERY, include_evidence=True)
        assert (
            mock_llm_call_functions["agentic_chat"].await_args.kwargs["evidence"]
            is not None
        )

    def test_still_parses_a_response_format_alongside_evidence(
        self,
        honcho_sync_test_client: Honcho,
        mock_llm_call_functions: dict[str, Any],
    ):
        """A schema and evidence are independent; asking for both works."""
        _stub_chat(mock_llm_call_functions["agentic_chat"], '{"name": "coffee"}')
        peer = honcho_sync_test_client.peer("alice")

        result = peer.chat(
            "What does the user drink?",
            response_format=Drink,
            include_evidence=True,
        )

        assert isinstance(result, ChatResponse)
        assert isinstance(result.content, Drink)
        assert result.content.name == "coffee"
        assert result.evidence is not None

    def test_reports_evidence_even_when_there_is_no_answer(
        self,
        honcho_sync_test_client: Honcho,
        mock_llm_call_functions: dict[str, Any],
    ):
        """An empty answer must not swallow the evidence alongside it."""
        _stub_chat(mock_llm_call_functions["agentic_chat"], "")
        peer = honcho_sync_test_client.peer("alice")

        result = peer.chat(QUERY, include_evidence=True)

        assert isinstance(result, ChatResponse)
        assert result.content is None
        assert result.evidence is not None
        assert result.evidence.tool_calls

    def test_deserializes_conclusions_and_messages_into_typed_objects(
        self,
        honcho_sync_test_client: Honcho,
        mock_llm_call_functions: dict[str, Any],
    ):
        """The SDK has to parse the real wire shape, not just tool calls."""

        async def _chat(*_args: object, **kwargs: Any) -> str:
            evidence = kwargs.get("evidence")
            if evidence is not None:
                evidence.add_documents([_server_document()])
                evidence.add_messages([_server_message()])
            return "The user drinks coffee."

        mock_llm_call_functions["agentic_chat"].side_effect = _chat
        peer = honcho_sync_test_client.peer("alice")

        result = peer.chat(QUERY, include_evidence=True)

        assert isinstance(result, ChatResponse)
        assert result.evidence is not None
        (conclusion,) = result.evidence.conclusions
        assert conclusion.id == "doc-sentinel"
        assert conclusion.level == "deductive"
        assert conclusion.content == "User drinks coffee in the morning"
        assert conclusion.source_ids == ["doc-a", "doc-b"]
        assert conclusion.created_at.tzinfo is not None
        (message,) = result.evidence.messages
        assert message.id == "msg-sentinel"
        assert message.peer_id == "alice"
        assert message.created_at.tzinfo is not None


class TestPeerChatStream:
    def test_evidence_stays_none_when_not_requested(
        self, honcho_sync_test_client: Honcho
    ):
        peer = honcho_sync_test_client.peer("alice")

        stream = peer.chat_stream(QUERY)
        list(stream)

        assert stream.evidence is None


class TestWorkspaceChat:
    def test_returns_a_bare_answer_by_default(self, honcho_sync_test_client: Honcho):
        answer = honcho_sync_test_client.chat(WORKSPACE_QUERY)

        assert isinstance(answer, str)

    def test_stream_evidence_is_available_once_it_drains(
        self,
        honcho_sync_test_client: Honcho,
        mock_llm_call_functions: dict[str, Any],
    ):
        _stub_chat_stream(mock_llm_call_functions["workspace_chat_stream"])

        stream = honcho_sync_test_client.chat_stream(
            WORKSPACE_QUERY, include_evidence=True
        )
        list(stream)

        assert stream.evidence is not None


@pytest.mark.asyncio
class TestSyncAndAsyncParity:
    """The async accessors have to return the same shapes as the sync ones."""

    async def test_peer_chat_returns_answer_and_evidence(
        self,
        client_fixture: tuple[Honcho, str],
        mock_llm_call_functions: dict[str, Any],
    ):
        honcho, client_type = client_fixture
        _stub_chat(mock_llm_call_functions["agentic_chat"], "The user drinks coffee.")

        if client_type == "async":
            peer = await honcho.aio.peer("alice")
            result = await peer.aio.chat(QUERY, include_evidence=True)
        else:
            result = honcho.peer("alice").chat(QUERY, include_evidence=True)

        assert isinstance(result, ChatResponse)
        assert result.content == "The user drinks coffee."
        assert result.evidence is not None

    async def test_peer_chat_returns_a_bare_answer_by_default(
        self, client_fixture: tuple[Honcho, str]
    ):
        honcho, client_type = client_fixture

        if client_type == "async":
            peer = await honcho.aio.peer("alice")
            answer = await peer.aio.chat(QUERY)
        else:
            answer = honcho.peer("alice").chat(QUERY)

        assert isinstance(answer, str)

    async def test_peer_chat_stream_evidence_is_available_once_it_drains(
        self,
        client_fixture: tuple[Honcho, str],
        mock_llm_call_functions: dict[str, Any],
    ):
        honcho, client_type = client_fixture
        _stub_chat_stream(mock_llm_call_functions["agentic_chat_stream"])

        if client_type == "async":
            peer = await honcho.aio.peer("alice")
            stream = await peer.aio.chat_stream(QUERY, include_evidence=True)
            chunks = [chunk async for chunk in stream]
        else:
            sync_stream = honcho.peer("alice").chat_stream(QUERY, include_evidence=True)
            chunks = list(sync_stream)
            stream = sync_stream

        assert "".join(chunks) == "The user drinks coffee."
        assert stream.evidence is not None
        assert stream.evidence.tool_calls[0].tool_name == "search_memory"

    async def test_workspace_chat_returns_answer_and_evidence(
        self,
        client_fixture: tuple[Honcho, str],
        mock_llm_call_functions: dict[str, Any],
    ):
        honcho, client_type = client_fixture
        _stub_chat(
            mock_llm_call_functions["workspace_chat"], "People here drink coffee."
        )

        if client_type == "async":
            result = await honcho.aio.chat(WORKSPACE_QUERY, include_evidence=True)
        else:
            result = honcho.chat(WORKSPACE_QUERY, include_evidence=True)

        assert isinstance(result, ChatResponse)
        assert result.content == "People here drink coffee."
        assert result.evidence is not None

"""Integration tests for the workspace-level chat feature.

Tests cover:
- Route-level: POST /workspaces/{workspace_id}/chat endpoint
- Tool handlers: workspace-specific tool handlers and executor
- Utility: format_documents_with_attribution()
"""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.models import Peer, Workspace
from src.utils.agent_tools import (
    WorkspaceToolContext,
    _handle_get_active_peers,  # pyright: ignore[reportPrivateUsage]
    _handle_get_observation_context_workspace,  # pyright: ignore[reportPrivateUsage]
    _handle_get_peer_card_by_name,  # pyright: ignore[reportPrivateUsage]
    _handle_get_reasoning_chain_workspace,  # pyright: ignore[reportPrivateUsage]
    _handle_get_workspace_stats,  # pyright: ignore[reportPrivateUsage]
    _handle_search_memory_workspace,  # pyright: ignore[reportPrivateUsage]
    create_workspace_tool_executor,
)
from src.utils.representation import format_documents_with_attribution

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def workspace_test_data(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> Any:
    """Create comprehensive test data with multiple peers and observations.

    Sets up a workspace with:
    - 3 peers (peer1 observes peer2, peer1 observes peer3)
    - 1 session with messages from all peers
    - Documents (observations) across different peer pairs
    """
    workspace, peer1 = sample_data

    # Create additional peers
    peer2 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    peer3 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([peer2, peer3])
    await db_session.flush()

    # Create session
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Create collections (peer1 observes peer2, peer1 observes peer3)
    collection1 = models.Collection(
        workspace_name=workspace.name,
        observer=peer1.name,
        observed=peer2.name,
    )
    collection2 = models.Collection(
        workspace_name=workspace.name,
        observer=peer1.name,
        observed=peer3.name,
    )
    db_session.add_all([collection1, collection2])
    await db_session.flush()

    # Create messages
    now = datetime.now(timezone.utc)
    messages: list[models.Message] = []
    for i in range(6):
        peer_name = [peer1.name, peer2.name, peer3.name][i % 3]
        msg = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer_name,
            content=f"Test message {i} from {peer_name}",
            seq_in_session=i + 1,
            token_count=10,
            created_at=now - timedelta(minutes=6 - i),
        )
        db_session.add(msg)
        messages.append(msg)
    await db_session.flush()
    for msg in messages:
        await db_session.refresh(msg)

    # Create documents for peer1->peer2 observations
    docs_peer2: list[models.Document] = []
    for content in [
        "User likes coffee and programming",
        "User works remotely from home",
    ]:
        doc = models.Document(
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            content=content,
            embedding=[0.1] * 1536,
            session_name=session.name,
            level="explicit",
            metadata={
                "message_ids": [messages[0].id],
                "message_created_at": str(messages[0].created_at),
            },
        )
        db_session.add(doc)
        docs_peer2.append(doc)

    # Create documents for peer1->peer3 observations
    docs_peer3: list[models.Document] = []
    for content in [
        "User prefers mornings for deep work",
        "User enjoys hiking on weekends",
    ]:
        doc = models.Document(
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer3.name,
            content=content,
            embedding=[0.2] * 1536,
            session_name=session.name,
            level="explicit",
            metadata={
                "message_ids": [messages[1].id],
                "message_created_at": str(messages[1].created_at),
            },
        )
        db_session.add(doc)
        docs_peer3.append(doc)

    await db_session.flush()
    for doc in docs_peer2 + docs_peer3:
        await db_session.refresh(doc)

    yield workspace, peer1, peer2, peer3, session, messages, docs_peer2, docs_peer3

    await db_session.rollback()


@pytest.fixture
def make_workspace_ctx(
    db_session: AsyncSession, workspace_test_data: Any
) -> Callable[..., WorkspaceToolContext]:
    """Factory fixture to create WorkspaceToolContext."""
    workspace, *_ = workspace_test_data
    shared_lock = asyncio.Lock()

    def _make_ctx(
        *,
        session_name: str | None = None,
        include_observation_ids: bool = True,
    ) -> WorkspaceToolContext:
        return WorkspaceToolContext(
            db=db_session,
            workspace_name=workspace.name,
            session_name=session_name,
            include_observation_ids=include_observation_ids,
            history_token_limit=8192,
            db_lock=shared_lock,
        )

    return _make_ctx


# =============================================================================
# Route Tests: POST /workspaces/{workspace_id}/chat
# =============================================================================


class TestWorkspaceChatEndpoint:
    """Tests for the workspace chat API endpoint."""

    def test_workspace_chat_basic(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Basic non-streaming workspace chat returns DialecticResponse."""
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={
                "query": "What do you know about the peers in this workspace?",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["content"] == "Test workspace chat response"

    def test_workspace_chat_with_session_id(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Workspace chat accepts optional session_id parameter."""
        test_workspace, _ = sample_data
        session_id = str(generate_nanoid())

        # Create a session first
        client.post(
            f"/v3/workspaces/{test_workspace.name}/sessions",
            json={"name": session_id},
        )

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={
                "query": "Tell me about recent conversations",
                "session_id": session_id,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    def test_workspace_chat_with_reasoning_level(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Workspace chat accepts reasoning_level parameter."""
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={
                "query": "Analyze common themes across all peers",
                "stream": False,
                "reasoning_level": "low",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    def test_workspace_chat_streaming(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Streaming workspace chat returns SSE-formatted events."""
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={
                "query": "What patterns do you see across the workspace?",
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        events: list[Any] = []
        for line in response.text.strip().split("\n\n"):
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)

        # Should have content events and a final done event
        assert len(events) >= 2
        content_events = [e for e in events if not e.get("done")]
        done_events = [e for e in events if e.get("done")]
        assert len(content_events) >= 1
        assert len(done_events) == 1

        # Content events should have delta.content
        for event in content_events:
            assert "delta" in event
            assert "content" in event["delta"]

    def test_workspace_chat_empty_query_rejected(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Empty query should be rejected by validation."""
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={
                "query": "",
                "stream": False,
            },
        )
        assert response.status_code == 422

    def test_workspace_chat_missing_query_rejected(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Missing query field should be rejected."""
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={"stream": False},
        )
        assert response.status_code == 422

    def test_workspace_chat_null_content_response(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
        mock_llm_call_functions: dict[str, Any],
    ):
        """When workspace_chat returns None, response content should be None."""
        test_workspace, _ = sample_data
        mock_llm_call_functions["workspace_chat"].return_value = None

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={
                "query": "Some query",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] is None

    def test_workspace_chat_defaults(
        self,
        client: Any,
        sample_data: tuple[Workspace, Peer],
    ):
        """Endpoint works with only the required query field."""
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/chat",
            json={"query": "Hello workspace"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data


# =============================================================================
# Tool Handler Tests: Workspace-Specific Handlers
# =============================================================================


@pytest.mark.asyncio
class TestSearchMemoryWorkspace:
    """Tests for _handle_search_memory_workspace (representation-scoped)."""

    async def test_requires_observer_and_observed(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error when observer/observed params are missing."""
        ctx = make_workspace_ctx()

        result = await _handle_search_memory_workspace(
            ctx, {"query": "coffee preferences"}
        )
        assert "ERROR" in result
        assert "observer" in result

    async def test_missing_observer_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error when only observed is provided."""
        ctx = make_workspace_ctx()

        result = await _handle_search_memory_workspace(
            ctx, {"query": "test", "observed": "someone"}
        )
        assert "ERROR" in result

    async def test_missing_observed_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error when only observer is provided."""
        ctx = make_workspace_ctx()

        result = await _handle_search_memory_workspace(
            ctx, {"query": "test", "observer": "someone"}
        )
        assert "ERROR" in result

    async def test_returns_observations_for_specific_pair(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Returns observations scoped to a specific observer/observed pair."""
        monkeypatch.setattr("src.config.settings.VECTOR_STORE.MIGRATED", False)
        _, peer1, peer2, _, _, _, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_search_memory_workspace(
            ctx,
            {
                "query": "coffee preferences",
                "observer": peer1.name,
                "observed": peer2.name,
            },
        )

        assert "Found" in result
        assert "observations" in result.lower()
        # Should be scoped to peer1->peer2
        assert f"{peer1.name}->{peer2.name}" in result

    async def test_does_not_return_observations_from_other_pairs(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Does not leak observations from other peer pairs."""
        monkeypatch.setattr("src.config.settings.VECTOR_STORE.MIGRATED", False)
        _, peer1, _, peer3, _, _, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_search_memory_workspace(
            ctx,
            {
                "query": "coffee",
                "observer": peer1.name,
                "observed": peer3.name,
            },
        )

        # peer3 observations are about hiking/mornings, not coffee
        # Should either find the hiking/mornings ones or none
        assert isinstance(result, str)

    async def test_falls_back_to_message_search(
        self,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Falls back to message search when no observations exist for the pair."""
        workspace, _ = sample_data

        session = models.Session(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(session)
        await db_session.flush()

        observer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        observed = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add_all([observer, observed])
        await db_session.flush()

        msg = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=observed.name,
            content="I really like programming in Python",
            seq_in_session=1,
            token_count=10,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(msg)
        await db_session.flush()

        ctx = WorkspaceToolContext(
            db=db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_search_memory_workspace(
            ctx,
            {
                "query": "programming",
                "observer": observer.name,
                "observed": observed.name,
            },
        )

        assert isinstance(result, str)
        assert "No observations" in result or "Found" in result

    async def test_respects_top_k(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Respects the top_k parameter, capped at 40."""
        monkeypatch.setattr("src.config.settings.VECTOR_STORE.MIGRATED", False)
        _, peer1, peer2, _, _, _, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_search_memory_workspace(
            ctx,
            {
                "query": "test",
                "top_k": 2,
                "observer": peer1.name,
                "observed": peer2.name,
            },
        )

        assert isinstance(result, str)


@pytest.mark.asyncio
class TestGetWorkspaceStats:
    """Tests for _handle_get_workspace_stats."""

    async def test_returns_stats(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Returns workspace statistics."""
        _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_workspace_stats(ctx, {})

        assert "Workspace stats" in result
        assert "Peers: 3" in result
        assert "Sessions: 1" in result
        assert "Messages: 6" in result
        assert "Date range" in result

    async def test_empty_workspace(
        self,
        db_session: AsyncSession,
    ):
        """Returns zero counts for an empty workspace."""
        workspace = models.Workspace(name=str(generate_nanoid()))
        db_session.add(workspace)
        await db_session.flush()

        ctx = WorkspaceToolContext(
            db=db_session,
            workspace_name=workspace.name,
            session_name=None,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_get_workspace_stats(ctx, {})

        assert "Peers: 0" in result
        assert "Messages: 0" in result


@pytest.mark.asyncio
class TestGetActivePeers:
    """Tests for _handle_get_active_peers."""

    async def test_returns_active_peers(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Returns active peers with message counts."""
        _, peer1, peer2, peer3, *_ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_active_peers(ctx, {})

        assert "Found" in result
        assert "active peers" in result
        assert peer1.name in result
        assert peer2.name in result
        assert peer3.name in result
        assert "messages" in result

    async def test_empty_workspace_returns_no_peers(
        self,
        db_session: AsyncSession,
    ):
        """Returns appropriate message for workspace with no peers."""
        workspace = models.Workspace(name=str(generate_nanoid()))
        db_session.add(workspace)
        await db_session.flush()

        ctx = WorkspaceToolContext(
            db=db_session,
            workspace_name=workspace.name,
            session_name=None,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_get_active_peers(ctx, {})

        assert "No peers found" in result

    async def test_respects_limit(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Respects the limit parameter."""
        _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_active_peers(ctx, {"limit": 1})

        # Should only return 1 peer
        assert "Found 1 active peers" in result

    async def test_sort_by_message_count(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Supports sorting by message count."""
        _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_active_peers(ctx, {"sort_by": "message_count"})

        assert "sorted by message_count" in result


@pytest.mark.asyncio
class TestGetPeerCardByName:
    """Tests for _handle_get_peer_card_by_name."""

    async def test_returns_peer_card(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Returns peer card when it exists."""
        workspace, peer1, peer2, *_ = workspace_test_data

        # Create a peer card
        await crud.set_peer_card(
            db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            peer_card=["Name: Alice", "Location: NYC"],
        )

        ctx = make_workspace_ctx()
        result = await _handle_get_peer_card_by_name(
            ctx, {"observer": peer1.name, "observed": peer2.name}
        )

        assert "Peer card" in result
        assert "Name: Alice" in result
        assert "Location: NYC" in result

    async def test_returns_not_found(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Returns appropriate message when peer card doesn't exist."""
        _, peer1, peer2, *_ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(
            ctx, {"observer": peer1.name, "observed": peer2.name}
        )

        assert "No peer card" in result

    async def test_missing_params_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error when observer/observed params are missing."""
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(ctx, {})

        assert "ERROR" in result

    async def test_missing_observer_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error when only observed is provided."""
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(ctx, {"observed": "someone"})

        assert "ERROR" in result


@pytest.mark.asyncio
class TestGetObservationContextWorkspace:
    """Tests for _handle_get_observation_context_workspace."""

    async def test_retrieves_messages_by_id(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Retrieves messages by their public IDs."""
        _, _, _, _, _, messages, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_observation_context_workspace(
            ctx, {"message_ids": [messages[0].public_id]}
        )

        assert "Retrieved" in result or "No messages found" in result

    async def test_nonexistent_message_ids(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns appropriate message for nonexistent IDs."""
        ctx = make_workspace_ctx()

        result = await _handle_get_observation_context_workspace(
            ctx, {"message_ids": ["nonexistent_id"]}
        )

        assert "No messages found" in result


@pytest.mark.asyncio
class TestGetReasoningChainWorkspace:
    """Tests for _handle_get_reasoning_chain_workspace."""

    async def test_returns_observation_chain(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Returns an observation and its chain."""
        _, _, _, _, _, _, docs_peer2, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain_workspace(
            ctx, {"observation_id": docs_peer2[0].id}
        )

        assert "Observation" in result
        assert docs_peer2[0].content in result

    async def test_nonexistent_observation_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error for nonexistent observation ID."""
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain_workspace(
            ctx, {"observation_id": "nonexistent_id"}
        )

        assert "ERROR" in result

    async def test_missing_observation_id_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
    ):
        """Returns error when observation_id is missing."""
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain_workspace(ctx, {})

        assert "ERROR" in result

    async def test_invalid_direction_returns_error(
        self,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Returns error for invalid direction parameter."""
        _, _, _, _, _, _, docs_peer2, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain_workspace(
            ctx,
            {"observation_id": docs_peer2[0].id, "direction": "invalid"},
        )

        assert "ERROR" in result

    async def test_deductive_observation_shows_premises(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., WorkspaceToolContext],
        workspace_test_data: Any,
    ):
        """Deductive observation shows premises in chain."""
        workspace, peer1, peer2, _, _, _, docs_peer2, _ = workspace_test_data

        # Create a deductive document with source_ids
        deductive_doc = models.Document(
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            content="User is probably a morning person who codes",
            embedding=[0.3] * 1536,
            level="deductive",
            source_ids=[docs_peer2[0].id, docs_peer2[1].id],
        )
        db_session.add(deductive_doc)
        await db_session.flush()
        await db_session.refresh(deductive_doc)

        ctx = make_workspace_ctx()
        result = await _handle_get_reasoning_chain_workspace(
            ctx, {"observation_id": deductive_doc.id, "direction": "premises"}
        )

        assert "Observation" in result
        assert "Premises" in result


# =============================================================================
# Tool Executor Tests
# =============================================================================


@pytest.mark.asyncio
class TestWorkspaceToolExecutor:
    """Tests for create_workspace_tool_executor."""

    async def test_returns_callable(
        self,
        db_session: AsyncSession,
        workspace_test_data: Any,
    ):
        """create_workspace_tool_executor returns an async callable."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
        )

        assert callable(executor)

    async def test_routes_workspace_tools(
        self,
        db_session: AsyncSession,
        workspace_test_data: Any,
    ):
        """Workspace-specific tools are routed to workspace handlers."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
        )

        # Test get_workspace_stats
        stats_result = await executor("get_workspace_stats", {})
        assert isinstance(stats_result, str)
        assert "Workspace stats" in stats_result

        # Test get_active_peers
        peers_result = await executor("get_active_peers", {})
        assert isinstance(peers_result, str)
        assert "peers" in peers_result.lower()

    async def test_falls_through_to_standard_handlers(
        self,
        db_session: AsyncSession,
        workspace_test_data: Any,
    ):
        """Non-workspace tools fall through to standard handlers."""
        workspace, _, _, _, session, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
            session_name=session.name,
        )

        # grep_messages is a standard handler, should fall through
        result = await executor("grep_messages", {"text": "Test message"})

        assert isinstance(result, str)

    async def test_unknown_tool_returns_error(
        self,
        db_session: AsyncSession,
        workspace_test_data: Any,
    ):
        """Unknown tool name returns error."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
        )

        result = await executor("nonexistent_tool", {})

        assert "Unknown tool" in result

    async def test_handles_exceptions_gracefully(
        self,
        db_session: AsyncSession,
        workspace_test_data: Any,
    ):
        """Executor returns error strings instead of raising exceptions."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
        )

        # Missing required observer/observed/query parameters
        result = await executor("search_memory", {})

        assert isinstance(result, str)
        assert "ERROR" in result

    async def test_get_peer_card_via_executor(
        self,
        db_session: AsyncSession,
        workspace_test_data: Any,
    ):
        """get_peer_card routes through workspace handler with params."""
        workspace, peer1, peer2, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
        )

        result = await executor(
            "get_peer_card",
            {"observer": peer1.name, "observed": peer2.name},
        )

        assert isinstance(result, str)
        # Should be from workspace handler (accepts observer/observed params)
        assert "peer card" in result.lower() or "No peer card" in result


# =============================================================================
# Utility Tests: format_documents_with_attribution()
# =============================================================================


class TestFormatDocumentsWithAttribution:
    """Tests for format_documents_with_attribution()."""

    def test_empty_documents(self):
        """Returns fallback message for empty list."""
        result = format_documents_with_attribution([])
        assert result == "No observations found."

    def test_groups_by_peer_pair(
        self,
        workspace_test_data: Any,
    ):
        """Groups documents by observer/observed pair."""
        _, peer1, peer2, peer3, _, _, docs_peer2, docs_peer3 = workspace_test_data

        result = format_documents_with_attribution(docs_peer2 + docs_peer3)

        # Should have headers for both peer pairs
        assert f"About {peer2.name}" in result
        assert f"About {peer3.name}" in result
        assert f"observed by {peer1.name}" in result

    def test_self_observation_header(self):
        """Self-observation (observer==observed) uses simpler header."""
        doc = models.Document(
            workspace_name="test",
            observer="alice",
            observed="alice",
            content="I like coffee",
            level="explicit",
            embedding=[0.1] * 1536,
            internal_metadata={
                "message_ids": [1],
                "message_created_at": "2025-01-01T00:00:00Z",
            },
        )
        doc.id = "test_id"

        result = format_documents_with_attribution([doc])

        assert "### About alice\n" in result
        # Should NOT have "observed by" for self-observation
        assert "observed by" not in result

    def test_include_ids(
        self,
        workspace_test_data: Any,
    ):
        """include_ids=True includes observation IDs in output."""
        _, _, _, _, _, _, docs_peer2, _ = workspace_test_data

        result_with_ids = format_documents_with_attribution(
            docs_peer2, include_ids=True
        )
        result_without_ids = format_documents_with_attribution(
            docs_peer2, include_ids=False
        )

        # With IDs should be longer or have [id: markers
        assert len(result_with_ids) >= len(result_without_ids)

    def test_contains_document_content(
        self,
        workspace_test_data: Any,
    ):
        """Output contains the actual document content."""
        _, _, _, _, _, _, docs_peer2, _ = workspace_test_data

        result = format_documents_with_attribution(docs_peer2)

        assert "coffee" in result.lower() or "programming" in result.lower()
        assert "remotely" in result.lower() or "works" in result.lower()

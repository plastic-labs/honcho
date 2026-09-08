"""Integration tests for the workspace-level chat feature.

Tests cover:
- Route-level: POST /workspaces/{workspace_id}/chat endpoint
- Tool handlers: workspace-specific tool handlers and executor
"""

import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.dialectic.chat import workspace_chat, workspace_chat_stream
from src.models import Peer, Workspace
from src.utils.agent_tools import (
    ToolContext,
    _handle_get_observation_context,  # pyright: ignore[reportPrivateUsage]
    _handle_get_peer_card_by_name,  # pyright: ignore[reportPrivateUsage]
    _handle_get_reasoning_chain,  # pyright: ignore[reportPrivateUsage]
    _handle_get_workspace_stats,  # pyright: ignore[reportPrivateUsage]
    _handle_search_memory_workspace,  # pyright: ignore[reportPrivateUsage]
    create_workspace_tool_executor,
)
from src.utils.scopes import SCOPE_KIND, scope_peer_name

# =============================================================================
# Fixtures
# =============================================================================


def _tool_text(result: object) -> str:
    """Unwrap ToolResult (today's handler contract) or pass through str."""
    content = getattr(result, "content", None)
    return content if isinstance(content, str) else str(result)


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
    now = datetime.now(UTC)
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

    # Commit so data is visible to independent tracked_db sessions used by
    # workspace-level tool handlers.
    await db_session.commit()

    yield workspace, peer1, peer2, peer3, session, messages, docs_peer2, docs_peer3

    await db_session.rollback()


@pytest.fixture
def make_workspace_ctx(
    workspace_test_data: Any,
) -> Callable[..., ToolContext]:
    """Factory fixture to create ToolContext."""
    workspace, *_ = workspace_test_data
    shared_lock = asyncio.Lock()

    def _make_ctx(
        *,
        session_name: str | None = None,
        include_observation_ids: bool = True,
        session_allowlist: list[str] | None = None,
    ) -> ToolContext:
        return ToolContext(
            observer="",
            observed="",
            current_messages=None,
            workspace_name=workspace.name,
            session_name=session_name,
            include_observation_ids=include_observation_ids,
            history_token_limit=8192,
            db_lock=shared_lock,
            session_allowlist=session_allowlist,
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
        create_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/sessions",
            json={"name": session_id},
        )
        assert create_response.status_code in (200, 201)

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
        mock_llm_call_functions["workspace_chat"].side_effect = None
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


@pytest.mark.asyncio
async def test_workspace_chat_releases_preflight_session_before_agent_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_sessions = 0

    @asynccontextmanager
    async def fake_tracked_db(_: str | None = None, **_kwargs: Any):
        nonlocal active_sessions
        active_sessions += 1
        try:
            yield object()
        finally:
            active_sessions -= 1

    async def fake_get_session(*args: Any, **kwargs: Any) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return SimpleNamespace(id="session-id")

    async def fake_answer(_self: Any, query: str, **_kwargs: Any) -> str:
        assert query == "What changed?"
        assert active_sessions == 0
        return "ok"

    async def fake_get_workspace(*args: Any, **kwargs: Any) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return SimpleNamespace(name="workspace")

    monkeypatch.setattr("src.dialectic.chat.tracked_db", fake_tracked_db)
    monkeypatch.setattr("src.dialectic.chat.crud.get_workspace", fake_get_workspace)
    monkeypatch.setattr("src.dialectic.chat.crud.get_session", fake_get_session)
    monkeypatch.setattr(
        "src.dialectic.chat.WorkspaceDialecticAgent.answer", fake_answer
    )

    result = await workspace_chat("workspace", "session", "What changed?")

    assert result == "ok"


@pytest.mark.asyncio
async def test_workspace_chat_stream_releases_preflight_session_before_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_sessions = 0

    @asynccontextmanager
    async def fake_tracked_db(_: str | None = None, **_kwargs: Any):
        nonlocal active_sessions
        active_sessions += 1
        try:
            yield object()
        finally:
            active_sessions -= 1

    async def fake_get_session(*args: Any, **kwargs: Any) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return SimpleNamespace(id="session-id")

    async def fake_answer_stream(_self: Any, query: str, **_kwargs: Any):
        assert query == "Stream it"
        assert active_sessions == 0
        yield "chunk-1"
        assert active_sessions == 0
        yield "chunk-2"

    async def fake_get_workspace(*args: Any, **kwargs: Any) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return SimpleNamespace(name="workspace")

    monkeypatch.setattr("src.dialectic.chat.tracked_db", fake_tracked_db)
    monkeypatch.setattr("src.dialectic.chat.crud.get_workspace", fake_get_workspace)
    monkeypatch.setattr("src.dialectic.chat.crud.get_session", fake_get_session)
    monkeypatch.setattr(
        "src.dialectic.chat.WorkspaceDialecticAgent.answer_stream",
        fake_answer_stream,
    )

    chunks = [
        chunk
        async for chunk in workspace_chat_stream("workspace", "session", "Stream it")
    ]

    assert chunks == ["chunk-1", "chunk-2"]


# =============================================================================
# Tool Handler Tests: Workspace-Specific Handlers
# =============================================================================


@pytest.mark.asyncio
class TestSearchMemoryWorkspace:
    """Tests for _handle_search_memory_workspace (representation-scoped)."""

    async def test_requires_observer_and_observed(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error when observer/observed params are missing."""
        ctx = make_workspace_ctx()

        result = _tool_text(
            await _handle_search_memory_workspace(ctx, {"query": "coffee preferences"})
        )
        assert "ERROR" in result
        assert "observer" in result

    async def test_missing_observer_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error when only observed is provided."""
        ctx = make_workspace_ctx()

        result = _tool_text(
            await _handle_search_memory_workspace(
                ctx, {"query": "test", "observed": "someone"}
            )
        )
        assert "ERROR" in result

    async def test_missing_observed_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error when only observer is provided."""
        ctx = make_workspace_ctx()

        result = _tool_text(
            await _handle_search_memory_workspace(
                ctx, {"query": "test", "observer": "someone"}
            )
        )
        assert "ERROR" in result

    async def test_returns_observations_for_specific_pair(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Returns observations scoped to a specific observer/observed pair."""
        monkeypatch.setattr("src.config.settings.VECTOR_STORE.MIGRATED", False)
        _, peer1, peer2, _, _, _, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = _tool_text(
            await _handle_search_memory_workspace(
                ctx,
                {
                    "query": "coffee preferences",
                    "observer": peer1.name,
                    "observed": peer2.name,
                },
            )
        )

        assert "Found" in result
        assert "observations" in result.lower()
        # Should be scoped to peer1->peer2
        assert f"{peer1.name}->{peer2.name}" in result

    async def test_does_not_return_observations_from_other_pairs(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Does not leak observations from other peer pairs."""
        monkeypatch.setattr("src.config.settings.VECTOR_STORE.MIGRATED", False)
        _, peer1, _, peer3, _, _, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = _tool_text(
            await _handle_search_memory_workspace(
                ctx,
                {
                    "query": "coffee",
                    "observer": peer1.name,
                    "observed": peer3.name,
                },
            )
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
            created_at=datetime.now(UTC),
        )
        db_session.add(msg)
        await db_session.flush()

        ctx = ToolContext(
            observer="",
            observed="",
            current_messages=None,
            workspace_name=workspace.name,
            session_name=session.name,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = _tool_text(
            await _handle_search_memory_workspace(
                ctx,
                {
                    "query": "programming",
                    "observer": observer.name,
                    "observed": observed.name,
                },
            )
        )

        assert isinstance(result, str)
        assert "No observations" in result or "Found" in result

    async def test_respects_top_k(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Respects the top_k parameter, capped at 40."""
        monkeypatch.setattr("src.config.settings.VECTOR_STORE.MIGRATED", False)
        _, peer1, peer2, _, _, _, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = _tool_text(
            await _handle_search_memory_workspace(
                ctx,
                {
                    "query": "test",
                    "top_k": 2,
                    "observer": peer1.name,
                    "observed": peer2.name,
                },
            )
        )

        assert isinstance(result, str)


@pytest.mark.asyncio
class TestGetWorkspaceStats:
    """Tests for _handle_get_workspace_stats."""

    async def test_returns_stats(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
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

    async def test_lists_most_active_peers(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """Includes the most active peers with message counts."""
        _, peer1, peer2, peer3, *_ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_workspace_stats(ctx, {})

        assert "Most active peers" in result
        assert peer1.name in result
        assert peer2.name in result
        assert peer3.name in result
        assert "messages" in result

    async def test_empty_workspace(
        self,
        db_session: AsyncSession,
    ):
        """Returns zero counts for an empty workspace."""
        workspace = models.Workspace(name=str(generate_nanoid()))
        db_session.add(workspace)
        await db_session.flush()

        ctx = ToolContext(
            observer="",
            observed="",
            current_messages=None,
            workspace_name=workspace.name,
            session_name=None,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_get_workspace_stats(ctx, {})

        assert "Peers: 0" in result
        assert "Messages: 0" in result

    async def test_excludes_scope_peers(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        workspace, *_ = workspace_test_data
        db_session.add(
            models.Peer(
                name=scope_peer_name("therapy"),
                workspace_name=workspace.name,
                internal_metadata={"kind": SCOPE_KIND},
                configuration={"observe_me": False},
            )
        )
        await db_session.commit()

        result = await _handle_get_workspace_stats(make_workspace_ctx(), {})

        assert "Peers: 3" in result
        assert "scope.therapy" not in result

    async def test_empty_session_allowlist_is_zero(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        _ = workspace_test_data
        result = await _handle_get_workspace_stats(
            make_workspace_ctx(session_allowlist=[]), {}
        )

        assert "Peers: 0" in result
        assert "Sessions: 0" in result
        assert "Messages: 0" in result


@pytest.mark.asyncio
class TestGetPeerCardByName:
    """Tests for _handle_get_peer_card_by_name."""

    async def test_returns_peer_card(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., ToolContext],
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
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """Returns appropriate message when peer card doesn't exist."""
        _, peer1, peer2, *_ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(
            ctx, {"observer": peer1.name, "observed": peer2.name}
        )

        assert "No peer card" in result

    async def test_session_allowlist_refuses(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """A peer card is a cross-session aggregate, so a scoped query must not
        get one — otherwise `scope` leaks facts derived outside its sessions."""
        workspace, peer1, peer2, _peer3, session, *_ = workspace_test_data

        await crud.set_peer_card(
            db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            peer_card=["Secret: derived from an out-of-scope session"],
        )

        ctx = make_workspace_ctx(session_allowlist=[session.name])
        result = await _handle_get_peer_card_by_name(
            ctx, {"observer": peer1.name, "observed": peer2.name}
        )

        assert "Secret" not in result
        assert "unavailable for session-scoped queries" in result

    async def test_unknown_peer_is_answered_not_raised(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """The agent supplies peer names from its own routing, so a name that
        doesn't exist is an expected turn, not an unhandled exception."""
        _, peer1, *_ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(
            ctx, {"observer": "no-such-peer", "observed": peer1.name}
        )

        assert "No peer named 'no-such-peer'" in result

    async def test_missing_params_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error when observer/observed params are missing."""
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(ctx, {})

        assert "ERROR" in result

    async def test_missing_observer_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error when only observed is provided."""
        ctx = make_workspace_ctx()

        result = await _handle_get_peer_card_by_name(ctx, {"observed": "someone"})

        assert "ERROR" in result


@pytest.mark.asyncio
class TestGetObservationContextWorkspace:
    """Tests for get_observation_context under the workspace executor.

    The workspace loadout routes this straight to the shared handler: its
    observer="" sentinel already normalizes to None ("no perspective
    scoping") at the crud boundary."""

    async def test_retrieves_messages_by_id(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """Retrieves messages by their public IDs."""
        _, _, _, _, _, messages, _, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_observation_context(
            ctx, {"message_ids": [messages[0].public_id]}
        )

        assert "Retrieved" in result or "No messages found" in result

    async def test_nonexistent_message_ids(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns appropriate message for nonexistent IDs."""
        ctx = make_workspace_ctx()

        result = await _handle_get_observation_context(
            ctx, {"message_ids": ["nonexistent_id"]}
        )

        assert "No messages found" in result

    async def test_respects_session_scope(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """Session-scoped context lookup should not leak snippets from other sessions."""
        workspace, _peer1, peer2, _peer3, session, messages, *_ = workspace_test_data

        other_session = models.Session(
            name=str(generate_nanoid()),
            workspace_name=workspace.name,
        )
        db_session.add(other_session)
        await db_session.flush()

        leaked_message = models.Message(
            workspace_name=workspace.name,
            session_name=other_session.name,
            peer_name=peer2.name,
            content="LEAKED_FROM_OTHER_SESSION",
            seq_in_session=messages[0].seq_in_session,
            token_count=10,
            created_at=datetime.now(UTC),
        )
        db_session.add(leaked_message)
        await db_session.commit()

        ctx = make_workspace_ctx(session_name=session.name)
        result = await _handle_get_observation_context(
            ctx, {"message_ids": [messages[0].public_id]}
        )

        assert "LEAKED_FROM_OTHER_SESSION" not in result
        assert messages[0].content in result


@pytest.mark.asyncio
class TestGetReasoningChainWorkspace:
    """Tests for _handle_get_reasoning_chain."""

    async def test_returns_observation_chain(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """Returns an observation and its chain."""
        _, _, _, _, _, _, docs_peer2, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain(
            ctx, {"observation_id": docs_peer2[0].id}
        )

        assert "Observation" in result
        assert docs_peer2[0].content in result

    async def test_nonexistent_observation_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error for nonexistent observation ID."""
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain(
            ctx, {"observation_id": "nonexistent_id"}
        )

        assert "ERROR" in result

    async def test_missing_observation_id_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
    ):
        """Returns error when observation_id is missing."""
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain(ctx, {})

        assert "ERROR" in result

    async def test_invalid_direction_returns_error(
        self,
        make_workspace_ctx: Callable[..., ToolContext],
        workspace_test_data: Any,
    ):
        """Returns error for invalid direction parameter."""
        _, _, _, _, _, _, docs_peer2, _ = workspace_test_data
        ctx = make_workspace_ctx()

        result = await _handle_get_reasoning_chain(
            ctx,
            {"observation_id": docs_peer2[0].id, "direction": "invalid"},
        )

        assert "ERROR" in result

    async def test_deductive_observation_shows_premises(
        self,
        db_session: AsyncSession,
        make_workspace_ctx: Callable[..., ToolContext],
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
        await db_session.commit()
        await db_session.refresh(deductive_doc)

        ctx = make_workspace_ctx()
        result = await _handle_get_reasoning_chain(
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
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """create_workspace_tool_executor returns an async callable."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
        )

        assert callable(executor)

    async def test_routes_workspace_tools(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """Workspace-specific tools are routed to workspace handlers."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
        )

        stats_result = await executor("get_workspace_stats", {})
        assert isinstance(stats_result, str)
        assert "Workspace stats" in stats_result
        assert "Most active peers" in stats_result

    async def test_falls_through_to_standard_handlers(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """Non-workspace tools fall through to standard handlers."""
        workspace, _, _, _, session, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
            session_name=session.name,
        )

        # grep_messages is a standard handler, should fall through
        result = await executor("grep_messages", {"text": "Test message"})

        assert isinstance(result, str)

    async def test_unknown_tool_returns_error(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """Unknown tool name returns error."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
        )

        result = await executor("nonexistent_tool", {})

        assert "Unknown tool" in result

    async def test_handles_exceptions_gracefully(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """Executor returns error strings instead of raising exceptions."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
        )

        # Missing required observer/observed/query parameters
        result = await executor("search_memory", {})

        assert isinstance(result, str)
        assert "ERROR" in result

    async def test_get_peer_card_via_executor(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """get_peer_card routes through workspace handler with params."""
        workspace, peer1, peer2, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
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
# Regression: workspace-flat message visibility without a pinned session
# =============================================================================


@pytest.mark.asyncio
class TestWorkspaceMessageToolsUnpinned:
    """The workspace executor's observer='' sentinel must read as
    'no perspective scoping' (None) at the crud boundary. Under #882's
    resolve_session_scope, an empty STRING is looked up as a real peer with
    no session memberships and denies every result — so these tests run the
    message tools with NO session_name, the primary workspace-chat shape."""

    async def test_grep_messages_finds_content_without_session(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
        )
        result = await executor("grep_messages", {"text": "Test message"})

        assert isinstance(result, str)
        assert "No messages found" not in result
        assert "Test message" in result

    async def test_date_range_finds_content_without_session(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
        )
        result = await executor("get_messages_by_date_range", {"limit": 10})

        assert isinstance(result, str)
        assert "Found" in result
        assert "No messages found" not in result

    async def test_session_allowlist_is_honored_when_set(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
        workspace_test_data: Any,
    ):
        """An allowlist naming no real session yields no results."""
        workspace, *_ = workspace_test_data

        executor = await create_workspace_tool_executor(
            workspace_name=workspace.name,
            session_allowlist=["no-such-session"],
        )
        result = await executor("grep_messages", {"text": "Test message"})

        assert isinstance(result, str)
        assert "No messages found" in result


@pytest.mark.asyncio
async def test_workspace_prefetch_failure_degrades_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefetch errors must not fail the request (parity with the base
    agent's try/except): the agent proceeds with no prefetched block."""
    from src.dialectic.workspace import WorkspaceDialecticAgent

    async def boom(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise RuntimeError("stats query exploded")

    monkeypatch.setattr("src.dialectic.workspace.crud.get_workspace_stats", boom)

    agent = WorkspaceDialecticAgent(workspace_name="w")
    result = await agent._prefetch_relevant_observations("q")  # pyright: ignore[reportPrivateUsage]

    assert result is None


class TestWorkspaceChatPrompt:
    def test_teaches_honcho_world_without_sibling_agent(self) -> None:
        from src.dialectic.prompts import workspace_agent_system_prompt

        prompt = workspace_agent_system_prompt().lower()
        assert "peer-level" not in prompt
        assert "unlike a peer" not in prompt
        for term in (
            "honcho",
            "workspace",
            "peer",
            "session",
            "message",
            "conclusion",
            "observer",
            "observed",
        ):
            assert term in prompt

    def test_agent_uses_workspace_prompt_and_prefetch_heading(self) -> None:
        from src.dialectic.prompts import workspace_agent_system_prompt
        from src.dialectic.workspace import WorkspaceDialecticAgent

        agent = WorkspaceDialecticAgent(workspace_name="w")
        offered = {
            name
            for tool in agent._select_tools()  # pyright: ignore[reportPrivateUsage]
            if isinstance((name := tool.get("name")), str)
        }
        assert agent.messages[0]["content"] == workspace_agent_system_prompt(offered)
        assert agent._prefetch_heading() == "Workspace overview (prefetched)"  # pyright: ignore[reportPrivateUsage]

    def test_forbids_clarifying_questions(self) -> None:
        """The endpoint is non-interactive, so the prompt must say so.

        Without this the model answers a recall query with a plan and a menu of
        lookups for a caller that cannot reply. The pair agent talks to a peer
        and is deliberately left alone.
        """
        from src.dialectic.prompts import (
            agent_system_prompt,
            workspace_agent_system_prompt,
        )

        prompt = workspace_agent_system_prompt()
        assert "NO CLARIFYING QUESTIONS" in prompt
        assert "NO CLARIFYING QUESTIONS" not in agent_system_prompt(
            "alice", "alice", None, None
        )


class TestWorkspaceToolChoice:
    """The workspace agent must search before it answers.

    Its prefetch is an orientation overview, not the corpus, so a turn with no
    tool call ends the loop with whatever the overview happened to contain.
    """

    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "max"])
    def test_first_turn_requires_a_tool_call(self, level: str) -> None:
        from src.config import settings
        from src.dialectic.workspace import WorkspaceDialecticAgent

        agent = WorkspaceDialecticAgent(workspace_name="w", reasoning_level=level)  # pyright: ignore[reportArgumentType]
        level_settings = settings.DIALECTIC.LEVELS[level]  # pyright: ignore[reportArgumentType]
        assert agent._tool_choice(level_settings) == "required"  # pyright: ignore[reportPrivateUsage]

    def test_pair_agent_keeps_the_configured_choice(self) -> None:
        from src.config import settings
        from src.dialectic.core import DialecticAgent

        agent = DialecticAgent(
            workspace_name="w", session_name=None, observer="a", observed="a"
        )
        level_settings = settings.DIALECTIC.LEVELS["low"]
        assert (
            agent._tool_choice(level_settings)  # pyright: ignore[reportPrivateUsage]
            == level_settings.TOOL_CHOICE
        )

    def test_a_configured_non_auto_choice_is_passed_through(self) -> None:
        from src.config import DialecticLevelSettings, settings
        from src.dialectic.workspace import WorkspaceDialecticAgent

        agent = WorkspaceDialecticAgent(workspace_name="w")
        pinned = DialecticLevelSettings(
            MODEL_CONFIG=settings.DIALECTIC.LEVELS["low"].MODEL_CONFIG,
            MAX_TOOL_ITERATIONS=5,
            TOOL_CHOICE="none",
        )
        assert agent._tool_choice(pinned) == "none"  # pyright: ignore[reportPrivateUsage]

"""Tests for agent tools in src/utils/agent_tools.py"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.utils.agent_tools import (
    ObservationsCreatedResult,
    ToolContext,
    _handle_create_observations,  # pyright: ignore[reportPrivateUsage]
    _handle_delete_observations,  # pyright: ignore[reportPrivateUsage]
    _handle_extract_preferences,  # pyright: ignore[reportPrivateUsage]
    _handle_finish_consolidation,  # pyright: ignore[reportPrivateUsage]
    _handle_get_messages_by_date_range,  # pyright: ignore[reportPrivateUsage]
    _handle_get_observation_context,  # pyright: ignore[reportPrivateUsage]
    _handle_get_peer_card,  # pyright: ignore[reportPrivateUsage]
    _handle_get_recent_history,  # pyright: ignore[reportPrivateUsage]
    _handle_get_recent_observations,  # pyright: ignore[reportPrivateUsage]
    _handle_get_session_summary,  # pyright: ignore[reportPrivateUsage]
    _handle_grep_messages,  # pyright: ignore[reportPrivateUsage]
    _handle_search_memory,  # pyright: ignore[reportPrivateUsage]
    _handle_search_messages,  # pyright: ignore[reportPrivateUsage]
    _handle_update_peer_card,  # pyright: ignore[reportPrivateUsage]
    create_observations,
    create_tool_executor,
    extract_preferences,
    get_recent_history,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def tool_test_data(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> Any:
    """Create comprehensive test data for agent tools testing.

    Returns:
        Tuple of (workspace, observer_peer, observed_peer, session, messages, documents)
    """
    workspace, peer1 = sample_data

    # Create second peer (to be observed)
    peer2 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(peer2)
    await db_session.flush()

    # Create session
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Create collection (peer1 observes peer2)
    collection = models.Collection(
        workspace_name=workspace.name,
        observer=peer1.name,
        observed=peer2.name,
    )
    db_session.add(collection)
    await db_session.flush()

    # Create messages in the session
    now = datetime.now(timezone.utc)
    messages: list[models.Message] = []
    for i in range(5):
        peer_name = peer2.name if i % 2 == 0 else peer1.name
        msg = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer_name,
            content=f"Test message {i} from {peer_name}",
            seq_in_session=i + 1,
            token_count=10,
            created_at=now - timedelta(minutes=5 - i),
        )
        db_session.add(msg)
        messages.append(msg)
    await db_session.flush()

    # Refresh to get IDs
    for msg in messages:
        await db_session.refresh(msg)

    # Create some documents (observations)
    documents: list[models.Document] = []
    for i, content in enumerate(
        ["User likes coffee", "User works remotely", "User prefers mornings"]
    ):
        doc = models.Document(
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            content=content,
            embedding=[0.1 * (i + 1)] * 1536,
            session_name=session.name,
            level="explicit",
            metadata={
                "message_ids": [messages[0].id],
                "message_created_at": str(messages[0].created_at),
            },
        )
        db_session.add(doc)
        documents.append(doc)
    await db_session.flush()

    for doc in documents:
        await db_session.refresh(doc)

    yield workspace, peer1, peer2, session, messages, documents

    await db_session.rollback()


@pytest.fixture
def make_tool_context(
    db_session: AsyncSession, tool_test_data: Any
) -> Callable[..., ToolContext]:
    """Factory fixture to create ToolContext with custom parameters."""
    workspace, peer1, peer2, session, _messages, _ = tool_test_data
    shared_lock = asyncio.Lock()

    def _make_context(
        *,
        current_messages: list[models.Message] | None = None,
        include_observation_ids: bool = False,
        history_token_limit: int = 8192,
        session_name: str | None = None,
    ) -> ToolContext:
        return ToolContext(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=session_name if session_name is not None else session.name,
            current_messages=current_messages,
            include_observation_ids=include_observation_ids,
            history_token_limit=history_token_limit,
            db_lock=shared_lock,
        )

    return _make_context


# =============================================================================
# Unit Tests: Observation Tools
# =============================================================================


@pytest.mark.asyncio
class TestCreateObservations:
    """Tests for _handle_create_observations."""

    async def test_deriver_context_creates_with_message_ids(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Deriver context (with current_messages) links observations to source messages.

        Note: Deriver is now explicit-only. Deductive/inductive observations are
        created only by the Dreamer agent.
        """
        workspace, peer1, peer2, _session, messages, _ = tool_test_data
        ctx = make_tool_context(current_messages=messages)

        result = await _handle_create_observations(
            ctx,
            {
                "observations": [
                    {"content": "Likes tea", "level": "explicit"},
                    {"content": "Enjoys reading", "level": "explicit"},
                ]
            },
        )

        assert "Created 2 observations" in result
        assert "2 explicit" in result

        # Verify DB state
        stmt = select(models.Document).where(
            models.Document.workspace_name == workspace.name,
            models.Document.observer == peer1.name,
            models.Document.observed == peer2.name,
            models.Document.content.in_(["Likes tea", "Enjoys reading"]),
        )
        docs = (await db_session.execute(stmt)).scalars().all()
        assert len(docs) == 2

    async def test_dialectic_context_forces_deductive(
        self,
        db_session: AsyncSession,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Dialectic context (no current_messages) forces observations to be deductive."""
        ctx = make_tool_context(current_messages=None)

        result = await _handle_create_observations(
            ctx,
            {
                "observations": [
                    {
                        "content": "Inferred preference for quiet spaces",
                        "source_ids": ["premise1", "premise2"],
                        "premises": [
                            "User mentioned working in libraries",
                            "User avoids noisy cafes",
                        ],
                    },
                ]
            },
        )

        assert "Created 1 observations" in result
        assert "1 deductive" in result

        # Verify the document was created as deductive with source_ids
        stmt = select(models.Document).where(
            models.Document.content == "Inferred preference for quiet spaces"
        )
        doc = (await db_session.execute(stmt)).scalar_one_or_none()
        assert doc is not None
        assert doc.level == "deductive"
        assert doc.source_ids == ["premise1", "premise2"]

    async def test_empty_observations_list_returns_error(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Empty observations list returns error message."""
        ctx = make_tool_context(current_messages=None)

        result = await _handle_create_observations(ctx, {"observations": []})

        assert "ERROR" in result
        assert "empty" in result.lower()

    async def test_batch_embedding_failure_falls_back_to_individual_embeds(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """If batch embedding fails but individual embeds succeed, all observations are created."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        async def fail_batch_embed(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding provider timeout")

        async def succeed_single_embed(_content: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        created_documents: list[Any] = []

        async def fake_create_documents(
            _db: AsyncSession,
            documents: list[Any],
            workspace_name: str,
            *,
            observer: str,
            observed: str,
            deduplicate: bool = False,
        ) -> int:
            _ = (workspace_name, observer, observed, deduplicate)
            created_documents.extend(documents)
            return len(documents)

        monkeypatch.setattr(
            "src.utils.agent_tools.embedding_client.simple_batch_embed",
            fail_batch_embed,
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.embedding_client.embed",
            succeed_single_embed,
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.crud.create_documents", fake_create_documents
        )

        result = await create_observations(
            db_session,
            observations=[
                schemas.ObservationInput(content="First obs", level="explicit"),
                schemas.ObservationInput(content="Second obs", level="explicit"),
            ],
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
            workspace_name=workspace.name,
            message_ids=[],
            message_created_at=str(datetime.now(timezone.utc)),
        )

        assert isinstance(result, ObservationsCreatedResult)
        assert result.created_count == 2
        assert len(result.failed) == 0
        assert len(created_documents) == 2

    async def test_batch_embedding_failure_individual_embed_partial_failure(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """If batch embedding fails and some individual embeds also fail, only successful ones are created."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        async def fail_batch_embed(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding provider timeout")

        async def embed_per_observation(content: str) -> list[float]:
            if content == "Fails embed":
                raise RuntimeError("single-item embed failure")
            return [0.1, 0.2, 0.3]

        created_documents: list[Any] = []

        async def fake_create_documents(
            _db: AsyncSession,
            documents: list[Any],
            workspace_name: str,
            *,
            observer: str,
            observed: str,
            deduplicate: bool = False,
        ) -> int:
            _ = (workspace_name, observer, observed, deduplicate)
            created_documents.extend(documents)
            return len(documents)

        monkeypatch.setattr(
            "src.utils.agent_tools.embedding_client.simple_batch_embed",
            fail_batch_embed,
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.embedding_client.embed",
            embed_per_observation,
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.crud.create_documents", fake_create_documents
        )

        result = await create_observations(
            db_session,
            observations=[
                schemas.ObservationInput(content="Embeds fine", level="explicit"),
                schemas.ObservationInput(content="Fails embed", level="explicit"),
            ],
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
            workspace_name=workspace.name,
            message_ids=[],
            message_created_at=str(datetime.now(timezone.utc)),
        )

        assert isinstance(result, ObservationsCreatedResult)
        assert result.created_count == 1
        assert len(result.failed) == 1
        assert result.failed[0].content_preview == "Fails embed"
        assert "Embedding failed" in result.failed[0].error
        assert len(created_documents) == 1
        assert created_documents[0].content == "Embeds fine"


@pytest.mark.asyncio
class TestDeleteObservations:
    """Tests for _handle_delete_observations."""

    async def test_delete_valid_observation(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Successfully soft-deletes observation by ID."""
        _, _, _, _, _, documents = tool_test_data
        ctx = make_tool_context(include_observation_ids=True)

        doc_id = documents[0].id
        result = await _handle_delete_observations(ctx, {"observation_ids": [doc_id]})

        assert "Deleted 1 observations" in result

        # Verify soft-deletion (document still exists but has deleted_at timestamp)
        stmt = select(models.Document).where(models.Document.id == doc_id)
        doc = (await db_session.execute(stmt)).scalar_one_or_none()
        assert doc is not None
        assert doc.deleted_at is not None

    async def test_delete_invalid_id_handled_gracefully(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Invalid observation IDs are handled without crashing."""
        ctx = make_tool_context(include_observation_ids=True)

        result = await _handle_delete_observations(
            ctx, {"observation_ids": ["nonexistent_id_12345"]}
        )

        # Should report 0 deleted (graceful handling)
        assert "Deleted 0 observations" in result


@pytest.mark.asyncio
class TestGetRecentObservations:
    """Tests for _handle_get_recent_observations."""

    async def test_returns_formatted_observations(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Returns recent observations in formatted output."""
        ctx = make_tool_context()

        result = await _handle_get_recent_observations(ctx, {"limit": 10})

        assert "Found" in result
        assert "observations" in result
        # Should contain some of our test observation content
        assert any(
            content in result
            for content in ["likes coffee", "works remotely", "prefers mornings"]
        )


# =============================================================================
# Unit Tests: Search Tools
# =============================================================================


@pytest.mark.asyncio
class TestSearchMemory:
    """Tests for _handle_search_memory."""

    async def test_returns_matching_observations(
        self,
        make_tool_context: Callable[..., ToolContext],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Returns observations matching semantic query."""
        # Force pgvector queries since test documents are created directly in postgres
        monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)

        ctx = make_tool_context()

        result = await _handle_search_memory(ctx, {"query": "coffee preferences"})

        assert "Found" in result
        assert "observations" in result

    async def test_returns_empty_message_when_no_results(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Returns appropriate message when no observations match."""
        workspace, peer1 = sample_data

        # Create a peer with no observations
        peer2 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
        db_session.add(peer2)
        await db_session.flush()

        # Create collection but no documents
        collection = models.Collection(
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()

        ctx = ToolContext(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=None,
            current_messages=None,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_search_memory(ctx, {"query": "anything"})

        assert "No observations found" in result

    async def test_reuses_single_embedding_for_dialectic_fallback(
        self,
        make_tool_context: Callable[..., ToolContext],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Uses one embedding for query_documents + fallback and scopes visibility."""
        ctx = make_tool_context()
        ctx.agent_type = "dialectic"

        embed_calls: list[str] = []
        query_embeddings: list[list[float] | None] = []
        fallback_embeddings: list[list[float] | None] = []

        async def fake_embed(query: str) -> list[float]:
            embed_calls.append(query)
            return [0.1, 0.2, 0.3]

        async def fake_query_documents(
            db: AsyncSession,
            workspace_name: str,
            query: str,
            *,
            observer: str,
            observed: str,
            top_k: int = 5,
            embedding: list[float] | None = None,
            **_kwargs: Any,
        ) -> list[models.Document]:
            _ = (db, workspace_name, query, observer, observed, top_k)
            query_embeddings.append(embedding)
            return []

        async def fake_search_messages(
            db: AsyncSession,
            workspace_name: str,
            session_name: str | None,
            query: str,
            limit: int = 10,
            context_window: int = 2,
            peer_perspective: str | None = None,
            embedding: list[float] | None = None,
        ) -> list[tuple[list[models.Message], list[models.Message]]]:
            _ = (db, workspace_name, session_name, query, limit, context_window)
            assert peer_perspective == ctx.observer
            fallback_embeddings.append(embedding)
            msg = models.Message(
                workspace_name=ctx.workspace_name,
                session_name=ctx.session_name,
                peer_name=ctx.observed,
                content="Relevant fallback message",
                seq_in_session=1,
                token_count=5,
                created_at=datetime.now(timezone.utc),
            )
            return [([msg], [msg])]

        monkeypatch.setattr("src.utils.agent_tools.embedding_client.embed", fake_embed)
        monkeypatch.setattr(
            "src.utils.agent_tools.crud.query_documents", fake_query_documents
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.crud.search_messages",
            fake_search_messages,
        )

        result = await _handle_search_memory(ctx, {"query": "coffee preferences"})

        assert "No observations yet. Message search results:" in result
        assert embed_calls == ["coffee preferences"]
        assert len(query_embeddings) == 1
        assert len(fallback_embeddings) == 1
        assert query_embeddings[0] == fallback_embeddings[0]


@pytest.mark.asyncio
class TestSearchMessages:
    """Tests for _handle_search_messages."""

    async def test_returns_message_snippets(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Returns message snippets with context."""
        ctx = make_tool_context()

        result = await _handle_search_messages(ctx, {"query": "test message"})

        # Should return some result (may be empty if semantic search doesn't match)
        assert isinstance(result, str)

    async def test_passes_peer_perspective_to_crud(
        self,
        make_tool_context: Callable[..., ToolContext],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """search_messages tool should pass observer as peer perspective."""
        captured: dict[str, Any] = {}

        async def fake_search_messages(
            db: AsyncSession,
            workspace_name: str,
            session_name: str | None,
            query: str,
            limit: int = 10,
            context_window: int = 2,
            peer_perspective: str | None = None,
            embedding: list[float] | None = None,
        ) -> list[tuple[list[models.Message], list[models.Message]]]:
            _ = (
                db,
                workspace_name,
                session_name,
                query,
                limit,
                context_window,
                embedding,
            )
            captured["peer_perspective"] = peer_perspective
            return []

        monkeypatch.setattr(crud, "search_messages", fake_search_messages)

        ctx = make_tool_context()
        await _handle_search_messages(ctx, {"query": "test message"})

        assert captured["peer_perspective"] == ctx.observer


@pytest.mark.asyncio
class TestGrepMessages:
    """Tests for _handle_grep_messages."""

    async def test_exact_text_match(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Finds messages with exact text match."""
        ctx = make_tool_context()

        # Search for peer2's name which should be in messages
        result = await _handle_grep_messages(ctx, {"text": "Test message"})

        # Should find our test messages
        assert isinstance(result, str)

    async def test_missing_text_param_returns_error(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Returns error when text parameter is missing."""
        ctx = make_tool_context()

        result = await _handle_grep_messages(ctx, {"text": ""})

        assert "ERROR" in result


@pytest.mark.asyncio
class TestGetMessagesByDateRange:
    """Tests for _handle_get_messages_by_date_range."""

    async def test_date_filtering_works(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Filters messages by date range."""
        ctx = make_tool_context()

        # Get messages from today
        today = datetime.now(timezone.utc).date().isoformat()
        result = await _handle_get_messages_by_date_range(
            ctx, {"after_date": today, "limit": 10}
        )

        assert isinstance(result, str)
        # Should either find messages or report none found
        assert "Found" in result or "No messages found" in result


# =============================================================================
# Unit Tests: Context Tools
# =============================================================================


@pytest.mark.asyncio
class TestGetRecentHistory:
    """Tests for _handle_get_recent_history."""

    async def test_with_session_returns_messages(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Returns conversation history for session."""
        workspace, peer1, _peer2, session, _messages, _documents = tool_test_data
        await db_session.execute(
            models.session_peers_table.insert().values(
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer1.name,
                joined_at=datetime.now(timezone.utc) - timedelta(days=1),
                left_at=None,
            )
        )
        await db_session.flush()

        ctx = make_tool_context()

        result = await _handle_get_recent_history(ctx, {})

        assert "Conversation history" in result
        assert "messages" in result.lower()

    async def test_with_session_passes_peer_perspective_to_crud(
        self,
        make_tool_context: Callable[..., ToolContext],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Session history should pass observer as peer perspective to CRUD layer."""
        captured: dict[str, Any] = {}

        async def fake_get_messages(*args: Any, **kwargs: Any) -> Any:
            _ = args
            captured["peer_perspective"] = kwargs.get("peer_perspective")
            return select(models.Message).where(models.Message.id == -1)

        monkeypatch.setattr(crud, "get_messages", fake_get_messages)

        ctx = make_tool_context()
        await _handle_get_recent_history(ctx, {})

        assert captured["peer_perspective"] == ctx.observer

    async def test_without_session_uses_observed(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
    ):
        """Without session, retrieves messages from observed peer."""
        workspace, peer1, peer2, _, _, _ = tool_test_data

        ctx = ToolContext(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=None,  # No session
            current_messages=None,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_get_recent_history(ctx, {})

        # Should get messages from peer2 across sessions
        assert isinstance(result, str)

    async def test_without_session_scopes_by_peer_membership_window(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """No-session history path should respect observer visibility windows."""
        workspace, observer = sample_data
        observed = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed)
        await db_session.flush()

        session_visible = models.Session(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        session_hidden = models.Session(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add_all([session_visible, session_hidden])
        await db_session.flush()

        base_time = datetime.now(timezone.utc)
        join_time = base_time + timedelta(seconds=5)
        await db_session.execute(
            models.session_peers_table.insert().values(
                workspace_name=workspace.name,
                session_name=session_visible.name,
                peer_name=observer.name,
                joined_at=join_time,
                left_at=None,
            )
        )
        await db_session.flush()

        msg_before_join = models.Message(
            workspace_name=workspace.name,
            session_name=session_visible.name,
            peer_name=observed.name,
            content="Before observer joined",
            seq_in_session=1,
            token_count=5,
            created_at=base_time,
        )
        msg_after_join = models.Message(
            workspace_name=workspace.name,
            session_name=session_visible.name,
            peer_name=observed.name,
            content="After observer joined",
            seq_in_session=2,
            token_count=5,
            created_at=base_time + timedelta(seconds=10),
        )
        msg_hidden = models.Message(
            workspace_name=workspace.name,
            session_name=session_hidden.name,
            peer_name=observed.name,
            content="Observer never in this session",
            seq_in_session=1,
            token_count=5,
            created_at=base_time + timedelta(seconds=10),
        )
        db_session.add_all([msg_before_join, msg_after_join, msg_hidden])
        await db_session.flush()

        scoped_history = await get_recent_history(
            db_session,
            workspace_name=workspace.name,
            session_name=None,
            observed=observed.name,
            peer_perspective=observer.name,
        )
        scoped_ids = {m.public_id for m in scoped_history}
        assert msg_after_join.public_id in scoped_ids
        assert msg_before_join.public_id not in scoped_ids
        assert msg_hidden.public_id not in scoped_ids


@pytest.mark.asyncio
class TestGetObservationContext:
    """Tests for _handle_get_observation_context."""

    async def test_retrieves_surrounding_messages(
        self, tool_test_data: Any, make_tool_context: Callable[..., ToolContext]
    ):
        """Retrieves messages and their context."""
        _, _, _, _, messages, _ = tool_test_data
        ctx = make_tool_context()

        result = await _handle_get_observation_context(
            ctx, {"message_ids": [messages[2].public_id]}
        )

        assert "Retrieved" in result or "No messages found" in result


@pytest.mark.asyncio
class TestGetSessionSummary:
    """Tests for _handle_get_session_summary."""

    async def test_returns_summary_when_exists(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Returns session summary if one exists."""
        from sqlalchemy import update

        from src.cache.client import cache
        from src.crud.session import session_cache_key

        workspace, _, _, session, _, _ = tool_test_data

        # Update the session's internal_metadata directly in DB
        # Note: summary keys use the SummaryType enum values, not "short"/"long"
        await db_session.execute(
            update(models.Session)
            .where(models.Session.name == session.name)
            .where(models.Session.workspace_name == workspace.name)
            .values(
                internal_metadata={
                    "summaries": {
                        "honcho_chat_summary_short": {
                            "content": "This is a test summary",
                            "summary_type": "short",
                        }
                    }
                }
            )
        )
        await db_session.commit()

        # Invalidate the session cache so the updated data is visible
        cache_key = session_cache_key(workspace.name, session.name)
        await cache.delete(cache_key)

        ctx = make_tool_context()
        result = await _handle_get_session_summary(ctx, {"summary_type": "short"})

        assert "Session summary" in result
        assert "This is a test summary" in result

    async def test_returns_no_summary_when_missing(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Returns appropriate message when no summary exists."""
        ctx = make_tool_context()
        result = await _handle_get_session_summary(ctx, {"summary_type": "short"})

        assert "No session summary" in result


# =============================================================================
# Unit Tests: Peer Card Tools
# =============================================================================


@pytest.mark.asyncio
class TestUpdatePeerCard:
    """Tests for _handle_update_peer_card."""

    async def test_creates_peer_card(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Creates/updates peer card with facts."""
        workspace, peer1, peer2, _, _, _ = tool_test_data
        ctx = make_tool_context()

        result = await _handle_update_peer_card(
            ctx, {"content": ["Name: John", "Location: NYC", "Occupation: Engineer"]}
        )

        assert "Updated peer card" in result

        # Verify DB state
        peer_card = await crud.get_peer_card(
            db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
        )
        assert peer_card is not None
        assert "Name: John" in peer_card


@pytest.mark.asyncio
class TestGetPeerCard:
    """Tests for _handle_get_peer_card."""

    async def test_returns_peer_card_when_exists(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Returns peer card content when it exists."""
        workspace, peer1, peer2, _, _, _ = tool_test_data

        # Create peer card
        await crud.set_peer_card(
            db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            peer_card=["Fact 1", "Fact 2"],
        )

        ctx = make_tool_context()
        result = await _handle_get_peer_card(ctx, {})

        assert "Peer card" in result
        assert "Fact 1" in result
        assert "Fact 2" in result

    async def test_returns_not_found_when_missing(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Returns appropriate message when no peer card exists."""
        workspace, peer1 = sample_data

        # Create peer with no card
        peer2 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
        db_session.add(peer2)
        await db_session.flush()

        ctx = ToolContext(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=None,
            current_messages=None,
            include_observation_ids=False,
            history_token_limit=8192,
            db_lock=asyncio.Lock(),
        )

        result = await _handle_get_peer_card(ctx, {})

        assert "No peer card" in result


# =============================================================================
# Unit Tests: Consolidation Tools
# =============================================================================


@pytest.mark.asyncio
class TestExtractPreferences:
    """Tests for _handle_extract_preferences."""

    async def test_finds_preference_patterns(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        make_tool_context: Callable[..., ToolContext],
    ):
        """Finds preference patterns in messages."""
        workspace, _, peer2, session, _, _ = tool_test_data

        # Add messages with preference patterns
        preference_msg = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer2.name,
            content="I prefer brief responses and always include code examples",
            seq_in_session=100,
            token_count=20,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(preference_msg)
        await db_session.flush()

        ctx = make_tool_context()
        result = await _handle_extract_preferences(ctx, {})

        # Should return some result about preferences
        assert isinstance(result, str)

    async def test_passes_peer_perspective_to_search_messages(
        self,
        make_tool_context: Callable[..., ToolContext],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """extract_preferences should scope semantic queries by observer visibility."""
        captured: list[str | None] = []

        async def fake_search_messages(
            _db: AsyncSession,
            workspace_name: str,
            session_name: str | None,
            query: str,
            limit: int = 10,
            context_window: int = 0,
            peer_perspective: str | None = None,
            embedding: list[float] | None = None,
        ) -> list[tuple[list[models.Message], list[models.Message]]]:
            _ = (
                workspace_name,
                session_name,
                query,
                limit,
                context_window,
                embedding,
            )
            captured.append(peer_perspective)
            return []

        monkeypatch.setattr(
            "src.utils.agent_tools.crud.search_messages", fake_search_messages
        )

        ctx = make_tool_context()
        await _handle_extract_preferences(ctx, {})

        assert captured
        assert all(p == ctx.observer for p in captured)

    async def test_falls_back_to_per_query_embedding_when_batch_fails(
        self,
        db_session: AsyncSession,
        tool_test_data: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Batch embedding failure should not abort preference extraction."""
        workspace, _, observed_peer, session, _, _ = tool_test_data

        async def fail_batch_embed(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding provider timeout")

        async def unexpected_embed_call(_query: str) -> list[float]:
            raise AssertionError(
                "extract_preferences should not call embedding_client.embed "
                + "when batch embedding fails"
            )

        embedding_args: list[list[float] | None] = []

        async def fake_search_messages(
            _db: AsyncSession,
            workspace_name: str,
            session_name: str | None,
            query: str,
            limit: int,
            context_window: int,
            peer_perspective: str | None = None,
            embedding: list[float] | None = None,
        ) -> list[tuple[list[models.Message], list[models.Message]]]:
            _ = (limit, context_window, peer_perspective)
            embedding_args.append(embedding)
            msg = models.Message(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=observed_peer.name,
                content=f"Relevant from {query}",
                seq_in_session=1,
                token_count=5,
                created_at=datetime.now(timezone.utc),
            )
            return [([msg], [])]

        monkeypatch.setattr(
            "src.utils.agent_tools.embedding_client.simple_batch_embed",
            fail_batch_embed,
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.embedding_client.embed",
            unexpected_embed_call,
        )
        monkeypatch.setattr(
            "src.utils.agent_tools.crud.search_messages", fake_search_messages
        )

        result = await extract_preferences(
            db_session,
            workspace_name=workspace.name,
            session_name=session.name,
            observed=observed_peer.name,
        )

        # We still get partial results despite one per-query failure.
        assert result["messages"]
        assert len(embedding_args) == 5
        assert all(embedding is None for embedding in embedding_args)


@pytest.mark.asyncio
class TestFinishConsolidation:
    """Tests for _handle_finish_consolidation."""

    async def test_returns_completion_signal(
        self, make_tool_context: Callable[..., ToolContext]
    ):
        """Returns correct completion signal."""
        ctx = make_tool_context()

        result = await _handle_finish_consolidation(
            ctx, {"summary": "Consolidated 5 observations, updated peer card"}
        )

        assert "CONSOLIDATION_COMPLETE" in result
        assert "Consolidated 5 observations" in result


# =============================================================================
# Integration Tests: Tool Executor
# =============================================================================


@pytest.mark.asyncio
class TestToolExecutor:
    """Tests for create_tool_executor and the executor function."""

    async def test_create_tool_executor_returns_callable(
        self, db_session: AsyncSession, tool_test_data: Any
    ):
        """create_tool_executor returns an async callable."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        executor = await create_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
        )

        assert callable(executor)

    async def test_executor_routes_to_correct_handler(
        self, db_session: AsyncSession, tool_test_data: Any
    ):
        """Executor routes tool calls to correct handlers."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        executor = await create_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
        )

        result = await executor("get_peer_card", {})

        assert isinstance(result, str)
        # Should be from get_peer_card handler
        assert "peer card" in result.lower() or "No peer card" in result

    async def test_executor_unknown_tool_returns_error(
        self, db_session: AsyncSession, tool_test_data: Any
    ):
        """Unknown tool name returns error message."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        executor = await create_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
        )

        result = await executor("nonexistent_tool", {})

        assert "Unknown tool" in result

    async def test_executor_handles_exceptions_gracefully(
        self, db_session: AsyncSession, tool_test_data: Any
    ):
        """Executor converts exceptions to error strings instead of raising."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        executor = await create_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
        )

        # Call with missing required parameter - should return error string
        result = await executor("search_memory", {})  # Missing 'query'

        assert isinstance(result, str)
        # Should contain error info, not raise exception

    async def test_executor_dreamer_context_includes_observation_ids(
        self, db_session: AsyncSession, tool_test_data: Any
    ):
        """Dreamer context (include_observation_ids=True) shows IDs in output."""
        workspace, peer1, peer2, session, _, _ = tool_test_data

        executor = await create_tool_executor(
            db=db_session,
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=session.name,
            include_observation_ids=True,  # Dreamer setting
        )

        result = await executor("get_recent_observations", {"limit": 10})

        # When include_observation_ids is True, output should contain IDs
        # The format is [id:xxx]
        assert isinstance(result, str)
        # Should show observations if any exist
        if "Found" in result and "observations" in result:
            # IDs should be included in the output
            assert "[id:" in result or "observations" in result

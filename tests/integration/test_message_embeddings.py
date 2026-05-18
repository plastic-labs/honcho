"""
Tests for message embedding functionality.

These tests verify that message embeddings are created, stored, and can be searched.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud import create_messages
from src.crud import message as message_crud
from src.models import Message, Peer, Workspace
from src.schemas import MessageCreate
from src.utils.search import search


class _FakeScalarResult:
    def __init__(self, rows: list[models.Message]):
        self._rows: list[Message] = rows

    def all(self) -> list[models.Message]:
        return self._rows


class _FakeResult:
    def __init__(self, rows: list[models.Message]):
        self._rows: list[Message] = rows

    def scalars(self) -> _FakeScalarResult:
        return _FakeScalarResult(self._rows)


class _CountingDb:
    def __init__(self, rows: list[models.Message]):
        self._rows: list[Message] = rows
        self.execute_count: int = 0

    async def execute(self, _stmt: Any) -> _FakeResult:
        self.execute_count += 1
        return _FakeResult(self._rows)


def _message(session_name: str, seq_in_session: int) -> models.Message:
    return models.Message(
        workspace_name="workspace",
        session_name=session_name,
        peer_name="peer",
        content=f"{session_name}:{seq_in_session}",
        public_id=generate_nanoid(),
        seq_in_session=seq_in_session,
        token_count=1,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_message_embedding_created_when_setting_enabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that MessageEmbedding is created when EMBED_MESSAGES setting is True"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.EMBED_MESSAGES", True)

    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create a message using the CRUD function directly
    test_message_content = "This is a test message for embedding"
    messages = [
        MessageCreate(
            content=test_message_content,
            peer_id=test_peer.name,
            metadata={"test": "embedding_enabled"},
        )
    ]

    created_messages = await create_messages(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    assert len(created_messages) == 1
    created_message = created_messages[0]

    # Query the MessageEmbedding table to verify an embedding was created
    stmt = select(models.MessageEmbedding).where(
        models.MessageEmbedding.message_id == created_message.public_id
    )
    result = await db_session.execute(stmt)
    embedding_record = result.scalar_one_or_none()

    # Verify the embedding record was created (embedding vectors are now stored externally)
    assert embedding_record is not None
    assert embedding_record.message_id == created_message.public_id
    assert embedding_record.content == test_message_content
    assert embedding_record.workspace_name == test_workspace.name
    assert embedding_record.session_name == test_session.name
    assert embedding_record.peer_name == test_peer.name


@pytest.mark.asyncio
async def test_blank_messages_are_not_sent_for_embedding(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    mock_openai_embeddings: dict[str, Any],
):
    """Blank messages should be persisted but excluded from embedding batches."""
    monkeypatch.setattr("src.config.settings.EMBED_MESSAGES", True)

    test_workspace, test_peer = sample_data

    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    blank_content = "   "
    nonblank_content = "This message should be embedded"
    messages = [
        MessageCreate(
            content=blank_content,
            peer_id=test_peer.name,
            metadata={"test": "blank_embedding"},
        ),
        MessageCreate(
            content=nonblank_content,
            peer_id=test_peer.name,
            metadata={"test": "blank_embedding"},
        ),
    ]

    created_messages = await create_messages(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    assert [message.content for message in created_messages] == [
        blank_content,
        nonblank_content,
    ]

    mock_openai_embeddings["batch_embed"].assert_awaited_once()
    batch_arg = mock_openai_embeddings["batch_embed"].await_args.args[0]
    assert batch_arg == {created_messages[1].public_id: nonblank_content}

    stmt = select(models.MessageEmbedding).where(
        models.MessageEmbedding.message_id.in_(
            [message.public_id for message in created_messages]
        )
    )
    result = await db_session.execute(stmt)
    embedding_records = list(result.scalars().all())

    assert len(embedding_records) == 1
    assert embedding_records[0].message_id == created_messages[1].public_id
    assert embedding_records[0].content == nonblank_content


@pytest.mark.asyncio
async def test_message_embedding_not_created_when_setting_disabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that MessageEmbedding is NOT created when EMBED_MESSAGES setting is False"""
    # Monkeypatch the setting to disable message embeddings
    monkeypatch.setattr("src.config.settings.EMBED_MESSAGES", False)

    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create a message using the CRUD function directly
    test_message_content = "This is a test message without embedding"
    messages = [
        MessageCreate(
            content=test_message_content,
            peer_id=test_peer.name,
            metadata={"test": "embedding_disabled"},
        )
    ]

    created_messages = await create_messages(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    assert len(created_messages) == 1
    created_message = created_messages[0]

    # Query the MessageEmbedding table to verify NO embedding was created
    stmt = select(models.MessageEmbedding).where(
        models.MessageEmbedding.message_id == created_message.public_id
    )
    result = await db_session.execute(stmt)
    embedding_record = result.scalar_one_or_none()

    # Verify no embedding was created
    assert embedding_record is None


@pytest.mark.asyncio
async def test_multiple_message_embeddings_created_when_setting_enabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that multiple MessageEmbeddings are created for batch message creation"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.EMBED_MESSAGES", True)

    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    # Create multiple messages
    messages = [
        MessageCreate(
            content="First test message",
            peer_id=test_peer.name,
            metadata={"order": 1},
        ),
        MessageCreate(
            content="Second test message",
            peer_id=test_peer.name,
            metadata={"order": 2},
        ),
    ]

    created_messages = await create_messages(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    assert len(created_messages) == 2

    # Query the MessageEmbedding table to verify embeddings were created for both messages
    for i, created_message in enumerate(created_messages):
        stmt = select(models.MessageEmbedding).where(
            models.MessageEmbedding.message_id == created_message.public_id
        )
        result = await db_session.execute(stmt)
        embedding_record = result.scalar_one_or_none()

        # Verify the embedding record was created (embedding vectors are now stored externally)
        assert embedding_record is not None
        assert embedding_record.message_id == created_message.public_id
        assert embedding_record.content == messages[i].content
        assert embedding_record.workspace_name == test_workspace.name
        assert embedding_record.session_name == test_session.name
        assert embedding_record.peer_name == test_peer.name


@pytest.mark.asyncio
async def test_semantic_search_when_embeddings_enabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    mock_openai_embeddings: dict[str, Any],
):
    """Test that search uses semantic search by default when EMBED_MESSAGES is True"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.EMBED_MESSAGES", True)

    test_workspace, test_peer = sample_data

    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message_content = (
        "I love programming with Python and building web applications"
    )
    messages = [
        MessageCreate(
            content=test_message_content,
            peer_id=test_peer.name,
            metadata={"test": "semantic_search"},
        )
    ]

    created_messages = await create_messages(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    assert len(created_messages) == 1
    created_message = created_messages[0]

    # Verify the embedding was created
    stmt = select(models.MessageEmbedding).where(
        models.MessageEmbedding.message_id == created_message.public_id
    )
    result = await db_session.execute(stmt)
    embedding_record = result.scalar_one_or_none()
    assert embedding_record is not None

    # Now test semantic search without explicitly setting semantic=True
    # This should use semantic search because EMBED_MESSAGES is True
    search_query = (
        "Python development and web apps"  # Similar meaning to the message content
    )

    # Check the call count before search
    initial_call_count: int = mock_openai_embeddings["embed"].call_count

    search_results = await search(
        search_query,
        filters={
            "workspace_id": test_workspace.name,
            "session_id": test_session.name,
        },
    )

    # Verify that the embed method was called during search - e.g. we used semantic search
    assert mock_openai_embeddings["embed"].call_count == initial_call_count + 1

    # Verify that our message was found via semantic search
    assert len(search_results) > 0
    found_message_ids = [msg.public_id for msg in search_results]
    assert created_message.public_id in found_message_ids


@pytest.mark.asyncio
async def test_build_merged_snippets_batches_context_query_across_sessions():
    """Context expansion should not issue one DB query per matched session."""
    matched_messages = [
        _message("session_a", 10),
        _message("session_b", 20),
        _message("session_c", 30),
    ]
    context_messages = [
        _message("session_a", 9),
        _message("session_a", 10),
        _message("session_a", 11),
        _message("session_a", 99),
        _message("session_b", 19),
        _message("session_b", 20),
        _message("session_b", 21),
        _message("session_c", 29),
        _message("session_c", 30),
        _message("session_c", 31),
    ]
    db = _CountingDb(context_messages)

    snippets = await message_crud._build_merged_snippets(  # pyright: ignore[reportPrivateUsage]
        db,  # pyright: ignore[reportArgumentType]
        workspace_name="workspace",
        matched_messages=matched_messages,
        context_window=1,
    )

    assert db.execute_count == 1
    assert [len(matches) for matches, _ in snippets] == [1, 1, 1]
    assert [
        [msg.content for msg in context_messages] for _, context_messages in snippets
    ] == [
        ["session_a:9", "session_a:10", "session_a:11"],
        ["session_b:19", "session_b:20", "session_b:21"],
        ["session_c:29", "session_c:30", "session_c:31"],
    ]


@pytest.mark.asyncio
async def test_search_messages_external_lookup_happens_before_tracked_db(
    monkeypatch: pytest.MonkeyPatch,
):
    """External semantic lookup should finish before opening tracked_db."""
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "external")

    call_order: list[str] = []
    message = models.Message(
        workspace_name="workspace",
        session_name="session",
        peer_name="peer",
        content="Relevant external search result",
        seq_in_session=1,
        token_count=5,
        created_at=datetime.now(timezone.utc),
    )

    class FakeDb:
        def expunge(self, _obj: object) -> None:
            call_order.append("expunge")

    fake_db = FakeDb()

    async def fake_search_messages_external(
        workspace_name: str,
        query_embedding: list[float],
        limit: int,
        *,
        session_name: str | None = None,
        allowed_session_names: list[str] | None = None,
        after_date: datetime | None = None,
        before_date: datetime | None = None,
    ) -> list[str]:
        _ = (
            workspace_name,
            query_embedding,
            limit,
            session_name,
            allowed_session_names,
            after_date,
            before_date,
        )
        call_order.append("external")
        return ["message-1"]

    async def fake_fetch_messages_by_ids(
        db: FakeDb,
        workspace_name: str,
        message_ids: list[str],
        *,
        after_date: datetime | None = None,
        before_date: datetime | None = None,
    ) -> list[models.Message]:
        _ = (workspace_name, message_ids, after_date, before_date)
        assert db is fake_db
        call_order.append("fetch")
        return [message]

    async def fake_build_merged_snippets(
        db: FakeDb,
        workspace_name: str,
        matched_messages: list[models.Message],
        context_window: int,
    ) -> list[tuple[list[models.Message], list[models.Message]]]:
        _ = (workspace_name, context_window)
        assert db is fake_db
        assert matched_messages == [message]
        call_order.append("build")
        return [([message], [message])]

    @asynccontextmanager
    async def fake_tracked_db(_operation_name: str | None = None):
        call_order.append("enter")
        yield fake_db
        call_order.append("exit")

    monkeypatch.setattr(
        message_crud, "_search_messages_external", fake_search_messages_external
    )
    monkeypatch.setattr(
        message_crud, "_fetch_messages_by_ids", fake_fetch_messages_by_ids
    )
    monkeypatch.setattr(
        message_crud, "_build_merged_snippets", fake_build_merged_snippets
    )
    monkeypatch.setattr(message_crud, "tracked_db", fake_tracked_db)

    snippets = await message_crud.search_messages(
        workspace_name="workspace",
        session_name="session",
        query="relevant query",
        embedding=[0.1, 0.2, 0.3],
    )

    assert snippets == [([message], [message])]
    assert call_order.index("external") < call_order.index("enter")


@pytest.mark.asyncio
async def test_search_messages_temporal_external_lookup_happens_before_tracked_db(
    monkeypatch: pytest.MonkeyPatch,
):
    """Temporal external semantic lookup should finish before opening tracked_db."""
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "external")

    call_order: list[str] = []
    after_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    before_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
    message = models.Message(
        workspace_name="workspace",
        session_name="session",
        peer_name="peer",
        content="Relevant temporal external search result",
        seq_in_session=1,
        token_count=5,
        created_at=datetime.now(timezone.utc),
    )

    class FakeDb:
        def expunge(self, _obj: object) -> None:
            call_order.append("expunge")

    fake_db = FakeDb()

    async def fake_search_messages_external(
        workspace_name: str,
        query_embedding: list[float],
        limit: int,
        *,
        session_name: str | None = None,
        allowed_session_names: list[str] | None = None,
        after_date: datetime | None = None,
        before_date: datetime | None = None,
    ) -> list[str]:
        _ = (
            workspace_name,
            query_embedding,
            limit,
            session_name,
            allowed_session_names,
        )
        assert after_date is not None
        assert before_date is not None
        call_order.append("external")
        return ["message-1"]

    async def fake_fetch_messages_by_ids(
        db: FakeDb,
        workspace_name: str,
        message_ids: list[str],
        *,
        after_date: datetime | None = None,
        before_date: datetime | None = None,
    ) -> list[models.Message]:
        _ = (workspace_name, message_ids)
        assert db is fake_db
        assert after_date is not None
        assert before_date is not None
        call_order.append("fetch")
        return [message]

    async def fake_build_merged_snippets(
        db: FakeDb,
        workspace_name: str,
        matched_messages: list[models.Message],
        context_window: int,
    ) -> list[tuple[list[models.Message], list[models.Message]]]:
        _ = (workspace_name, context_window)
        assert db is fake_db
        assert matched_messages == [message]
        call_order.append("build")
        return [([message], [message])]

    @asynccontextmanager
    async def fake_tracked_db(_operation_name: str | None = None):
        call_order.append("enter")
        yield fake_db
        call_order.append("exit")

    monkeypatch.setattr(
        message_crud, "_search_messages_external", fake_search_messages_external
    )
    monkeypatch.setattr(
        message_crud, "_fetch_messages_by_ids", fake_fetch_messages_by_ids
    )
    monkeypatch.setattr(
        message_crud, "_build_merged_snippets", fake_build_merged_snippets
    )
    monkeypatch.setattr(message_crud, "tracked_db", fake_tracked_db)

    snippets = await message_crud.search_messages_temporal(
        workspace_name="workspace",
        session_name="session",
        query="relevant query",
        after_date=after_date,
        before_date=before_date,
        embedding=[0.1, 0.2, 0.3],
    )

    assert snippets == [([message], [message])]
    assert call_order.index("external") < call_order.index("enter")


@pytest.mark.asyncio
async def test_message_chunking_creates_multiple_embeddings(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    mock_openai_embeddings: dict[str, Any],
):
    """Test that messages exceeding token limits are chunked and create multiple embeddings"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.EMBED_MESSAGES", True)

    # Mock a low token limit to force chunking
    monkeypatch.setattr("src.config.settings.EMBEDDING.MAX_INPUT_TOKENS", 10)

    test_workspace, test_peer = sample_data

    # Create a test session
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()

    test_message_content = "This is a very long message that should be chunked into multiple pieces because it exceeds the token limit that we set for testing purposes. This message contains many words and should definitely be split into multiple chunks."

    def mock_batch_embed_chunked(
        id_resource_dict: dict[str, str],
    ) -> dict[str, list[list[float]]]:
        return {
            text_id: [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]  # 3 chunks per message
            for text_id in id_resource_dict
        }

    mock_openai_embeddings["batch_embed"].side_effect = mock_batch_embed_chunked

    messages = [
        MessageCreate(
            content=test_message_content,
            peer_id=test_peer.name,
            metadata={"test": "chunking"},
        )
    ]

    created_messages = await create_messages(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    assert len(created_messages) == 1
    created_message = created_messages[0]

    # Query the MessageEmbedding table to verify multiple embeddings were created
    stmt = select(models.MessageEmbedding).where(
        models.MessageEmbedding.message_id == created_message.public_id
    )
    result = await db_session.execute(stmt)
    embedding_records = list(result.scalars().all())

    # Verify multiple embedding records were created (one per chunk)
    # Embedding vectors are now stored externally in the vector store
    assert len(embedding_records) == 3  # Should have 3 embeddings for 3 chunks

    for _, embedding_record in enumerate(embedding_records):
        assert embedding_record.message_id == created_message.public_id
        assert (
            embedding_record.content == test_message_content
        )  # Full content stored in each
        assert embedding_record.workspace_name == test_workspace.name
        assert embedding_record.session_name == test_session.name
        assert embedding_record.peer_name == test_peer.name

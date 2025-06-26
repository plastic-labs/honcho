from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud import create_messages, search
from src.models import Peer, Workspace
from src.schemas import MessageCreate


@pytest.mark.asyncio
async def test_message_embedding_created_when_setting_enabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that MessageEmbedding is created when EMBED_MESSAGES setting is True"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.LLM.EMBED_MESSAGES", True)

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

    # Verify the embedding was created
    assert embedding_record is not None
    assert embedding_record.message_id == created_message.public_id
    assert embedding_record.content == test_message_content
    assert embedding_record.workspace_name == test_workspace.name
    assert embedding_record.session_name == test_session.name
    assert embedding_record.peer_name == test_peer.name
    # Verify embedding vector exists and is not empty
    assert embedding_record.embedding is not None
    assert len(embedding_record.embedding) > 0


@pytest.mark.asyncio
async def test_message_embedding_not_created_when_setting_disabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that MessageEmbedding is NOT created when EMBED_MESSAGES setting is False"""
    # Monkeypatch the setting to disable message embeddings
    monkeypatch.setattr("src.config.settings.LLM.EMBED_MESSAGES", False)

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
    monkeypatch.setattr("src.config.settings.LLM.EMBED_MESSAGES", True)

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

        # Verify the embedding was created
        assert embedding_record is not None
        assert embedding_record.message_id == created_message.public_id
        assert embedding_record.content == messages[i].content
        assert embedding_record.workspace_name == test_workspace.name
        assert embedding_record.session_name == test_session.name
        assert embedding_record.peer_name == test_peer.name
        assert embedding_record.embedding is not None
        assert len(embedding_record.embedding) > 0


@pytest.mark.asyncio
async def test_message_embedding_with_peer_only_messages(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that MessageEmbedding is created for peer-only messages (no session)"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.LLM.EMBED_MESSAGES", True)

    test_workspace, test_peer = sample_data

    # Import the peer-specific message creation function
    from src.crud import create_messages_for_peer

    # Create a message for peer only (no session)
    test_message_content = "This is a peer-only message with embedding"
    messages = [
        MessageCreate(
            content=test_message_content,
            peer_id=test_peer.name,  # This will be overridden by the function
            metadata={"test": "peer_only"},
        )
    ]

    created_messages = await create_messages_for_peer(
        db=db_session,
        messages=messages,
        workspace_name=test_workspace.name,
        peer_name=test_peer.name,
    )

    assert len(created_messages) == 1
    created_message = created_messages[0]

    # Verify the message was created with peer but no session
    assert created_message.peer_name == test_peer.name
    assert created_message.session_name is None

    # Query the MessageEmbedding table to verify an embedding was created
    stmt = select(models.MessageEmbedding).where(
        models.MessageEmbedding.message_id == created_message.public_id
    )
    result = await db_session.execute(stmt)
    embedding_record = result.scalar_one_or_none()

    # Verify the embedding was created
    assert embedding_record is not None
    assert embedding_record.message_id == created_message.public_id
    assert embedding_record.content == test_message_content
    assert embedding_record.workspace_name == test_workspace.name
    assert (
        embedding_record.session_name is None
    )  # Should be None for peer-only messages
    assert embedding_record.peer_name == test_peer.name
    assert embedding_record.embedding is not None
    assert len(embedding_record.embedding) > 0


@pytest.mark.asyncio
async def test_semantic_search_when_embeddings_enabled(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    mock_openai_embeddings: dict[str, Any],
):
    """Test that search uses semantic search by default when EMBED_MESSAGES is True"""
    # Monkeypatch the setting to enable message embeddings
    monkeypatch.setattr("src.config.settings.LLM.EMBED_MESSAGES", True)

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

    search_stmt = await search(
        query=search_query,
        workspace_name=test_workspace.name,
        session_name=test_session.name,
    )

    # Verify that the embed method was called during search - e.g. we used semantic search
    assert mock_openai_embeddings["embed"].call_count == initial_call_count + 1

    search_result = await db_session.execute(search_stmt)
    found_messages = list(search_result.scalars().all())

    # Verify that our message was found via semantic search
    assert len(found_messages) > 0
    found_message_ids = [msg.public_id for msg in found_messages]
    assert created_message.public_id in found_message_ids

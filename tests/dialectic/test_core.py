from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dialectic.chat import agentic_chat, agentic_chat_stream
from src.dialectic.core import DialecticAgent
from src.utils.representation import Representation


@pytest.mark.asyncio
async def test_initialize_session_history_respects_peer_perspective_visibility(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, observer = sample_data
    sender = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(sender)
    await db_session.flush()

    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    t0 = datetime.now(timezone.utc)
    join_time = t0 + timedelta(seconds=5)
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=observer.name,
            joined_at=join_time,
            left_at=None,
        )
    )
    await db_session.flush()

    msg_before_join = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=sender.name,
        content="HIDDEN_BEFORE_JOIN",
        seq_in_session=1,
        token_count=5,
        created_at=t0,
    )
    msg_after_join = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=sender.name,
        content="VISIBLE_AFTER_JOIN",
        seq_in_session=2,
        token_count=5,
        created_at=t0 + timedelta(seconds=10),
    )
    db_session.add_all([msg_before_join, msg_after_join])
    await db_session.commit()

    monkeypatch.setattr(
        "src.config.settings.DIALECTIC.SESSION_HISTORY_MAX_TOKENS", 4096
    )

    agent = DialecticAgent(
        db=None,
        workspace_name=workspace.name,
        session_name=session.name,
        observer=observer.name,
        observed=observer.name,
    )

    await agent._initialize_session_history()  # pyright: ignore[reportPrivateUsage]

    system_prompt = str(agent.messages[0]["content"])
    assert "<session_history>" in system_prompt
    assert "VISIBLE_AFTER_JOIN" in system_prompt
    assert "HIDDEN_BEFORE_JOIN" not in system_prompt

    # Guard against duplicate insertion if initialization is called again.
    await agent._initialize_session_history()  # pyright: ignore[reportPrivateUsage]
    updated_system_prompt = str(agent.messages[0]["content"])
    assert updated_system_prompt.count("<session_history>") == 1


@pytest.mark.asyncio
async def test_prefetch_relevant_observations_reuses_query_embedding(
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, observer = sample_data
    agent = DialecticAgent(
        db=None,
        workspace_name=workspace.name,
        session_name=None,
        observer=observer.name,
        observed=observer.name,
    )

    embed_calls = 0
    passed_embeddings: list[list[float]] = []

    async def fake_embed(query: str) -> list[float]:
        nonlocal embed_calls
        assert query == "coffee"
        embed_calls += 1
        return [0.1, 0.2, 0.3]

    async def fake_search_memory(**kwargs: object) -> Representation:
        embedding = kwargs.get("embedding")
        assert isinstance(embedding, list)
        passed_embeddings.append(cast(list[float], embedding))
        return Representation()

    monkeypatch.setattr("src.dialectic.core.embedding_client.embed", fake_embed)
    monkeypatch.setattr("src.dialectic.core.search_memory", fake_search_memory)

    result = await agent._prefetch_relevant_observations(  # pyright: ignore[reportPrivateUsage]
        "coffee"
    )

    assert result is None
    assert embed_calls == 1
    assert passed_embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_agentic_chat_releases_preflight_session_before_agent_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_sessions = 0

    @asynccontextmanager
    async def fake_tracked_db(_: str | None = None):
        nonlocal active_sessions
        active_sessions += 1
        try:
            yield object()
        finally:
            active_sessions -= 1

    async def fake_get_peer(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return object()

    async def fake_get_session(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return object()

    async def fake_get_workspace(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return object()

    async def fake_answer(_self: object, query: str) -> str:
        assert query == "What changed?"
        assert active_sessions == 0
        return "ok"

    def fake_get_configuration(
        message_configuration: object | None,
        session: object | None,
        workspace: object | None = None,
    ) -> Any:
        _ = (message_configuration, session, workspace)
        return SimpleNamespace(peer_card=SimpleNamespace(use=False))

    monkeypatch.setattr("src.dialectic.chat.tracked_db", fake_tracked_db)
    monkeypatch.setattr("src.dialectic.chat.crud.get_peer", fake_get_peer)
    monkeypatch.setattr("src.dialectic.chat.crud.get_session", fake_get_session)
    monkeypatch.setattr("src.dialectic.chat.crud.get_workspace", fake_get_workspace)
    monkeypatch.setattr("src.dialectic.chat.get_configuration", fake_get_configuration)
    monkeypatch.setattr("src.dialectic.chat.DialecticAgent.answer", fake_answer)

    result = await agentic_chat(
        workspace_name="workspace",
        session_name="session",
        query="What changed?",
        observer="observer",
        observed="observer",
    )

    assert result == "ok"


@pytest.mark.asyncio
async def test_agentic_chat_stream_releases_preflight_session_before_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_sessions = 0

    @asynccontextmanager
    async def fake_tracked_db(_: str | None = None):
        nonlocal active_sessions
        active_sessions += 1
        try:
            yield object()
        finally:
            active_sessions -= 1

    async def fake_get_peer(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return object()

    async def fake_get_session(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return object()

    async def fake_get_workspace(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        assert active_sessions == 1
        return object()

    async def fake_answer_stream(_self: object, query: str):
        assert query == "Stream it"
        assert active_sessions == 0
        yield "chunk-1"
        assert active_sessions == 0
        yield "chunk-2"

    def fake_get_configuration(
        message_configuration: object | None,
        session: object | None,
        workspace: object | None = None,
    ) -> Any:
        _ = (message_configuration, session, workspace)
        return SimpleNamespace(peer_card=SimpleNamespace(use=False))

    monkeypatch.setattr("src.dialectic.chat.tracked_db", fake_tracked_db)
    monkeypatch.setattr("src.dialectic.chat.crud.get_peer", fake_get_peer)
    monkeypatch.setattr("src.dialectic.chat.crud.get_session", fake_get_session)
    monkeypatch.setattr("src.dialectic.chat.crud.get_workspace", fake_get_workspace)
    monkeypatch.setattr("src.dialectic.chat.get_configuration", fake_get_configuration)
    monkeypatch.setattr(
        "src.dialectic.chat.DialecticAgent.answer_stream",
        fake_answer_stream,
    )

    chunks = [
        chunk
        async for chunk in agentic_chat_stream(
            workspace_name="workspace",
            session_name="session",
            query="Stream it",
            observer="observer",
            observed="observer",
        )
    ]

    assert chunks == ["chunk-1", "chunk-2"]

from datetime import datetime, timedelta, timezone

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dialectic.core import DialecticAgent


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
    await db_session.flush()

    monkeypatch.setattr(
        "src.config.settings.DIALECTIC.SESSION_HISTORY_MAX_TOKENS", 4096
    )

    agent = DialecticAgent(
        db=db_session,
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

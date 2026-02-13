import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models


def _embedding(seed: float) -> list[float]:
    return [seed] * 1536


@pytest.mark.asyncio
async def test_search_messages_peer_perspective_filters_visibility_windows(
    db_session: AsyncSession,
):
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    observer = models.Peer(name="observer", workspace_name=workspace.name)
    sender = models.Peer(name="sender", workspace_name=workspace.name)
    db_session.add_all([observer, sender])
    await db_session.flush()

    session_visible = models.Session(
        name="session-visible", workspace_name=workspace.name
    )
    session_hidden = models.Session(
        name="session-hidden", workspace_name=workspace.name
    )
    db_session.add_all([session_visible, session_hidden])
    await db_session.flush()

    t0 = datetime.datetime.now(datetime.timezone.utc)
    join_time = t0 + datetime.timedelta(seconds=5)

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
        peer_name=sender.name,
        content="Before observer joined",
        seq_in_session=1,
        token_count=5,
        created_at=t0,
    )
    msg_after_join = models.Message(
        workspace_name=workspace.name,
        session_name=session_visible.name,
        peer_name=sender.name,
        content="After observer joined",
        seq_in_session=2,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    msg_other_session = models.Message(
        workspace_name=workspace.name,
        session_name=session_hidden.name,
        peer_name=sender.name,
        content="Observer never in this session",
        seq_in_session=1,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    db_session.add_all([msg_before_join, msg_after_join, msg_other_session])
    await db_session.flush()

    db_session.add_all(
        [
            models.MessageEmbedding(
                content=msg_before_join.content,
                embedding=_embedding(0.1),
                message_id=msg_before_join.public_id,
                workspace_name=workspace.name,
                session_name=session_visible.name,
                peer_name=sender.name,
            ),
            models.MessageEmbedding(
                content=msg_after_join.content,
                embedding=_embedding(0.2),
                message_id=msg_after_join.public_id,
                workspace_name=workspace.name,
                session_name=session_visible.name,
                peer_name=sender.name,
            ),
            models.MessageEmbedding(
                content=msg_other_session.content,
                embedding=_embedding(0.3),
                message_id=msg_other_session.public_id,
                workspace_name=workspace.name,
                session_name=session_hidden.name,
                peer_name=sender.name,
            ),
        ]
    )
    await db_session.flush()

    scoped = await crud.search_messages(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        query="joined",
        limit=10,
        context_window=1,
        peer_perspective=observer.name,
    )

    scoped_matched_ids = {msg.public_id for matched, _ in scoped for msg in matched}
    scoped_context_ids = {msg.public_id for _, context in scoped for msg in context}

    assert msg_after_join.public_id in scoped_matched_ids

    assert msg_before_join.public_id not in scoped_matched_ids
    assert msg_other_session.public_id not in scoped_matched_ids
    assert msg_before_join.public_id not in scoped_context_ids
    assert msg_other_session.public_id not in scoped_context_ids

    unscoped = await crud.search_messages(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        query="joined",
        limit=10,
        context_window=0,
        peer_perspective=None,
    )
    unscoped_matched_ids = {msg.public_id for matched, _ in unscoped for msg in matched}

    assert msg_before_join.public_id in unscoped_matched_ids
    assert msg_other_session.public_id in unscoped_matched_ids


@pytest.mark.asyncio
async def test_get_messages_with_peer_perspective_scopes_session_history(
    db_session: AsyncSession,
):
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    observer = models.Peer(name="observer", workspace_name=workspace.name)
    sender = models.Peer(name="sender", workspace_name=workspace.name)
    db_session.add_all([observer, sender])
    await db_session.flush()

    session = models.Session(name="session-1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    t0 = datetime.datetime.now(datetime.timezone.utc)
    join_time = t0 + datetime.timedelta(seconds=10)

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

    old_msg = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=sender.name,
        content="Old message",
        seq_in_session=1,
        token_count=5,
        created_at=t0,
    )
    new_msg = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=sender.name,
        content="New message",
        seq_in_session=2,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=20),
    )
    db_session.add_all([old_msg, new_msg])
    await db_session.flush()

    scoped_stmt = await crud.get_messages(
        workspace_name=workspace.name,
        session_name=session.name,
        reverse=False,
        peer_perspective=observer.name,
    )
    scoped_messages = (await db_session.execute(scoped_stmt)).scalars().all()
    scoped_ids = [msg.public_id for msg in scoped_messages]

    assert old_msg.public_id not in scoped_ids
    assert new_msg.public_id in scoped_ids

    unscoped_stmt = await crud.get_messages(
        workspace_name=workspace.name,
        session_name=session.name,
        reverse=False,
        peer_perspective=None,
    )
    unscoped_messages = (await db_session.execute(unscoped_stmt)).scalars().all()
    unscoped_ids = [msg.public_id for msg in unscoped_messages]

    assert old_msg.public_id in unscoped_ids
    assert new_msg.public_id in unscoped_ids


@pytest.mark.asyncio
async def test_grep_messages_with_peer_perspective_filters_context(
    db_session: AsyncSession,
):
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    observer = models.Peer(name="observer", workspace_name=workspace.name)
    sender = models.Peer(name="sender", workspace_name=workspace.name)
    db_session.add_all([observer, sender])
    await db_session.flush()

    session_visible = models.Session(
        name="session-visible", workspace_name=workspace.name
    )
    session_hidden = models.Session(
        name="session-hidden", workspace_name=workspace.name
    )
    db_session.add_all([session_visible, session_hidden])
    await db_session.flush()

    t0 = datetime.datetime.now(datetime.timezone.utc)
    join_time = t0 + datetime.timedelta(seconds=5)

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
        peer_name=sender.name,
        content="LEAK_TOKEN before join",
        seq_in_session=1,
        token_count=5,
        created_at=t0,
    )
    msg_after_join = models.Message(
        workspace_name=workspace.name,
        session_name=session_visible.name,
        peer_name=sender.name,
        content="LEAK_TOKEN after join",
        seq_in_session=2,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    msg_hidden = models.Message(
        workspace_name=workspace.name,
        session_name=session_hidden.name,
        peer_name=sender.name,
        content="LEAK_TOKEN hidden session",
        seq_in_session=1,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    db_session.add_all([msg_before_join, msg_after_join, msg_hidden])
    await db_session.flush()

    scoped = await crud.grep_messages(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        text="LEAK_TOKEN",
        limit=10,
        context_window=1,
        peer_perspective=observer.name,
    )
    scoped_matched_ids = {msg.public_id for matched, _ in scoped for msg in matched}
    scoped_context_ids = {msg.public_id for _, context in scoped for msg in context}

    assert msg_after_join.public_id in scoped_matched_ids

    assert msg_before_join.public_id not in scoped_matched_ids
    assert msg_hidden.public_id not in scoped_matched_ids
    assert msg_before_join.public_id not in scoped_context_ids
    assert msg_hidden.public_id not in scoped_context_ids

    unscoped = await crud.grep_messages(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        text="LEAK_TOKEN",
        limit=10,
        context_window=0,
        peer_perspective=None,
    )
    unscoped_matched_ids = {msg.public_id for matched, _ in unscoped for msg in matched}

    assert msg_before_join.public_id in unscoped_matched_ids
    assert msg_hidden.public_id in unscoped_matched_ids


@pytest.mark.asyncio
async def test_get_messages_by_date_range_with_peer_perspective_filters_visibility(
    db_session: AsyncSession,
):
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    observer = models.Peer(name="observer", workspace_name=workspace.name)
    sender = models.Peer(name="sender", workspace_name=workspace.name)
    db_session.add_all([observer, sender])
    await db_session.flush()

    session_visible = models.Session(
        name="session-visible", workspace_name=workspace.name
    )
    session_hidden = models.Session(
        name="session-hidden", workspace_name=workspace.name
    )
    db_session.add_all([session_visible, session_hidden])
    await db_session.flush()

    t0 = datetime.datetime.now(datetime.timezone.utc)
    join_time = t0 + datetime.timedelta(seconds=5)

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
        peer_name=sender.name,
        content="Before join",
        seq_in_session=1,
        token_count=5,
        created_at=t0,
    )
    msg_after_join = models.Message(
        workspace_name=workspace.name,
        session_name=session_visible.name,
        peer_name=sender.name,
        content="After join",
        seq_in_session=2,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    msg_hidden = models.Message(
        workspace_name=workspace.name,
        session_name=session_hidden.name,
        peer_name=sender.name,
        content="Hidden session",
        seq_in_session=1,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    db_session.add_all([msg_before_join, msg_after_join, msg_hidden])
    await db_session.flush()

    scoped = await crud.get_messages_by_date_range(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        after_date=t0 - datetime.timedelta(seconds=1),
        before_date=t0 + datetime.timedelta(seconds=20),
        limit=20,
        order="asc",
        peer_perspective=observer.name,
    )
    scoped_ids = {msg.public_id for msg in scoped}
    assert msg_after_join.public_id in scoped_ids

    assert msg_before_join.public_id not in scoped_ids
    assert msg_hidden.public_id not in scoped_ids

    unscoped = await crud.get_messages_by_date_range(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        after_date=t0 - datetime.timedelta(seconds=1),
        before_date=t0 + datetime.timedelta(seconds=20),
        limit=20,
        order="asc",
        peer_perspective=None,
    )
    unscoped_ids = {msg.public_id for msg in unscoped}
    assert msg_before_join.public_id in unscoped_ids
    assert msg_hidden.public_id in unscoped_ids


@pytest.mark.asyncio
async def test_search_messages_temporal_with_peer_perspective_filters_visibility(
    db_session: AsyncSession,
):
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    observer = models.Peer(name="observer", workspace_name=workspace.name)
    sender = models.Peer(name="sender", workspace_name=workspace.name)
    db_session.add_all([observer, sender])
    await db_session.flush()

    session_visible = models.Session(
        name="session-visible", workspace_name=workspace.name
    )
    session_hidden = models.Session(
        name="session-hidden", workspace_name=workspace.name
    )
    db_session.add_all([session_visible, session_hidden])
    await db_session.flush()

    t0 = datetime.datetime.now(datetime.timezone.utc)
    join_time = t0 + datetime.timedelta(seconds=5)

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
        peer_name=sender.name,
        content="Temporal topic before join",
        seq_in_session=1,
        token_count=5,
        created_at=t0,
    )
    msg_after_join = models.Message(
        workspace_name=workspace.name,
        session_name=session_visible.name,
        peer_name=sender.name,
        content="Temporal topic after join",
        seq_in_session=2,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    msg_hidden = models.Message(
        workspace_name=workspace.name,
        session_name=session_hidden.name,
        peer_name=sender.name,
        content="Temporal topic hidden session",
        seq_in_session=1,
        token_count=5,
        created_at=t0 + datetime.timedelta(seconds=10),
    )
    db_session.add_all([msg_before_join, msg_after_join, msg_hidden])
    await db_session.flush()

    db_session.add_all(
        [
            models.MessageEmbedding(
                content=msg_before_join.content,
                embedding=_embedding(0.11),
                message_id=msg_before_join.public_id,
                workspace_name=workspace.name,
                session_name=session_visible.name,
                peer_name=sender.name,
            ),
            models.MessageEmbedding(
                content=msg_after_join.content,
                embedding=_embedding(0.22),
                message_id=msg_after_join.public_id,
                workspace_name=workspace.name,
                session_name=session_visible.name,
                peer_name=sender.name,
            ),
            models.MessageEmbedding(
                content=msg_hidden.content,
                embedding=_embedding(0.33),
                message_id=msg_hidden.public_id,
                workspace_name=workspace.name,
                session_name=session_hidden.name,
                peer_name=sender.name,
            ),
        ]
    )
    await db_session.flush()

    scoped = await crud.search_messages_temporal(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        query="Temporal topic",
        after_date=t0 - datetime.timedelta(seconds=1),
        before_date=t0 + datetime.timedelta(seconds=20),
        limit=10,
        context_window=1,
        peer_perspective=observer.name,
    )
    scoped_matched_ids = {msg.public_id for matched, _ in scoped for msg in matched}
    scoped_context_ids = {msg.public_id for _, context in scoped for msg in context}

    assert msg_after_join.public_id in scoped_matched_ids
    assert msg_before_join.public_id not in scoped_matched_ids
    assert msg_hidden.public_id not in scoped_matched_ids
    assert msg_before_join.public_id not in scoped_context_ids
    assert msg_hidden.public_id not in scoped_context_ids

    unscoped = await crud.search_messages_temporal(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        query="Temporal topic",
        after_date=t0 - datetime.timedelta(seconds=1),
        before_date=t0 + datetime.timedelta(seconds=20),
        limit=10,
        context_window=0,
        peer_perspective=None,
    )
    unscoped_matched_ids = {msg.public_id for matched, _ in unscoped for msg in matched}

    assert msg_before_join.public_id in unscoped_matched_ids
    assert msg_hidden.public_id in unscoped_matched_ids

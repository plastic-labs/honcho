"""Message erasure and the deletion lifecycle barrier."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.cache.client import cache
from src.config import settings
from src.crud.erasure import MessageIdentity, TenantContext, erase_message
from src.crud.lifecycle import capture_fence, check_fence
from src.crud.peer import peer_cache_key
from src.exceptions import StaleLifecycleError
from src.utils import summarizer
from src.utils.representation import ExplicitObservationBase, PromptRepresentation

CANARY = "canary-7f3a"


@dataclass
class Scenario:
    workspace: str
    session: str
    other_session: str
    sender: str
    observers: list[str]
    message: models.Message
    survivor: models.Message
    session_doc_ids: list[str]
    descendant_id: str
    unrelated_id: str


def _doc(
    workspace: str,
    observer: str,
    observed: str,
    session: str | None,
    content: str,
    level: str = "explicit",
    source_ids: list[str] | None = None,
) -> models.Document:
    document = models.Document(
        workspace_name=workspace,
        observer=observer,
        observed=observed,
        session_name=session,
        content=content,
        level=level,
        internal_metadata={},
    )
    document.id = generate_nanoid()
    if source_ids:
        document.source_ids = source_ids
    return document


async def _seed(db: AsyncSession) -> Scenario:
    workspace = generate_nanoid()
    session, other_session = generate_nanoid(), generate_nanoid()
    sender, observer_b, observer_c = (generate_nanoid() for _ in range(3))
    db.add(models.Workspace(name=workspace))
    await db.flush()
    for peer in (sender, observer_b, observer_c):
        db.add(models.Peer(name=peer, workspace_name=workspace))
    for name in (session, other_session):
        db.add(
            models.Session(
                name=name,
                workspace_name=workspace,
                internal_metadata={
                    summarizer.SUMMARIES_KEY: {
                        summarizer.SummaryType.SHORT.value: {
                            "content": f"short {CANARY}",
                            "message_id": 1,
                        },
                        summarizer.SummaryType.LONG.value: {
                            "content": f"long {CANARY}",
                            "message_id": 1,
                        },
                    }
                },
            )
        )
    await db.flush()

    message = models.Message(
        workspace_name=workspace,
        session_name=session,
        peer_name=sender,
        content=f"I live in {CANARY}",
        seq_in_session=1,
    )
    survivor = models.Message(
        workspace_name=workspace,
        session_name=session,
        peer_name=observer_b,
        content="unrelated reply",
        seq_in_session=2,
    )
    db.add_all([message, survivor])
    await db.flush()
    db.add(
        models.MessageEmbedding(
            content=message.content,
            message_id=message.public_id,
            workspace_name=workspace,
            session_name=session,
            peer_name=sender,
        )
    )

    for observer in (observer_b, observer_c):
        db.add(
            models.Collection(
                observer=observer, observed=sender, workspace_name=workspace
            )
        )
    await db.flush()

    session_docs = [
        _doc(workspace, observer, sender, session, f"{observer} knows {CANARY}")
        for observer in (observer_b, observer_c)
    ]
    db.add_all(session_docs)
    await db.flush()
    descendant = _doc(
        workspace,
        observer_b,
        sender,
        None,
        f"deduced from {CANARY}",
        level="deductive",
        source_ids=[session_docs[0].id],
    )
    unrelated = _doc(workspace, observer_b, sender, other_session, "likes tea")
    db.add_all([descendant, unrelated])

    work_unit_key = f"representation:{workspace}:{session}:{sender}"
    db.add_all(
        [
            models.QueueItem(
                session_id=None,
                work_unit_key=work_unit_key,
                task_type="representation",
                payload={
                    "observers": [observer_b, observer_c],
                    "observed": sender,
                    "session_name": session,
                },
                workspace_name=workspace,
                message_id=message.id,
            ),
            models.QueueItem(
                work_unit_key=f"summary:{workspace}:{session}",
                task_type="summary",
                payload={"session_name": session},
                processed=True,
                workspace_name=workspace,
                message_id=message.id,
            ),
            models.ActiveQueueSession(work_unit_key=work_unit_key),
        ]
    )
    for observer in (observer_b, observer_c):
        peer = await db.scalar(
            select(models.Peer).where(
                models.Peer.workspace_name == workspace, models.Peer.name == observer
            )
        )
        assert peer is not None
        peer.internal_metadata = {
            f"{sender}_peer_card": [f"ATTRIBUTE: lives in {CANARY}"],
            "unrelated": True,
        }
    await db.commit()

    return Scenario(
        workspace=workspace,
        session=session,
        other_session=other_session,
        sender=sender,
        observers=[observer_b, observer_c],
        message=message,
        survivor=survivor,
        session_doc_ids=[doc.id for doc in session_docs],
        descendant_id=descendant.id,
        unrelated_id=unrelated.id,
    )


async def _count(db: AsyncSession, stmt: Any) -> int:
    return int(await db.scalar(select(func.count()).select_from(stmt.subquery())) or 0)


@pytest.mark.asyncio
async def test_delete_route_erases_message_and_invalidates_session(
    client: TestClient, db_session: AsyncSession
):
    scenario = await _seed(db_session)
    base = f"/v3/workspaces/{scenario.workspace}/sessions/{scenario.session}/messages"
    survivor_url = f"{base}/{scenario.survivor.public_id}"

    response = client.delete(f"{base}/{scenario.message.public_id}")
    assert response.status_code == 200, response.text
    assert client.get(f"{base}/{scenario.message.public_id}").status_code == 404
    assert client.delete(f"{base}/{scenario.message.public_id}").status_code == 404

    db_session.expire_all()
    ws = scenario.workspace
    assert not await _count(
        db_session,
        select(models.MessageEmbedding).where(
            models.MessageEmbedding.message_id == scenario.message.public_id
        ),
    )
    assert not await _count(
        db_session,
        select(models.QueueItem).where(models.QueueItem.workspace_name == ws),
    )
    assert not await _count(db_session, select(models.ActiveQueueSession))

    remaining_docs = set(
        (
            await db_session.scalars(
                select(models.Document.id).where(models.Document.workspace_name == ws)
            )
        ).all()
    )
    assert remaining_docs == {scenario.unrelated_id}

    for observer in scenario.observers:
        card = await crud.get_peer_card(
            db_session, ws, observer=observer, observed=scenario.sender
        )
        assert card is None
        metadata = await db_session.scalar(
            select(models.Peer.internal_metadata).where(
                models.Peer.workspace_name == ws, models.Peer.name == observer
            )
        )
        assert metadata == {"unrelated": True}

    assert await summarizer.get_both_summaries(db_session, ws, scenario.session) == (
        None,
        None,
    )
    short, long = await summarizer.get_both_summaries(
        db_session, ws, scenario.other_session
    )
    assert short is not None and long is not None

    session_generation = await db_session.scalar(
        select(models.Session.deletion_generation).where(
            models.Session.name == scenario.session
        )
    )
    assert session_generation == 1
    collection_generations = (
        await db_session.scalars(
            select(models.Collection.deletion_generation).where(
                models.Collection.workspace_name == ws
            )
        )
    ).all()
    assert sorted(collection_generations) == [1, 1]

    survivor = client.get(survivor_url)
    assert survivor.status_code == 200


@pytest.mark.asyncio
async def test_create_documents_refuses_stale_fence_including_fallback(
    db_session: AsyncSession,
):
    scenario = await _seed(db_session)
    ws, observer = scenario.workspace, scenario.observers[0]
    fence = await capture_fence(
        db_session,
        ws,
        session_name=scenario.session,
        representations=[(observer, scenario.sender)],
    )
    await db_session.commit()

    await erase_message(
        db_session,
        TenantContext(workspace_name=ws),
        MessageIdentity(scenario.session, scenario.message.public_id),
    )

    documents = [
        schemas.DocumentCreate(
            content=f"{observer} knows {CANARY}",
            session_name=scenario.session,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[scenario.message.id],
                message_created_at="2026-01-01T00:00:00Z",
            ),
            embedding=[0.0] * settings.EMBEDDING.VECTOR_DIMENSIONS,
        ),
        schemas.DocumentCreate(
            content=f"new fact about {CANARY}",
            session_name=scenario.session,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[scenario.message.id],
                message_created_at="2026-01-01T00:00:00Z",
            ),
            embedding=[0.0] * settings.EMBEDDING.VECTOR_DIMENSIONS,
        ),
    ]
    row_updates = AsyncMock(return_value=[])
    with (
        patch("src.crud.document._apply_document_row_updates", row_updates),
        pytest.raises(StaleLifecycleError),
    ):
        await crud.create_documents(
            db_session,
            documents,
            ws,
            observer=observer,
            observed=scenario.sender,
            fence=fence,
        )
    # Refused before reinforcement and its fallback insert
    row_updates.assert_not_called()

    contents = (
        await db_session.scalars(
            select(models.Document.content).where(models.Document.workspace_name == ws)
        )
    ).all()
    assert all(CANARY not in content for content in contents)


@pytest.mark.asyncio
async def test_fenced_writer_and_erasure_serialize(
    db_session: AsyncSession, db_engine: AsyncEngine
):
    """A writer that passed its fence commits first; the erasure then removes its output."""
    scenario = await _seed(db_session)
    ws, observer = scenario.workspace, scenario.observers[0]
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async with session_factory() as writer, session_factory() as eraser:
        fence = await capture_fence(writer, ws, session_name=scenario.session)
        await check_fence(writer, fence)
        writer.add(
            _doc(ws, observer, scenario.sender, scenario.session, f"late {CANARY}")
        )
        await writer.flush()

        erasure = asyncio.create_task(
            erase_message(
                eraser,
                TenantContext(workspace_name=ws),
                MessageIdentity(scenario.session, scenario.message.public_id),
            )
        )
        await asyncio.sleep(0.5)
        assert not erasure.done(), "erasure must wait for the fenced writer"
        await writer.commit()
        await asyncio.wait_for(erasure, timeout=10)

    db_session.expire_all()
    contents = (
        await db_session.scalars(
            select(models.Document.content).where(models.Document.workspace_name == ws)
        )
    ).all()
    assert f"late {CANARY}" not in contents


@pytest.mark.asyncio
async def test_summary_save_refuses_stale_fence_and_merges_server_side(
    db_session: AsyncSession,
):
    scenario = await _seed(db_session)
    ws = scenario.workspace
    fence = await capture_fence(db_session, ws, session_name=scenario.session)
    await db_session.commit()
    await erase_message(
        db_session,
        TenantContext(workspace_name=ws),
        MessageIdentity(scenario.session, scenario.message.public_id),
    )

    stale_summary: Any = {
        "content": f"stale {CANARY}",
        "message_id": scenario.message.id,
        "summary_type": summarizer.SummaryType.SHORT.value,
        "created_at": "2026-01-01T00:00:00Z",
        "token_count": 3,
        "message_public_id": scenario.message.public_id,
    }
    with pytest.raises(StaleLifecycleError):
        await summarizer._save_summary(  # pyright: ignore[reportPrivateUsage]
            db_session, stale_summary, ws, scenario.session, fence=fence
        )

    fresh_fence = await capture_fence(db_session, ws, session_name=scenario.session)
    fresh_summary: Any = {**stale_summary, "content": "fresh"}
    await summarizer._save_summary(  # pyright: ignore[reportPrivateUsage]
        db_session, fresh_summary, ws, scenario.session, fence=fresh_fence
    )
    short, long = await summarizer.get_both_summaries(db_session, ws, scenario.session)
    assert short is not None and short["content"] == "fresh"
    assert long is None


@pytest.mark.asyncio
async def test_card_reads_bypass_cache_and_fenced_writes_refuse(
    db_session: AsyncSession,
):
    scenario = await _seed(db_session)
    ws, observer = scenario.workspace, scenario.observers[0]

    await crud.get_peer(db_session, ws, observer)
    assert await cache.get(peer_cache_key(ws, observer)) is not None
    fence = await capture_fence(
        db_session, ws, representations=[(observer, scenario.sender)]
    )
    await db_session.commit()

    # A failed eviction leaves the pre-delete peer in the cache
    cached_peer = await cache.get(peer_cache_key(ws, observer))
    await erase_message(
        db_session,
        TenantContext(workspace_name=ws),
        MessageIdentity(scenario.session, scenario.message.public_id),
    )
    await cache.set(peer_cache_key(ws, observer), cached_peer)

    assert (
        await crud.get_peer_card(
            db_session, ws, observer=observer, observed=scenario.sender
        )
        is None
    )
    with pytest.raises(StaleLifecycleError):
        await crud.set_peer_card(
            db_session,
            ws,
            [f"ATTRIBUTE: lives in {CANARY}"],
            observer=observer,
            observed=scenario.sender,
            fence=fence,
        )
    assert (
        await crud.get_peer_card(
            db_session, ws, observer=observer, observed=scenario.sender
        )
        is None
    )


@pytest.mark.asyncio
async def test_deriver_batch_with_stale_fence_writes_nothing(db_session: AsyncSession):
    from src.deriver.deriver import process_representation_tasks_batch

    scenario = await _seed(db_session)
    ws = scenario.workspace
    pairs = [(observer, scenario.sender) for observer in scenario.observers]
    fence = await capture_fence(
        db_session, ws, session_name=scenario.session, representations=pairs
    )
    message = await db_session.scalar(
        select(models.Message).where(models.Message.id == scenario.message.id)
    )
    assert message is not None
    db_session.expunge(message)
    await db_session.commit()

    await erase_message(
        db_session,
        TenantContext(workspace_name=ws),
        MessageIdentity(scenario.session, scenario.message.public_id),
    )
    before = await _count(
        db_session, select(models.Document).where(models.Document.workspace_name == ws)
    )

    llm_response = SimpleNamespace(
        content=PromptRepresentation(
            explicit=[ExplicitObservationBase(content=f"lives in {CANARY}")]
        ),
        output_tokens=5,
        input_tokens=50,
        finish_reasons=["stop"],
        hit_input_token_cap=False,
    )
    with (
        patch(
            "src.deriver.deriver.honcho_llm_call",
            AsyncMock(return_value=llm_response),
        ),
        pytest.raises(StaleLifecycleError),
    ):
        await process_representation_tasks_batch(
            [message],
            schemas.ResolvedConfiguration(
                reasoning=schemas.ResolvedReasoningConfiguration(enabled=True),
                peer_card=schemas.ResolvedPeerCardConfiguration(use=True, create=True),
                summary=schemas.ResolvedSummaryConfiguration(
                    enabled=True,
                    messages_per_short_summary=20,
                    messages_per_long_summary=60,
                ),
                dream=schemas.ResolvedDreamConfiguration(enabled=False),
            ),
            observers=scenario.observers,
            observed=scenario.sender,
            queue_item_message_ids=[scenario.message.id],
            session_id="unused",
            fence=fence,
        )

    after = await _count(
        db_session, select(models.Document).where(models.Document.workspace_name == ws)
    )
    assert after == before

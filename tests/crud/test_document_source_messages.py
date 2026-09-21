"""document_source_messages: written from DocumentCreate, dropped with either end."""

from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.models import Peer, Workspace


async def _seed(
    db_session: AsyncSession, workspace: Workspace, peer: Peer
) -> tuple[str, list[int], list[str]]:
    """Seed a session with two messages; returns (session_name, row ids, public ids)."""
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()
    messages = [
        models.Message(
            session_name=session.name,
            workspace_name=workspace.name,
            peer_name=peer.name,
            content=f"message {seq}",
            seq_in_session=seq,
        )
        for seq in (1, 2)
    ]
    db_session.add_all(messages)
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    row_ids = [m.id for m in messages]
    public_ids = [m.public_id for m in messages]
    await db_session.commit()
    return session.name, row_ids, public_ids


def _document_create(
    session_name: str, source_message_ids: list[str]
) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content="peer said something",
        session_name=session_name,
        level="explicit",
        metadata=schemas.DocumentMetadata(
            message_ids=[1], message_created_at="2026-01-01T00:00:00Z"
        ),
        embedding=[0.1] * 1536,
        source_message_ids=source_message_ids,
    )


async def _only_document_id(db_session: AsyncSession, workspace_name: str) -> str:
    result = await db_session.execute(
        select(models.Document.id).where(
            models.Document.workspace_name == workspace_name
        )
    )
    return result.scalar_one()


async def _edges(db_session: AsyncSession, derived_id: str) -> list[str]:
    result = await db_session.execute(
        select(models.DocumentSourceMessage.message_id)
        .where(models.DocumentSourceMessage.derived_id == derived_id)
        .order_by(models.DocumentSourceMessage.position)
    )
    return list(result.scalars().all())


async def test_create_documents_writes_citation_edges(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    # Duplicate and malformed entries are dropped; order is preserved.
    await crud.create_documents(
        db_session,
        [_document_create(session_name, [ids[1], ids[0], ids[1], "not-a-message-id"])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
    )
    await db_session.commit()

    doc_id = await _only_document_id(db_session, workspace_name)
    assert await _edges(db_session, doc_id) == [ids[1], ids[0]]

    db_session.expire_all()
    doc = (await crud.get_documents_by_ids(db_session, workspace_name, [doc_id]))[0]
    assert doc.source_message_ids == [ids[1], ids[0]]


async def test_citation_edges_follow_message_and_document_deletion(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, row_ids, ids = await _seed(db_session, workspace, peer)

    await crud.create_documents(
        db_session,
        [_document_create(session_name, ids)],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
    )
    await db_session.commit()
    doc_id = await _only_document_id(db_session, workspace_name)

    # Deleting a cited message drops only its edge; the conclusion survives.
    await db_session.execute(
        delete(models.Message).where(models.Message.id == row_ids[0])
    )
    await db_session.commit()
    db_session.expire_all()
    assert await _edges(db_session, doc_id) == [ids[1]]
    doc = (await crud.get_documents_by_ids(db_session, workspace_name, [doc_id]))[0]
    assert doc.source_message_ids == [ids[1]]

    # Deleting the conclusion drops the rest.
    await db_session.execute(
        delete(models.Document).where(models.Document.id == doc_id)
    )
    await db_session.commit()
    assert await _edges(db_session, doc_id) == []

"""conclusion_source_messages: written from DocumentCreate, merged on dedup, dropped with either end."""

import pytest
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
    session_name: str,
    source_message_ids: list[str],
    content: str = "peer said something",
) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
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


async def _edges(db_session: AsyncSession, conclusion_id: str) -> list[str]:
    result = await db_session.execute(
        select(models.ConclusionSourceMessage.message_id)
        .where(models.ConclusionSourceMessage.conclusion_id == conclusion_id)
        .order_by(models.ConclusionSourceMessage.position)
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


async def _create(
    db_session: AsyncSession,
    workspace_name: str,
    peer_name: str,
    documents: list[schemas.DocumentCreate],
    *,
    deduplicate: bool = False,
) -> crud.CreateDocumentsResult:
    result = await crud.create_documents(
        db_session,
        documents,
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        deduplicate=deduplicate,
    )
    await db_session.commit()
    db_session.expire_all()
    return result


async def _live_documents(
    db_session: AsyncSession, workspace_name: str
) -> list[models.Document]:
    result = await db_session.execute(
        select(models.Document).where(
            models.Document.workspace_name == workspace_name,
            models.Document.deleted_at.is_(None),
        )
    )
    return list(result.scalars().all())


async def test_reinforcing_an_existing_conclusion_merges_its_citations(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[0]])],
    )
    result = await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[1], ids[0]])],
    )

    assert result.exact_dup_existing_count == 1
    [doc] = await _live_documents(db_session, workspace_name)
    assert doc.times_derived == 2
    # Existing citations keep their positions; new ones follow.
    assert await _edges(db_session, doc.id) == [ids[0], ids[1]]


async def test_in_batch_duplicates_merge_their_citations(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    result = await _create(
        db_session,
        workspace_name,
        peer_name,
        [
            _document_create(session_name, [ids[1]]),
            _document_create(session_name, [ids[0], ids[1]]),
        ],
    )

    assert result.exact_dup_in_batch_count == 1
    [doc] = await _live_documents(db_session, workspace_name)
    assert await _edges(db_session, doc.id) == [ids[1], ids[0]]


async def test_in_batch_duplicate_drops_malformed_citations(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    await _create(
        db_session,
        workspace_name,
        peer_name,
        [
            _document_create(session_name, [ids[0]]),
            _document_create(session_name, ["not-a-message-id", ids[1]]),
        ],
    )

    [doc] = await _live_documents(db_session, workspace_name)
    assert await _edges(db_session, doc.id) == [ids[0], ids[1]]


async def test_in_batch_duplicates_of_an_existing_conclusion_merge_into_it(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[0]])],
    )
    result = await _create(
        db_session,
        workspace_name,
        peer_name,
        [
            _document_create(session_name, [ids[0]]),
            _document_create(session_name, [ids[1]]),
        ],
    )

    assert result.exact_dup_existing_count == 1
    assert result.exact_dup_in_batch_count == 1
    [doc] = await _live_documents(db_session, workspace_name)
    assert await _edges(db_session, doc.id) == [ids[0], ids[1]]


async def test_semantic_rejection_merges_citations_into_the_kept_conclusion(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[0]], "Ann likes coffee and tea")],
    )
    result = await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[1]], "Ann likes coffee")],
        deduplicate=True,
    )

    assert result.semantic_dup_rejected_count == 1
    [doc] = await _live_documents(db_session, workspace_name)
    assert doc.content == "Ann likes coffee and tea"
    assert await _edges(db_session, doc.id) == [ids[0], ids[1]]


async def test_semantic_replacement_inherits_the_replaced_citations(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[0]], "Ann likes coffee")],
    )
    result = await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[1]], "Ann likes coffee and tea")],
        deduplicate=True,
    )

    assert result.semantic_dup_replaced_count == 1
    [doc] = await _live_documents(db_session, workspace_name)
    assert doc.content == "Ann likes coffee and tea"
    assert await _edges(db_session, doc.id) == [ids[0], ids[1]]


@pytest.mark.parametrize("replacement_first", [False, True])
async def test_replacement_keeps_citations_from_a_reinforce_of_the_same_row(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    replacement_first: bool,
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, _row_ids, ids = await _seed(db_session, workspace, peer)

    await _create(
        db_session,
        workspace_name,
        peer_name,
        [_document_create(session_name, [ids[0]], "Ann likes coffee")],
    )
    # One batch both reinforces the existing row (exact match) and replaces
    # it (a richer semantic match), in either order.
    reinforce = _document_create(session_name, [ids[1]], "Ann likes coffee")
    replace = _document_create(session_name, [], "Ann likes coffee and tea")
    batch = [replace, reinforce] if replacement_first else [reinforce, replace]
    result = await _create(
        db_session, workspace_name, peer_name, batch, deduplicate=True
    )

    assert result.semantic_dup_replaced_count == 1
    assert result.exact_dup_existing_count == 1
    [doc] = await _live_documents(db_session, workspace_name)
    assert doc.content == "Ann likes coffee and tea"
    assert await _edges(db_session, doc.id) == [ids[0], ids[1]]

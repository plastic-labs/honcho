"""Tombstoned messages, and messages and conclusions under tombstoned parents,
are hidden from reads.

Nothing sets ``deleted_at`` yet, so these tests set it directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.crud.message import (
    _fetch_messages_by_ids,  # pyright: ignore[reportPrivateUsage]
)
from src.crud.representation import RepresentationManager
from src.utils.agent_tools import get_observation_context
from src.utils.search import fetch_messages_by_ids, search

# Every row shares one embedding so semantic queries match all of them, and
# only the lifecycle filters decide what comes back.
EMBEDDING = [1.0] + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 1)

Target = Literal["none", "message", "author", "session", "workspace"]


@dataclass
class Graph:
    workspace: str
    session: str
    authored: str
    replied: str


async def _seed(db: AsyncSession) -> Graph:
    workspace = str(generate_nanoid())
    db.add(models.Workspace(name=workspace))
    await db.flush()
    db.add_all(
        [
            models.Peer(name=name, workspace_name=workspace)
            for name in ("author", "bystander")
        ]
        + [models.Session(name="session", workspace_name=workspace)]
    )
    await db.flush()
    authored = models.Message(
        workspace_name=workspace,
        session_name="session",
        peer_name="author",
        content="needle from the author",
        seq_in_session=1,
    )
    replied = models.Message(
        workspace_name=workspace,
        session_name="session",
        peer_name="bystander",
        content="needle from the bystander",
        seq_in_session=2,
    )
    db.add_all([authored, replied])
    await db.flush()
    db.add_all(
        [
            models.MessageEmbedding(
                content=message.content,
                embedding=EMBEDDING,
                message_id=message.public_id,
                workspace_name=workspace,
                session_name="session",
                peer_name=message.peer_name,
                sync_state="synced",
            )
            for message in (authored, replied)
        ]
    )
    await db.commit()
    return Graph(
        workspace=workspace,
        session="session",
        authored=authored.public_id,
        replied=replied.public_id,
    )


async def _tombstone(db: AsyncSession, graph: Graph, target: Target) -> None:
    now = datetime.now(UTC)
    match target:
        case "none":
            return
        case "message":
            stmt = update(models.Message).where(
                models.Message.public_id == graph.authored
            )
        case "author":
            stmt = update(models.Peer).where(
                models.Peer.workspace_name == graph.workspace,
                models.Peer.name == "author",
            )
        case "session":
            stmt = update(models.Session).where(
                models.Session.workspace_name == graph.workspace,
                models.Session.name == graph.session,
            )
        case "workspace":
            stmt = update(models.Workspace).where(
                models.Workspace.name == graph.workspace
            )
    await db.execute(stmt.values(deleted_at=now))
    await db.commit()


def _expected(graph: Graph, target: Target) -> set[str]:
    expected: dict[Target, set[str]] = {
        "none": {graph.authored, graph.replied},
        "message": {graph.replied},
        "author": {graph.replied},
        "session": set(),
        "workspace": set(),
    }
    return expected[target]


def _snippet_ids(
    snippets: list[tuple[list[models.Message], list[models.Message]]],
) -> tuple[set[str], set[str]]:
    matched = {m.public_id for matches, _ in snippets for m in matches}
    context = {m.public_id for _, ctx in snippets for m in ctx}
    return matched, context


TARGETS: list[Target] = ["none", "message", "author", "session", "workspace"]


@pytest.mark.parametrize("target", TARGETS)
async def test_message_reads_hide_tombstones(db_session: AsyncSession, target: Target):
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, target)
    expected = _expected(graph, target)
    ws, session = graph.workspace, graph.session

    for stmt in (
        await crud.get_messages(ws, session),
        await crud.get_messages(ws, session, token_limit=10_000),
        await crud.get_messages(ws, session, message_count_limit=10),
    ):
        listed = {m.public_id for m in (await db_session.scalars(stmt)).all()}
        assert listed == expected

    fetched = await crud.get_message(db_session, ws, session, graph.authored)
    assert (fetched is not None) == (graph.authored in expected)

    by_id = await crud.get_messages_id_range(db_session, ws, session)
    assert {m.public_id for m in by_id} == expected
    by_seq = await crud.get_messages_by_seq_range(db_session, ws, session)
    assert {m.public_id for m in by_seq} == expected

    by_date = await crud.get_messages_by_date_range(db_session, ws, session)
    assert {m.public_id for m in by_date} == expected
    by_date_ws = await crud.get_messages_by_date_range(db_session, ws, None)
    assert {m.public_id for m in by_date_ws} == expected

    matched, context = _snippet_ids(await crud.grep_messages(ws, None, "needle"))
    assert matched == expected
    assert context == expected

    context_rows = await get_observation_context(
        db_session, ws, None, [graph.authored, graph.replied]
    )
    assert {m.public_id for m in context_rows} == expected

    matched, context = _snippet_ids(
        await crud.search_messages(ws, None, "needle", embedding=EMBEDDING)
    )
    assert matched == expected
    assert context == expected

    found = await search("needle", filters={"workspace_id": ws})
    assert {m.public_id for m in found} == expected

    # External vector stores return ids; the database fetch decides liveness.
    ids = [graph.authored, graph.replied]
    assert {m.public_id for m in await _fetch_messages_by_ids(db_session, ws, ids)} == (
        expected
    )
    assert {m.public_id for m in await fetch_messages_by_ids(db_session, ids)} == (
        expected
    )


async def test_semantic_search_hides_tombstones(db_session: AsyncSession):
    """Hybrid search must not let the semantic leg reintroduce a hidden row."""
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, "message")

    # A query the full-text leg cannot match, so only the semantic leg answers.
    found = await search(
        "zzz-no-lexical-match", filters={"workspace_id": graph.workspace}
    )
    assert {m.public_id for m in found} == {graph.replied}


async def test_grep_context_skips_a_tombstoned_neighbour(db_session: AsyncSession):
    """A live match keeps its snippet; a tombstoned neighbour drops out of it."""
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, "message")

    matched, context = _snippet_ids(
        await crud.grep_messages(graph.workspace, None, "bystander")
    )
    assert matched == {graph.replied}
    assert context == {graph.replied}


async def test_observation_context_skips_neighbours_of_a_tombstoned_target(
    db_session: AsyncSession,
):
    """A hidden target must not pull its live neighbours into the context."""
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, "message")

    rows = await get_observation_context(
        db_session, graph.workspace, None, [graph.authored]
    )
    assert rows == []


async def test_clone_copies_only_live_messages(db_session: AsyncSession):
    graph = await _seed(db_session)
    await _tombstone(db_session, graph, "message")

    clone = await crud.clone_session(db_session, graph.workspace, graph.session)
    copied = (
        await db_session.scalars(await crud.get_messages(graph.workspace, clone.name))
    ).all()
    assert [m.content for m in copied] == ["needle from the bystander"]

    # A tombstoned cutoff message is not found, like any other missing one.
    with pytest.raises(ValueError):
        await crud.clone_session(
            db_session, graph.workspace, graph.session, graph.authored
        )


async def _seed_conclusions(db: AsyncSession, graph: Graph) -> tuple[str, str]:
    db.add(
        models.Collection(
            workspace_name=graph.workspace, observer="author", observed="author"
        )
    )
    await db.flush()
    sessionless = models.Document(
        workspace_name=graph.workspace,
        observer="author",
        observed="author",
        session_name=None,
        content="from nowhere in particular",
        level="explicit",
        embedding=EMBEDDING,
    )
    db.add(sessionless)
    await db.flush()
    # Derived from the sessionless conclusion, so it is also its child.
    in_session = models.Document(
        workspace_name=graph.workspace,
        observer="author",
        observed="author",
        session_name=graph.session,
        content="from the session",
        level="explicit",
        embedding=EMBEDDING,
        legacy_source_ids=[sessionless.id],
    )
    db.add(in_session)
    await db.commit()
    return in_session.id, sessionless.id


@pytest.mark.parametrize(
    ("target", "visible"),
    [
        ("none", {"in_session", "sessionless"}),
        ("session", {"sessionless"}),
        ("author", set[str]()),
        ("workspace", set[str]()),
    ],
)
async def test_conclusion_reads_hide_tombstoned_parents(
    db_session: AsyncSession, target: Target, visible: set[str]
):
    graph = await _seed(db_session)
    in_session, sessionless = await _seed_conclusions(db_session, graph)
    await _tombstone(db_session, graph, target)
    names = {in_session: "in_session", sessionless: "sessionless"}

    stmt = crud.get_all_documents(graph.workspace, observer="author", observed="author")
    listed = {names[d.id] for d in (await db_session.scalars(stmt)).all()}
    assert listed == visible

    recent = await crud.query_documents_recent(
        db_session, graph.workspace, observer="author", observed="author"
    )
    assert {names[d.id] for d in recent} == visible

    ws = graph.workspace
    ids = [in_session, sessionless]

    def seen(docs: Sequence[models.Document]) -> set[str]:
        return {names[d.id] for d in docs}

    assert seen(await crud.get_documents_by_ids(db_session, ws, ids)) == visible
    assert (
        seen(
            await crud.fetch_documents_by_ids(
                db_session, ws, document_ids=ids, observer="author", observed="author"
            )
        )
        == visible
    )
    assert (
        seen(
            (
                await db_session.scalars(
                    crud.get_documents_with_filters(ws, filters={"observer": "author"})
                )
            ).all()
        )
        == visible
    )
    assert (
        seen(
            await crud.query_documents_most_derived(
                db_session, ws, observer="author", observed="author"
            )
        )
        == visible
    )
    explicit, _ = await crud.query_explicit_documents_after(
        db_session,
        ws,
        after=None,
        since=None,
        limit=10,
        observer="author",
        observed="author",
    )
    assert seen(explicit) == visible
    summary = await crud.summarize_documents_by_level(
        db_session, ws, observer="author", observed="author"
    )
    assert sum(summary.counts_by_level.values()) == len(visible)
    assert (
        seen(
            await crud.query_documents(
                db_session,
                ws,
                "anything",
                embedding=EMBEDDING,
                top_k=10,
                observer="author",
                observed="author",
            )
        )
        == visible
    )

    children = (
        await db_session.scalars(crud.get_child_observations(ws, sessionless))
    ).all()
    assert seen(children) == visible & {"in_session"}

    manager = RepresentationManager(ws, observer="author", observed="author")
    assert seen(await manager._query_documents_recent(db_session, 10)) == visible  # pyright: ignore[reportPrivateUsage]
    assert seen(await manager._query_documents_most_derived(db_session, 10)) == visible  # pyright: ignore[reportPrivateUsage]
    assert (
        seen(
            await manager._query_documents_semantic(  # pyright: ignore[reportPrivateUsage]
                db_session, "anything", 10, embedding=EMBEDDING
            )
        )
        == visible
    )


async def test_conclusion_reads_hide_either_tombstoned_peer(db_session: AsyncSession):
    """Tombstoning a peer hides conclusions it observes and conclusions about it."""
    graph = await _seed(db_session)
    ws = graph.workspace
    ids: dict[str, str] = {}
    for observer, observed in (("author", "bystander"), ("bystander", "author")):
        db_session.add(
            models.Collection(workspace_name=ws, observer=observer, observed=observed)
        )
        await db_session.flush()
        document = models.Document(
            workspace_name=ws,
            observer=observer,
            observed=observed,
            content=f"{observer} about {observed}",
            level="explicit",
        )
        db_session.add(document)
        await db_session.flush()
        ids[document.id] = f"{observer}->{observed}"
    await db_session.commit()

    def visible(docs: Sequence[models.Document]) -> set[str]:
        return {ids[d.id] for d in docs}

    by_id = await crud.get_documents_by_ids(db_session, ws, list(ids))
    assert visible(by_id) == {"author->bystander", "bystander->author"}

    await _tombstone(db_session, graph, "author")
    assert visible(await crud.get_documents_by_ids(db_session, ws, list(ids))) == set()
    stmt = crud.get_documents_with_filters(ws, filters={"observer": "bystander"})
    assert visible((await db_session.scalars(stmt)).all()) == set()

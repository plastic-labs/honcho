"""Representation History: recording a dream's changes and reading them back."""

import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.dream_history import DreamHistoryTarget
from src.exceptions import ResourceNotFoundException
from src.models import Peer, Workspace


async def _session(
    db_session: AsyncSession, workspace_name: str, peer_name: str
) -> str:
    """An empty session and the peer's self-collection; returns the session name."""
    session_name = str(generate_nanoid())
    db_session.add(models.Session(name=session_name, workspace_name=workspace_name))
    db_session.add(
        models.Collection(
            workspace_name=workspace_name, observer=peer_name, observed=peer_name
        )
    )
    await db_session.commit()
    return session_name


async def _seed(
    db_session: AsyncSession, workspace_name: str, peer_name: str
) -> tuple[str, list[str]]:
    """A session and two explicit conclusions about the peer; returns (session, ids)."""
    session_name = await _session(db_session, workspace_name, peer_name)
    doc_ids = [str(generate_nanoid()) for _ in range(2)]
    db_session.add_all(
        models.Document(
            id=doc_id,
            workspace_name=workspace_name,
            observer=peer_name,
            observed=peer_name,
            content=content,
            level="explicit",
            session_name=session_name,
            embedding=[0.1 * (i + 1)] * 1536,
        )
        for i, (doc_id, content) in enumerate(
            zip(doc_ids, ["Ann drinks coffee daily", "Ann works nights"], strict=True)
        )
    )
    await db_session.commit()
    return session_name, doc_ids


async def _open(workspace_name: str, peer_name: str) -> DreamHistoryTarget:
    target = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=peer_name,
        observed=peer_name,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is not None
    return DreamHistoryTarget(run_id=target.run_id, specialist="deduction")


def _deductive(content: str, source_ids: list[str]) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
        session_name=None,
        level="deductive",
        metadata=schemas.DocumentMetadata(
            message_ids=[], message_created_at="2026-01-01T00:00:00Z"
        ),
        embedding=[0.9] * 1536,
        source_ids=source_ids,
    )


async def _hard_delete(db_session: AsyncSession, doc_id: str) -> None:
    """What the reconciler does to a soft-deleted conclusion a few minutes later."""
    await db_session.execute(
        delete(models.Document).where(models.Document.id == doc_id)
    )
    await db_session.commit()


async def test_run_lifecycle(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    target = await _open(workspace_name, peer_name)

    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.status == "running"
    assert detail.finished_at is None

    await crud.close_run(target.run_id, succeeded=False)
    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.status == "failed"
    assert detail.finished_at is not None

    # Closing is one-way: a second close doesn't rewrite the outcome.
    await crud.close_run(target.run_id, succeeded=True)
    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.status == "failed"


async def test_history_disabled_opens_nothing(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    monkeypatch.setattr(settings.DREAM, "HISTORY_ENABLED", False)

    target = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=peer_name,
        observed=peer_name,
        dream_type=None,
        trigger_reason=None,
    )

    assert target is None
    assert (await db_session.scalar(select(models.DreamRun.id))) is None


async def test_created_conclusion_keeps_sources_readable_after_they_are_removed(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    first = await _open(workspace_name, peer_name)

    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=first,
    )

    # A later dream removes one of the sources, and the reconciler hard-deletes it.
    second = await _open(workspace_name, peer_name)
    await crud.delete_documents(
        db_session,
        workspace_name,
        [coffee_id],
        observer=peer_name,
        observed=peer_name,
        history=second,
    )
    await _hard_delete(db_session, coffee_id)

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, first.run_id)
    assert detail.created_count == 1
    [created] = detail.created
    assert created.conclusion.content == "Ann relies on caffeine to work nights"
    assert created.conclusion.status == "current"
    assert created.specialist == "deduction"
    coffee, nights = created.sources
    assert (coffee.id, coffee.status, coffee.removed_in_run) == (
        coffee_id,
        "removed",
        second.run_id,
    )
    assert coffee.content == "Ann drinks coffee daily"
    assert (nights.status, nights.content) == ("current", "Ann works nights")

    removal = await crud.get_run_detail(db_session, workspace_name, second.run_id)
    assert removal.removed_count == 1
    [removed] = removal.removed
    assert (removed.id, removed.content, removed.level) == (
        coffee_id,
        "Ann drinks coffee daily",
        "explicit",
    )


async def test_semantic_replacement_is_recorded_as_a_removal(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    # No seeded conclusions: every test embedding is parallel, so any of them
    # would be an equally close duplicate candidate.
    session_name = await _session(db_session, workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [
            schemas.DocumentCreate(
                content="Ann likes coffee",
                session_name=session_name,
                level="explicit",
                metadata=schemas.DocumentMetadata(
                    message_ids=[1], message_created_at="2026-01-01T00:00:00Z"
                ),
                embedding=[0.5] * 1536,
            )
        ],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
    )
    old_id = await db_session.scalar(
        select(models.Document.id).where(models.Document.content == "Ann likes coffee")
    )
    target = await _open(workspace_name, peer_name)

    result = await crud.create_documents(
        db_session,
        [
            schemas.DocumentCreate(
                content="Ann likes coffee and tea",
                session_name=session_name,
                level="explicit",
                metadata=schemas.DocumentMetadata(
                    message_ids=[1], message_created_at="2026-01-01T00:00:00Z"
                ),
                embedding=[0.5] * 1536,
            )
        ],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        deduplicate=True,
        history=target,
    )

    assert result.semantic_dup_replaced_count == 1
    new_id = await db_session.scalar(
        select(models.Document.id).where(
            models.Document.content == "Ann likes coffee and tea"
        )
    )
    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert [c.conclusion.id for c in detail.created] == [new_id]
    [removed] = detail.removed
    assert (removed.id, removed.content, removed.replaced_by) == (
        old_id,
        "Ann likes coffee",
        new_id,
    )
    assert (detail.created_count, detail.removed_count) == (1, 1)


async def test_peer_card_change_records_before_and_after(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    target = await _open(workspace_name, peer_name)

    await crud.set_peer_card(
        db_session,
        workspace_name,
        ["IDENTITY: Name: Ann"],
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    await crud.set_peer_card(
        db_session,
        workspace_name,
        ["IDENTITY: Name: Ann", "ATTRIBUTE: Works nights"],
        observer=peer_name,
        observed=peer_name,
        history=target,
    )

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.peer_card_updated
    assert detail.peer_card == schemas.DreamPeerCardChange(
        before=None, after=["IDENTITY: Name: Ann", "ATTRIBUTE: Works nights"]
    )


async def test_writes_without_history_record_nothing(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, _nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )

    await crud.delete_documents(
        db_session,
        workspace_name,
        [coffee_id],
        observer=peer_name,
        observed=peer_name,
    )
    await crud.set_peer_card(
        db_session,
        workspace_name,
        ["IDENTITY: Name: Ann"],
        observer=peer_name,
        observed=peer_name,
    )

    assert (await db_session.scalar(select(models.DreamRunChange.id))) is None


async def test_deleting_the_session_deletes_its_removed_copies(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, (coffee_id, _nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    target = await _open(workspace_name, peer_name)
    await crud.delete_documents(
        db_session,
        workspace_name,
        [coffee_id],
        observer=peer_name,
        observed=peer_name,
        history=target,
    )

    await crud.delete_session(db_session, workspace_name, session_name)

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.removed == []


async def test_a_conclusion_deleted_through_the_api_is_unavailable(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    target = await _open(workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    created_id = await db_session.scalar(
        select(models.Document.id).where(models.Document.level == "deductive")
    )
    assert created_id is not None

    await crud.delete_document_by_id(db_session, workspace_name, created_id)
    await db_session.commit()

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    [created] = detail.created
    assert created.conclusion.status == "unavailable"
    assert created.conclusion.content is None


async def test_deleting_a_restored_conclusion_erases_the_copy_a_dream_kept(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    first = await _open(workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=first,
    )
    second = await _open(workspace_name, peer_name)
    await crud.delete_documents(
        db_session,
        workspace_name,
        [coffee_id],
        observer=peer_name,
        observed=peer_name,
        history=second,
    )
    # Something restores the same conclusion id (as a scope backfill can).
    await db_session.execute(
        update(models.Document)
        .where(models.Document.id == coffee_id)
        .values(deleted_at=None)
    )
    await db_session.commit()

    await crud.delete_document_by_id(db_session, workspace_name, coffee_id)

    db_session.expire_all()
    removal = await crud.get_run_detail(db_session, workspace_name, second.run_id)
    assert removal.removed == []
    detail = await crud.get_run_detail(db_session, workspace_name, first.run_id)
    coffee = next(s for s in detail.created[0].sources if s.id == coffee_id)
    assert (coffee.status, coffee.content) == ("unavailable", None)


async def test_deleting_the_workspace_deletes_its_history(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, _nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    target = await _open(workspace_name, peer_name)
    await crud.delete_documents(
        db_session,
        workspace_name,
        [coffee_id],
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    await db_session.execute(
        update(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .values(is_active=False)
    )
    await db_session.commit()

    await crud.delete_workspace(db_session, workspace_name)

    assert (await db_session.scalar(select(models.DreamRun.id))) is None
    assert (await db_session.scalar(select(models.DreamRunChange.id))) is None


async def test_prune_deletes_old_runs_and_interrupts_stale_ones(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    old = await _open(workspace_name, peer_name)
    stale = await _open(workspace_name, peer_name)
    fresh = await _open(workspace_name, peer_name)
    now = datetime.datetime.now(datetime.UTC)
    retention = datetime.timedelta(days=settings.DREAM.HISTORY_RETENTION_DAYS)
    await db_session.execute(
        update(models.DreamRun)
        .where(models.DreamRun.id == old.run_id)
        .values(started_at=now - retention - datetime.timedelta(hours=1))
    )
    await db_session.execute(
        update(models.DreamRun)
        .where(models.DreamRun.id == stale.run_id)
        .values(started_at=now - datetime.timedelta(days=2))
    )
    await db_session.commit()

    assert await crud.prune_dream_history() == 1

    db_session.expire_all()
    with pytest.raises(ResourceNotFoundException):
        await crud.get_run_detail(db_session, workspace_name, old.run_id)
    stale_detail = await crud.get_run_detail(db_session, workspace_name, stale.run_id)
    assert stale_detail.status == "interrupted"
    fresh_detail = await crud.get_run_detail(db_session, workspace_name, fresh.run_id)
    assert fresh_detail.status == "running"


async def _age_past_retention(db_session: AsyncSession, run_id: str) -> None:
    now = datetime.datetime.now(datetime.UTC)
    retention = datetime.timedelta(days=settings.DREAM.HISTORY_RETENTION_DAYS)
    await db_session.execute(
        update(models.DreamRun)
        .where(models.DreamRun.id == run_id)
        .values(started_at=now - retention - datetime.timedelta(hours=1))
    )
    await db_session.commit()


async def test_prune_keeps_what_a_dream_created_while_it_exists(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    target = await _open(workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    await crud.delete_documents(
        db_session,
        workspace_name,
        [coffee_id],
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    await crud.set_peer_card(
        db_session,
        workspace_name,
        ["IDENTITY: Name: Ann"],
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    await _age_past_retention(db_session, target.run_id)

    assert await crud.prune_dream_history() == 0

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.pruned_at is not None
    [created] = detail.created
    assert created.conclusion.content == "Ann relies on caffeine to work nights"
    assert created.conclusion.status == "current"
    assert detail.removed == []
    assert detail.peer_card is None
    # The counts are what the dream did, not what is left.
    assert (detail.created_count, detail.removed_count) == (1, 1)
    assert detail.peer_card_updated

    # Once that conclusion is gone, the next prune deletes the run.
    await crud.delete_documents(
        db_session,
        workspace_name,
        [created.conclusion.id],
        observer=peer_name,
        observed=peer_name,
    )
    assert await crud.prune_dream_history() == 1
    db_session.expire_all()
    with pytest.raises(ResourceNotFoundException):
        await crud.get_run_detail(db_session, workspace_name, target.run_id)


async def test_prune_leaves_a_run_with_nothing_to_drop_unmarked(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    target = await _open(workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=target,
    )
    await _age_past_retention(db_session, target.run_id)

    assert await crud.prune_dream_history() == 0

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, target.run_id)
    assert detail.pruned_at is None
    assert len(detail.created) == 1


async def test_prune_walks_every_batch(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    _session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    kept = await _open(workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=kept,
    )
    empty = [await _open(workspace_name, peer_name) for _ in range(3)]
    for run in [kept, *empty]:
        await _age_past_retention(db_session, run.run_id)
    monkeypatch.setattr("src.crud.dream_history._PRUNE_BATCH_SIZE", 1)

    assert await crud.prune_dream_history() == 3

    db_session.expire_all()
    remaining = (
        await db_session.scalars(
            select(models.DreamRun.id).where(
                models.DreamRun.workspace_name == workspace_name
            )
        )
    ).all()
    assert remaining == [kept.run_id]


async def test_get_run_detail_is_scoped_to_its_workspace(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    target = await _open(workspace_name, peer_name)
    other_name = str(generate_nanoid())
    db_session.add(models.Workspace(name=other_name))
    await db_session.commit()

    with pytest.raises(ResourceNotFoundException):
        await crud.get_run_detail(db_session, other_name, target.run_id)


# Nothing sets `deleted_at` on a workspace, peer or session yet, so these
# tombstone directly. History must read as it would after a hard deletion.


async def _tombstone(
    db_session: AsyncSession,
    model: type[models.Peer] | type[models.Session],
    workspace_name: str,
    name: str,
) -> None:
    await db_session.execute(
        update(model)
        .where(model.workspace_name == workspace_name, model.name == name)
        .values(deleted_at=datetime.datetime.now(datetime.UTC))
    )
    await db_session.commit()


async def test_a_tombstoned_session_hides_its_conclusions_and_copies(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session_name, (coffee_id, nights_id) = await _seed(
        db_session, workspace_name, peer_name
    )
    first = await _open(workspace_name, peer_name)
    await crud.create_documents(
        db_session,
        [_deductive("Ann relies on caffeine to work nights", [coffee_id, nights_id])],
        workspace_name,
        observer=peer_name,
        observed=peer_name,
        history=first,
    )
    second = await _open(workspace_name, peer_name)
    await crud.delete_documents(
        db_session,
        workspace_name,
        [nights_id],
        observer=peer_name,
        observed=peer_name,
        history=second,
    )

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, first.run_id)
    assert [s.status for s in detail.created[0].sources] == ["current", "removed"]

    await _tombstone(db_session, models.Session, workspace_name, session_name)

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, first.run_id)
    [created] = detail.created
    # The deductive conclusion has no session, so the session doesn't hide it.
    assert created.conclusion.status == "current"
    assert [(s.id, s.status, s.content) for s in created.sources] == [
        (coffee_id, "unavailable", None),
        (nights_id, "unavailable", None),
    ]
    removal = await crud.get_run_detail(db_session, workspace_name, second.run_id)
    assert removal.removed == []


@pytest.mark.parametrize("deleted", ["observer", "observed"])
async def test_a_tombstoned_peer_hides_the_runs_about_it(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    deleted: str,
):
    workspace, peer = sample_data
    workspace_name = workspace.name
    observer, other = "observer-" + str(generate_nanoid()), peer.name
    db_session.add(models.Peer(name=observer, workspace_name=workspace_name))
    await db_session.commit()
    target = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=observer,
        observed=peer.name,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is not None
    unrelated = await _open(workspace_name, other) if deleted == "observer" else None

    await _tombstone(
        db_session,
        models.Peer,
        workspace_name,
        observer if deleted == "observer" else peer.name,
    )

    listed = {
        run.id
        for run in (await db_session.scalars(crud.list_runs_stmt(workspace_name)))
    }
    assert target.run_id not in listed
    if unrelated is not None:
        assert unrelated.run_id in listed
    with pytest.raises(ResourceNotFoundException):
        await crud.get_run_detail(db_session, workspace_name, target.run_id)


async def test_a_tombstoned_workspace_hides_its_runs(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    target = await _open(workspace.name, peer.name)

    await db_session.execute(
        update(models.Workspace)
        .where(models.Workspace.name == workspace.name)
        .values(deleted_at=datetime.datetime.now(datetime.UTC))
    )
    await db_session.commit()

    assert (await db_session.scalars(crud.list_runs_stmt(workspace.name))).all() == []
    with pytest.raises(ResourceNotFoundException):
        await crud.get_run_detail(db_session, workspace.name, target.run_id)


async def test_a_removed_copy_from_a_hidden_run_is_unavailable(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, ann = workspace.name, peer.name
    assistant = "assistant-" + str(generate_nanoid())
    db_session.add(models.Peer(name=assistant, workspace_name=workspace_name))
    await db_session.flush()
    await _session(db_session, workspace_name, ann)  # ann's self-collection
    db_session.add(
        models.Collection(
            workspace_name=workspace_name, observer=assistant, observed=ann
        )
    )
    source_id = str(generate_nanoid())
    db_session.add(
        models.Document(
            id=source_id,
            workspace_name=workspace_name,
            observer=assistant,
            observed=ann,
            content="Ann mentioned a night shift",
            level="explicit",
            session_name=None,
            embedding=[0.3] * 1536,
        )
    )
    await db_session.commit()

    # A run on ann's own representation cites a source from another pair.
    cites = await _open(workspace_name, ann)
    await crud.create_documents(
        db_session,
        [_deductive("Ann works nights", [source_id])],
        workspace_name,
        observer=ann,
        observed=ann,
        history=cites,
    )
    removal = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=assistant,
        observed=ann,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert removal is not None
    await crud.delete_documents(
        db_session,
        workspace_name,
        [source_id],
        observer=assistant,
        observed=ann,
        history=DreamHistoryTarget(run_id=removal.run_id, specialist="deduction"),
    )

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, cites.run_id)
    [source] = detail.created[0].sources
    assert (source.status, source.content) == ("removed", "Ann mentioned a night shift")

    await _tombstone(db_session, models.Peer, workspace_name, assistant)

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, cites.run_id)
    [source] = detail.created[0].sources
    assert (source.status, source.content) == ("unavailable", None)

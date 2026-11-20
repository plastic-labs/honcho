"""Following a source live: edits and deletes in the dry run, its end, `auto`, sync states."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.config import settings
from src.crud import ingest as ingest_crud
from src.crud.ingest_fence import take_workspace_fence


def _workspace(client: TestClient, *, auto: bool = True) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name, "auto": auto})
    assert response.status_code in (200, 201), response.text
    return name


def _record(source_id: str = "m1", **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": "discord",
        "source_instance": "guild1",
        "source_id": source_id,
        "created_at": "2026-11-04T20:47:00Z",
        "revised_at": "2026-11-04T20:47:00Z",
        "container": {"id": "chan-1", "name": "dev"},
        "author": {"id": "user-1", "name": "Jessica"},
        "content": "Northstar needs SSO before rollout.",
    }
    record.update(overrides)
    return record


def _delete(source_id: str = "m1", **overrides: Any) -> dict[str, Any]:
    return {
        "op": "delete",
        "source": "discord",
        "source_instance": "guild1",
        "source_id": source_id,
        **overrides,
    }


def _ingest(
    client: TestClient, workspace: str, *records: dict[str, Any], derive: str = "later"
) -> list[dict[str, Any]]:
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"derive": derive, "records": list(records)},
    )
    assert response.status_code == 200, response.text
    return response.json()["records"]


def _messages(client: TestClient, workspace: str, session: str) -> list[str]:
    response = client.post(
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list", json={}
    )
    assert response.status_code == 200, response.text
    return [m["content"] for m in response.json()["items"]]


def _derive(client: TestClient, workspace: str) -> None:
    response = client.post(f"/v3/workspaces/{workspace}/ingest/derive")
    assert response.status_code == 200, response.text


async def _derived_at(db: AsyncSession, workspace: str) -> Any:
    return await db.scalar(
        select(models.Workspace.ingest_derived_at).where(
            models.Workspace.name == workspace
        )
    )


# ---------------------------------------------------------------------------
# Edits and deletes during the dry run
# ---------------------------------------------------------------------------


def test_an_edit_in_the_dry_run_replaces_the_text(client: TestClient) -> None:
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())

    [edited] = _ingest(
        client,
        workspace,
        _record(revised_at="2026-11-04T21:00:00Z", content="Northstar needs SSO."),
    )

    assert edited["outcome"] == "updated"
    assert edited["message_id"] == created["message_id"]
    assert edited["derivation"] == "deferred"
    assert _messages(client, workspace, created["session_id"]) == [
        "Northstar needs SSO."
    ]
    # An older version is stale; the same content again is a duplicate.
    [stale] = _ingest(client, workspace, _record(content="older"))
    assert (stale["outcome"], stale["reason"]) == ("rejected", "stale_revision")


def test_a_newer_revision_with_the_same_text_moves_the_revision_forward(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record())
    [bumped] = _ingest(client, workspace, _record(revised_at="2026-11-05T00:00:00Z"))
    assert bumped["outcome"] == "duplicate"

    # An edit older than the bump is stale: the bump is the newest version.
    [stale] = _ingest(
        client,
        workspace,
        _record(revised_at="2026-11-04T23:00:00Z", content="in between"),
    )
    assert (stale["outcome"], stale["reason"]) == ("rejected", "stale_revision")


def test_a_delete_in_the_dry_run_removes_the_message_and_keeps_a_tombstone(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    created, _ = _ingest(client, workspace, _record(), _record("m2", content="kept"))

    [deleted] = _ingest(client, workspace, _delete())

    assert deleted["outcome"] == "deleted"
    assert deleted["derivation"] == "deleted"
    assert _messages(client, workspace, created["session_id"]) == ["kept"]
    # It never comes back, even as a newer version; deleting again changes nothing.
    [again] = _ingest(client, workspace, _record(revised_at="2026-12-01T00:00:00Z"))
    assert (again["outcome"], again["reason"]) == ("rejected", "deleted")
    [repeat] = _ingest(client, workspace, _delete())
    assert repeat["outcome"] == "duplicate"


def test_a_delete_for_a_record_never_seen_blocks_it_later(client: TestClient) -> None:
    workspace = _workspace(client)
    [tombstone] = _ingest(client, workspace, _delete("m9"))
    assert (tombstone["outcome"], tombstone["session_id"]) == ("deleted", None)

    [refused] = _ingest(client, workspace, _record("m9"))
    assert (refused["outcome"], refused["reason"]) == ("rejected", "deleted")


def test_a_delete_older_than_the_stored_version_changes_nothing(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    [stale] = _ingest(client, workspace, _delete(revised_at="2026-11-01T00:00:00Z"))
    assert (stale["outcome"], stale["reason"]) == ("rejected", "stale_revision")
    assert len(_messages(client, workspace, created["session_id"])) == 1


def test_one_identity_in_a_batch_is_processed_once_at_its_newest(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    first, edited = _ingest(
        client,
        workspace,
        _record(),
        _record(revised_at="2026-11-04T21:00:00Z", content="edited at once"),
    )
    assert (first["outcome"], first["reason"]) == ("rejected", "superseded")
    assert edited["outcome"] == "created"
    assert _messages(client, workspace, edited["session_id"]) == ["edited at once"]

    # A delete beats an upsert at the same revision.
    upsert, delete = _ingest(
        client,
        workspace,
        _record("m3"),
        _delete("m3", revised_at="2026-11-04T20:47:00Z"),
    )
    assert (upsert["reason"], delete["outcome"]) == ("superseded", "deleted")


def test_records_that_arent_valid_deletes_are_refused(client: TestClient) -> None:
    workspace = _workspace(client)
    for record in (
        {"op": "delete", "source": "discord", "source_instance": "guild1"},
        {**_record(), "op": "edit"},
    ):
        response = client.post(
            f"/v3/workspaces/{workspace}/ingest", json={"records": [record]}
        )
        assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_an_edit_is_refused_once_its_message_has_embedding_rows(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    # Deferred records aren't embedded at ingest.
    assert not await db_session.scalar(
        select(func.count())
        .select_from(models.MessageEmbedding)
        .where(models.MessageEmbedding.message_id == created["message_id"])
    )
    db_session.add(
        models.MessageEmbedding(
            content="Northstar",
            message_id=created["message_id"],
            workspace_name=workspace,
            session_name=created["session_id"],
            peer_name=created["peer_id"],
            sync_state="pending",
        )
    )
    await db_session.commit()

    [edit] = _ingest(
        client, workspace, _record(revised_at="2026-11-05T00:00:00Z", content="x")
    )
    [delete] = _ingest(client, workspace, _delete())
    assert edit["reason"] == delete["reason"] == "edited_after_derivation"


@pytest.mark.asyncio
async def test_released_records_get_their_embedding_rows(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "EMBED_MESSAGES", True)
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())

    _derive(client, workspace)

    rows = await db_session.scalar(
        select(func.count())
        .select_from(models.MessageEmbedding)
        .where(models.MessageEmbedding.message_id == created["message_id"])
    )
    assert rows and rows > 0


# ---------------------------------------------------------------------------
# The end of the dry run, and derive: auto
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deriving_ends_the_dry_run_and_edits_are_refused_after(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    assert await _derived_at(db_session, workspace) is None

    _derive(client, workspace)

    db_session.expire_all()
    assert await _derived_at(db_session, workspace) is not None
    [edit] = _ingest(
        client, workspace, _record(revised_at="2026-11-05T00:00:00Z", content="x")
    )
    [delete] = _ingest(client, workspace, _delete())
    assert edit["reason"] == delete["reason"] == "edited_after_derivation"
    assert _messages(client, workspace, created["session_id"]) == [_record()["content"]]


@pytest.mark.asyncio
async def test_derive_now_ends_the_dry_run(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record(), derive="now")
    assert await _derived_at(db_session, workspace) is not None


@pytest.mark.asyncio
async def test_a_message_through_the_message_routes_ends_the_dry_run(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    response = client.post(
        f"/v3/workspaces/{workspace}/sessions/{created['session_id']}/messages",
        json={"messages": [{"peer_id": created["peer_id"], "content": "by hand"}]},
    )
    assert response.status_code in (200, 201), response.text

    assert await _derived_at(db_session, workspace) is not None
    [edit] = _ingest(
        client, workspace, _record(revised_at="2026-11-05T00:00:00Z", content="x")
    )
    assert edit["reason"] == "edited_after_derivation"


def test_auto_waits_in_the_dry_run_and_derives_after_it(client: TestClient) -> None:
    workspace = _workspace(client)
    [before] = _ingest(client, workspace, _record(), derive="auto")
    assert before["derivation"] == "deferred"

    _derive(client, workspace)

    [after] = _ingest(client, workspace, _record("m2"), derive="auto")
    assert after["derivation"] == "queued"


@pytest.mark.asyncio
async def test_a_batch_that_changes_records_waits_for_every_other_ingest_writer(
    db_engine: AsyncEngine, client: TestClient
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record())
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    edit = schemas.IngestBatch.model_validate(
        {
            "derive": "later",
            "records": [_record(revised_at="2026-11-05T00:00:00Z", content="edited")],
        }
    )
    async with factory() as holder, factory() as db:
        # Another batch, or a derive call, holding the shared fence.
        await take_workspace_fence(holder, workspace, mode="shared")
        await db.execute(text("SET LOCAL lock_timeout = '200ms'"))
        with pytest.raises(DBAPIError, match="lock timeout"):
            await ingest_crud.ingest_batch(db, workspace, edit)
        await db.rollback()
        await holder.rollback()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_keeps_tombstones_and_forgets_sync_states(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record(), _record("m2"))
    _ingest(client, workspace, _delete())
    saved = client.put(
        f"/v3/workspaces/{workspace}/ingest/sync/discord/guild1",
        json={"position": {"chan-1": "m2"}},
    )
    assert saved.status_code == 200, saved.text

    response = client.post(f"/v3/workspaces/{workspace}/ingest/reset")
    assert response.status_code == 202, response.text
    for name in response.json()["sessions"]:
        await crud.delete_session(db_session, workspace, name)
    await db_session.commit()

    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()
    assert status["sources"] == []
    deleted, kept = _ingest(client, workspace, _record(), _record("m2"))
    assert (deleted["reason"], kept["outcome"]) == ("deleted", "created")
    # A save naming the revision read before the reset is refused.
    stale = client.put(
        f"/v3/workspaces/{workspace}/ingest/sync/discord/guild1",
        json={"position": {}, "revision": saved.json()["revision"]},
    )
    assert stale.status_code == 409, stale.text


# ---------------------------------------------------------------------------
# Sync states
# ---------------------------------------------------------------------------


def test_a_sync_state_is_saved_read_and_forgotten(client: TestClient) -> None:
    workspace = _workspace(client)
    url = f"/v3/workspaces/{workspace}/ingest/sync/discord/guild1"
    assert client.get(url).status_code == 404

    first = client.put(url, json={"position": {"chan-1": "100"}})
    assert first.status_code == 200, first.text
    second = client.put(
        url,
        json={
            "position": {"chan-1": "200"},
            "revision": first.json()["revision"],
            "last_record_at": "2026-11-20T10:00:00Z",
        },
    )
    assert second.status_code == 200, second.text
    assert second.json()["revision"] != first.json()["revision"]

    # A second runner holding the first revision is refused.
    conflict = client.put(
        url, json={"position": {"chan-1": "150"}, "revision": first.json()["revision"]}
    )
    assert conflict.status_code == 409, conflict.text

    # A save without last_record_at keeps it; an error shows in the health.
    errored = client.put(url, json={"position": {"chan-1": "200"}, "error": "429"})
    state = client.get(url).json()
    assert state["position"] == {"chan-1": "200"}
    assert state["revision"] == errored.json()["revision"]
    assert state["last_record_at"].startswith("2026-11-20T10:00:00")
    [listed] = client.get(f"/v3/workspaces/{workspace}/ingest/sync").json()
    assert (listed["source"], listed["error"]) == ("discord", "429")
    assert "position" not in listed
    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()
    assert [s["source_instance"] for s in status["sources"]] == ["guild1"]

    assert client.delete(url).json() == {"forgotten": True}
    assert client.delete(url).json() == {"forgotten": False}
    assert client.get(url).status_code == 404


def test_a_sync_state_needs_an_auto_workspace_and_a_modest_position(
    client: TestClient,
) -> None:
    regular = _workspace(client, auto=False)
    response = client.put(
        f"/v3/workspaces/{regular}/ingest/sync/discord/guild1", json={"position": {}}
    )
    assert response.status_code == 409, response.text

    workspace = _workspace(client)
    huge = {"blob": "x" * 1_100_000}
    response = client.put(
        f"/v3/workspaces/{workspace}/ingest/sync/discord/guild1",
        json={"position": huge},
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_workspace_deletion_removes_its_sync_states(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    client.put(
        f"/v3/workspaces/{workspace}/ingest/sync/discord/guild1", json={"position": {}}
    )
    await crud.delete_workspace(db_session, workspace)
    await db_session.commit()
    remaining = await db_session.scalar(
        select(func.count())
        .select_from(models.IngestSyncState)
        .where(models.IngestSyncState.workspace_name == workspace)
    )
    assert remaining == 0


@pytest.mark.asyncio
async def test_a_dream_scheduled_by_hand_ends_the_dry_run(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    response = client.post(
        f"/v3/workspaces/{workspace}/schedule_dream",
        json={"observer": created["peer_id"], "dream_type": "omni"},
    )
    assert response.status_code in (200, 202, 204), response.text
    assert await _derived_at(db_session, workspace) is not None


@pytest.mark.asyncio
async def test_deletes_alone_dont_end_the_dry_run(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record())
    [deleted] = _ingest(client, workspace, _delete(), derive="now")
    assert deleted["outcome"] == "deleted"
    assert await _derived_at(db_session, workspace) is None


def test_status_counts_tombstones_apart_from_messages(client: TestClient) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record(), _record("m2"))
    _ingest(client, workspace, _delete(), _delete("never-seen"))

    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()
    assert status["records"]["deleted"] == 2
    assert status["records"]["deferred"] == 1
    [session] = client.get(f"/v3/workspaces/{workspace}/ingest/sessions").json()[
        "items"
    ]
    assert session["messages"] == 1


def test_the_newest_record_time_only_moves_forward(client: TestClient) -> None:
    workspace = _workspace(client)
    url = f"/v3/workspaces/{workspace}/ingest/sync/discord/guild1"
    client.put(url, json={"position": {}, "last_record_at": "2026-11-20T12:00:00Z"})
    client.put(url, json={"position": {}, "last_record_at": "2026-11-20T11:00:00Z"})
    assert client.get(url).json()["last_record_at"].startswith("2026-11-20T12:00:00")


@pytest.mark.asyncio
async def test_a_dream_refused_for_a_scope_doesnt_end_the_dry_run(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    response = client.post(
        f"/v3/workspaces/{workspace}/schedule_dream",
        json={
            "observer": created["peer_id"],
            "observed": "scope.missing",
            "dream_type": "omni",
        },
    )
    assert response.status_code == 422, response.text
    assert await _derived_at(db_session, workspace) is None


@pytest.mark.asyncio
async def test_memory_work_isnt_queued_for_a_workspace_created_again(
    client: TestClient, db_session: AsyncSession
) -> None:
    from src.crud.ingest_dry_run import end_dry_run
    from src.deriver.enqueue import enqueue_dream

    workspace = _workspace(client)
    [created] = _ingest(client, workspace, _record())
    old_id = await end_dry_run(db_session, workspace)
    assert old_id is not None
    # Deleted and created again, in a new dry run, before the dream is queued.
    await crud.delete_workspace(db_session, workspace)
    await db_session.commit()
    client.post("/v3/workspaces", json={"id": workspace, "auto": True})
    _ingest(client, workspace, _record())

    await enqueue_dream(
        workspace,
        observer=created["peer_id"],
        observed=created["peer_id"],
        dream_type=schemas.DreamType.OMNI,
        auto_workspace_id=old_id,
    )

    queued = await db_session.scalar(
        select(func.count())
        .select_from(models.QueueItem)
        .where(models.QueueItem.workspace_name == workspace)
    )
    assert queued == 0
    assert await _derived_at(db_session, workspace) is None

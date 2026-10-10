"""Public active-work contract, live cursors, and workspace authorization."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.security import JWTParams, create_jwt


async def add_work(
    db: AsyncSession,
    workspace: str,
    ident: str,
    *,
    task: str = "dream",
    session: models.Session | None = None,
    observers: list[str] | None = None,
    processed: bool = False,
) -> models.ActiveQueueSession:
    key = f"{task}:{workspace}:{ident}"
    db.add(
        models.QueueItem(
            work_unit_key=key,
            workspace_name=workspace,
            task_type=task,
            session_id=session.id if session else None,
            processed=processed,
            payload={
                "observer": "alice",
                "observed": "bob",
                "observers": observers or [],
            },
        )
    )
    claim = models.ActiveQueueSession(id=ident, work_unit_key=key)
    db.add(claim)
    await db.commit()
    return claim


@pytest.mark.asyncio
async def test_work_units_are_distinct_scoped_and_do_not_change_counts(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, _ = sample_data
    claim = await add_work(
        db_session, workspace.name, "attempt-a", task="representation"
    )
    # Several queue items can belong to one claim; status still counts items.
    db_session.add(
        models.QueueItem(
            work_unit_key=claim.work_unit_key,
            workspace_name=workspace.name,
            task_type="representation",
            payload={"observer": "later", "observed": "bob"},
            processed=False,
        )
    )
    await add_work(db_session, workspace.name, "attempt-b", task="summary")
    await add_work(db_session, workspace.name, "hidden", task="webhook")
    await add_work(db_session, workspace.name, "finished", processed=True)
    # Same peer names in a different workspace must not escape the workspace filter.
    other = models.Workspace(name="progress-other")
    db_session.add(other)
    await db_session.commit()
    await add_work(db_session, other.name, "other-workspace")
    url = f"/v3/workspaces/{workspace.name}/queue"
    response = client.get(f"{url}/work-units")
    assert response.status_code == 200
    rows = response.json()["items"]
    assert [r["id"] for r in rows] == ["attempt-a", "attempt-b"]
    assert all(r["workspace_id"] == workspace.name for r in rows)
    assert all(r["stage"] == "claimed" for r in rows)
    assert all(datetime.fromisoformat(r["started_at"]).tzinfo is not None for r in rows)
    assert (
        client.get(f"{url}/work-units", params={"observer_id": "later"}).json()["items"]
        == []
    )
    counts = client.get(f"{url}/status").json()
    assert counts["total_work_units"] == 4
    assert counts["completed_work_units"] == 1
    assert counts["in_progress_work_units"] == 3


@pytest.mark.asyncio
async def test_filters_intersect_and_preserve_multi_observer_scope(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, _ = sample_data
    session = models.Session(name="progress-session", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.commit()
    await add_work(
        db_session,
        workspace.name,
        "filtered",
        task="representation",
        session=session,
        observers=["alice", "carol"],
    )
    url = f"/v3/workspaces/{workspace.name}/queue/work-units"
    params = {
        "task_type": "representation",
        "observer_id": "carol",
        "sender_id": "bob",
        "session_id": session.name,
    }
    rows = client.get(url, params=params).json()["items"]
    assert len(rows) == 1
    assert rows[0]["observers"] == ["alice", "carol"]
    assert rows[0]["session_id"] == session.name
    for key, value in (
        ("task_type", "dream"),
        ("observer_id", "missing"),
        ("sender_id", "alice"),
        ("session_id", "missing"),
    ):
        assert client.get(url, params={**params, key: value}).json()["items"] == []


@pytest.mark.asyncio
async def test_cursor_survives_removed_anchor_and_finished_work(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, _ = sample_data
    for ident in ("a", "b", "c"):
        await add_work(db_session, workspace.name, ident)
    url = f"/v3/workspaces/{workspace.name}/queue/work-units"
    first = client.get(url, params={"size": 1}).json()
    assert [r["id"] for r in first["items"]] == ["a"]
    assert first["total"] is None
    await db_session.execute(
        delete(models.ActiveQueueSession).where(models.ActiveQueueSession.id == "a")
    )
    await db_session.commit()
    second = client.get(url, params={"size": 1, "cursor": first["next_page"]}).json()
    assert [r["id"] for r in second["items"]] == ["b"]
    await db_session.execute(
        update(models.QueueItem)
        .where(models.QueueItem.workspace_name == workspace.name)
        .values(processed=True)
    )
    await db_session.commit()
    empty = client.get(url, params={"size": 1, "cursor": second["next_page"]})
    assert empty.status_code == 200
    assert empty.json()["items"] == []
    assert empty.json()["next_page"] is None


@pytest.mark.parametrize(
    "params,code",
    [
        ({"size": 0}, 422),
        ({"size": 101}, 422),
        ({"task_type": "webhook"}, 422),
        ({"cursor": "not-a-cursor"}, 400),
    ],
)
def test_invalid_progress_parameters(
    client: TestClient, params: dict[str, object], code: int
) -> None:
    assert (
        client.get("/v3/workspaces/test/queue/work-units", params=params).status_code
        == code
    )


@pytest.mark.asyncio
async def test_progress_requires_workspace_or_admin_key(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, peer = sample_data
    await add_work(db_session, workspace.name, "private")
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(
        settings.AUTH, "JWT_SECRET", "progress-test-secret-not-a-real-key"
    )
    url = f"/v3/workspaces/{workspace.name}/queue/work-units"
    for claims in (
        JWTParams(w="other"),
        JWTParams(w=workspace.name, p=peer.name),
        JWTParams(w=workspace.name, s="session"),
    ):
        response = client.get(
            url, headers={"Authorization": f"Bearer {create_jwt(claims)}"}
        )
        assert response.status_code == 401
    for claims in (JWTParams(w=workspace.name), JWTParams(ad=True)):
        response = client.get(
            url, headers={"Authorization": f"Bearer {create_jwt(claims)}"}
        )
        assert response.status_code == 200
        assert response.json()["items"][0]["id"] == "private"


@pytest.mark.asyncio
async def test_scoped_dream_uses_payload_session_with_workspace_isolation(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    from src.deriver.enqueue import create_dream_record
    from src.schemas import DreamType

    workspace, peer = sample_data
    session = models.Session(name="scoped-dream", workspace_name=workspace.name)
    other = models.Workspace(name="other-dream-scope")
    db_session.add_all([session, other])
    await db_session.commit()
    db_session.add(models.Session(name=session.name, workspace_name=other.name))
    for ws, ident in ((workspace.name, "scoped"), (other.name, "foreign")):
        record = create_dream_record(
            ws,
            observer=peer.name,
            observed=peer.name,
            dream_type=DreamType.OMNI,
            session_name=session.name,
        )
        assert record["session_id"] is None
        db_session.add(models.QueueItem(**record))
        db_session.add(
            models.ActiveQueueSession(id=ident, work_unit_key=record["work_unit_key"])
        )
    await db_session.commit()
    url = f"/v3/workspaces/{workspace.name}/queue/work-units"
    response = client.get(
        url, params={"session_id": session.name, "observer_id": peer.name}
    )
    assert response.status_code == 200
    assert [(r["id"], r["session_id"]) for r in response.json()["items"]] == [
        ("scoped", session.name)
    ]
    assert client.get(url, params={"session_id": "missing"}).json()["items"] == []


@pytest.mark.asyncio
async def test_stage_scope_is_bound_to_actual_batch_not_enqueue_order(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    from sqlalchemy import select

    from src.utils.work_unit_progress import work_unit_progress

    workspace, _ = sample_data
    claim = await add_work(
        db_session,
        workspace.name,
        "batch-scope",
        task="representation",
        observers=["queued"],
    )
    # Real modern representation payloads have observers and no single observer.
    later = models.QueueItem(
        work_unit_key=claim.work_unit_key,
        workspace_name=workspace.name,
        task_type="representation",
        processed=False,
        payload={"observers": ["running", "scope_team"], "observed": "bob"},
    )
    db_session.add(later)
    await db_session.commit()
    # This is the exact item chosen by the worker (which can differ from enqueue order).
    await work_unit_progress(claim.work_unit_key, claim.id, later.id)("representation")
    url = f"/v3/workspaces/{workspace.name}/queue/work-units"
    rows = client.get(url, params={"observer_id": "running"}).json()["items"]
    assert len(rows) == 1 and rows[0]["observer"] is None
    assert rows[0]["observers"] == ["running", "scope_team"]
    assert client.get(url, params={"observer_id": "queued"}).json()["items"] == []
    later.processed = True
    await db_session.commit()
    # Do not attribute the old stage to the next queued item between batches.
    assert client.get(url).json()["items"] == []
    pending = (
        await db_session.execute(
            select(models.QueueItem).where(
                models.QueueItem.work_unit_key == claim.work_unit_key,
                models.QueueItem.processed.is_(False),
            )
        )
    ).scalar_one()
    await work_unit_progress(claim.work_unit_key, claim.id, pending.id)(
        "representation"
    )
    assert len(client.get(url, params={"observer_id": "queued"}).json()["items"]) == 1

"""Representation History routes: listing dream runs and reading one."""

import datetime

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt


async def _run(
    db_session: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    started_minutes_ago: int,
) -> str:
    target = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is not None
    await db_session.execute(
        update(models.DreamRun)
        .where(models.DreamRun.id == target.run_id)
        .values(
            started_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(minutes=started_minutes_ago)
        )
    )
    await db_session.commit()
    return target.run_id


async def test_list_filters_by_peer_newest_first(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    workspace, peer = sample_data
    workspace_name, maya = workspace.name, peer.name
    db_session.add(models.Peer(name="assistant", workspace_name=workspace_name))
    await db_session.commit()
    older = await _run(
        db_session, workspace_name, observer=maya, observed=maya, started_minutes_ago=10
    )
    newer = await _run(
        db_session, workspace_name, observer=maya, observed=maya, started_minutes_ago=1
    )
    from_other_observer = await _run(
        db_session,
        workspace_name,
        observer="assistant",
        observed=maya,
        started_minutes_ago=5,
    )
    await _run(
        db_session,
        workspace_name,
        observer="assistant",
        observed="assistant",
        started_minutes_ago=2,
    )
    root = f"/v3/workspaces/{workspace_name}/dreams"

    response = client.post(f"{root}/list", json={"observed_id": maya})
    assert response.status_code == 200
    page = response.json()
    assert [run["id"] for run in page["items"]] == [newer, from_other_observer, older]
    first = page["items"][0]
    assert (first["observer_id"], first["observed_id"], first["status"]) == (
        maya,
        maya,
        "running",
    )

    response = client.post(
        f"{root}/list", json={"observed_id": maya, "observer_id": maya}
    )
    assert [run["id"] for run in response.json()["items"]] == [newer, older]

    response = client.post(f"{root}/list?cursor=&size=2", json={"observed_id": maya})
    assert response.status_code == 200
    assert [run["id"] for run in response.json()["items"]] == [
        newer,
        from_other_observer,
    ]


async def test_get_returns_the_run_and_its_changes(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    workspace, peer = sample_data
    workspace_name, maya = workspace.name, peer.name
    run_id = await _run(
        db_session, workspace_name, observer=maya, observed=maya, started_minutes_ago=1
    )
    await crud.set_peer_card(
        db_session,
        workspace_name,
        ["IDENTITY: Name: Maya"],
        observer=maya,
        observed=maya,
        history=crud.DreamHistoryTarget(run_id=run_id, specialist="induction"),
    )
    await crud.close_run(run_id, succeeded=True)
    root = f"/v3/workspaces/{workspace_name}/dreams"

    response = client.get(f"{root}/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run_id
    assert body["status"] == "completed"
    assert body["peer_card_updated"] is True
    assert body["peer_card"] == {"before": None, "after": ["IDENTITY: Name: Maya"]}
    assert body["created"] == []
    assert body["removed"] == []

    assert client.get(f"{root}/{generate_nanoid()}").status_code == 404


async def test_peer_and_session_keys_cannot_read_history(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    workspace_name, maya = workspace.name, peer.name
    run_id = await _run(
        db_session, workspace_name, observer=maya, observed=maya, started_minutes_ago=1
    )
    root = f"/v3/workspaces/{workspace_name}/dreams"
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")

    for params in (
        JWTParams(w=workspace_name, p=maya),
        JWTParams(w=workspace_name, s="some-session"),
    ):
        client.headers["Authorization"] = f"Bearer {create_jwt(params)}"
        assert client.post(f"{root}/list", json={}).status_code == 401
        assert client.get(f"{root}/{run_id}").status_code == 401

    client.headers["Authorization"] = (
        f"Bearer {create_jwt(JWTParams(w=workspace_name))}"
    )
    assert client.post(f"{root}/list", json={}).status_code == 200
    assert client.get(f"{root}/{run_id}").status_code == 200


@pytest.mark.parametrize("field", ["observed_id", "observer_id"])
async def test_list_rejects_a_lone_surrogate_in_a_filter(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    field: str,
):
    workspace, _peer = sample_data

    response = client.post(
        f"/v3/workspaces/{workspace.name}/dreams/list",
        content='{"' + field + '": "\\ud800"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


async def test_deleting_a_removed_conclusion_through_the_api_erases_its_copy(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    workspace, peer = sample_data
    workspace_name, maya = workspace.name, peer.name
    run_id = await _run(
        db_session, workspace_name, observer=maya, observed=maya, started_minutes_ago=1
    )
    conclusion_id = generate_nanoid()
    db_session.add(
        models.DreamRunChange(
            run_id=run_id,
            workspace_name=workspace_name,
            kind="removed",
            conclusion_id=conclusion_id,
            level="explicit",
            content="Maya drinks coffee daily",
        )
    )
    await db_session.commit()

    response = client.delete(
        f"/v3/workspaces/{workspace_name}/conclusions/{conclusion_id}"
    )
    assert response.status_code == 204

    detail = client.get(f"/v3/workspaces/{workspace_name}/dreams/{run_id}").json()
    assert detail["removed"] == []
    # Deleting it again finds nothing left to delete.
    response = client.delete(
        f"/v3/workspaces/{workspace_name}/conclusions/{conclusion_id}"
    )
    assert response.status_code == 404


async def test_list_rejects_nul_in_a_filter(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
):
    workspace, _peer = sample_data

    response = client.post(
        f"/v3/workspaces/{workspace.name}/dreams/list",
        json={"observed_id": "maya\x00"},
    )

    assert response.status_code == 422

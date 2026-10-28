"""Tests for the scope list's session counts and name filter."""

from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


def _create_scope(client: TestClient, ws: str, scope_name: str) -> None:
    """Create a scope in the workspace (get-or-create)."""
    response = client.post(f"/v3/workspaces/{ws}/scopes", json={"id": scope_name})
    assert response.status_code in [200, 201], response.text


def _create_sessions(client: TestClient, ws: str, count: int) -> list[str]:
    """Create count sessions with random names; returns their names in creation order."""
    names = [str(generate_nanoid()) for _ in range(count)]
    for name in names:
        response = client.post(f"/v3/workspaces/{ws}/sessions", json={"id": name})
        assert response.status_code in [200, 201], response.text
    return names


def _add(client: TestClient, ws: str, scope_name: str, sessions: list[str]) -> None:
    """Add sessions to a scope via the facade, asserting the 204."""
    response = client.post(
        f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions",
        json={"session_ids": sessions},
    )
    assert response.status_code == 204, response.text


def _list_scopes(
    client: TestClient,
    ws: str,
    body: dict[str, Any] | None = None,
    **query: Any,
):
    """POST the workspace scope list with an optional body and query params."""
    return client.post(f"/v3/workspaces/{ws}/scopes/list", json=body, params=query)


def _counts(response: Any) -> dict[str, int | None]:
    """Map each listed scope ID to its session_count (None when omitted)."""
    assert response.status_code == 200, response.text
    return {item["id"]: item.get("session_count") for item in response.json()["items"]}


def _member_count(client: TestClient, ws: str, scope_name: str) -> int:
    """The scope's member total as scopes/{id}/sessions/list reports it."""
    response = client.post(f"/v3/workspaces/{ws}/scopes/{scope_name}/sessions/list")
    assert response.status_code == 200, response.text
    return response.json()["total"]


async def test_scope_list_session_counts(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Counts follow the member list: departed, deleted and archived members don't count."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    busy, empty = str(generate_nanoid()), str(generate_nanoid())
    for scope_name in (busy, empty):
        _create_scope(client, ws, scope_name)
    kept_1, kept_2, departed, deleted, archived = _create_sessions(client, ws, 5)
    _add(client, ws, busy, [kept_1, kept_2, departed, deleted, archived])
    response = client.delete(f"/v3/workspaces/{ws}/scopes/{busy}/sessions/{departed}")
    assert response.status_code == 204
    await db_session.execute(
        update(models.Session)
        .where(models.Session.workspace_name == ws)
        .where(models.Session.name == deleted)
        .values(deleted_at=datetime.now(UTC))
    )
    await db_session.execute(
        update(models.Session)
        .where(models.Session.workspace_name == ws)
        .where(models.Session.name == archived)
        .values(is_active=False)
    )
    await db_session.commit()

    counts = _counts(_list_scopes(client, ws, include_session_count=True))
    assert counts[busy] == 2
    assert counts[empty] == 0
    for scope_name in (busy, empty):
        assert counts[scope_name] == _member_count(client, ws, scope_name)

    # Omitting the option leaves the response shape unchanged
    response = _list_scopes(client, ws)
    assert response.status_code == 200, response.text
    assert all("session_count" not in item for item in response.json()["items"])


def test_scope_list_session_counts_stay_in_their_workspace(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """The same scope and session names in another workspace never show up.

    Session names are shared too, so a membership joined to the other
    workspace's same-named session would double-count.
    """
    test_workspace, _ = sample_data
    ws = test_workspace.name
    other_ws = str(generate_nanoid())
    response = client.post("/v3/workspaces", json={"id": other_ws})
    assert response.status_code in [200, 201], response.text
    scope_name = str(generate_nanoid())
    shared, only_other_1, only_other_2 = (str(generate_nanoid()) for _ in range(3))
    for workspace_name in (ws, other_ws):
        _create_scope(client, workspace_name, scope_name)
    for workspace_name, sessions in (
        (ws, [shared]),
        (other_ws, [shared, only_other_1, only_other_2]),
    ):
        for session_name in sessions:
            response = client.post(
                f"/v3/workspaces/{workspace_name}/sessions", json={"id": session_name}
            )
            assert response.status_code in [200, 201], response.text
        _add(client, workspace_name, scope_name, sessions)

    by_name = {"filters": {"id": scope_name}}
    for workspace_name, expected in ((ws, 1), (other_ws, 3)):
        response = _list_scopes(
            client, workspace_name, by_name, include_session_count=True
        )
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 1
        assert [
            (item["id"], item["session_count"]) for item in response.json()["items"]
        ] == [(scope_name, expected)]


def test_scope_list_session_counts_on_every_page(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Counts and the name filter carry through every cursor page."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    prefix = str(generate_nanoid(size=8)).replace("-", "_")
    scopes = {f"{prefix}{n}": n for n in range(1, 4)}
    for scope_name, size in scopes.items():
        _create_scope(client, ws, scope_name)
        _add(client, ws, scope_name, _create_sessions(client, ws, size))
    by_prefix = {"filters": {"id": {"icontains": prefix}}}

    seen: dict[str, int | None] = {}
    cursor: str | None = ""
    while cursor is not None:
        response = _list_scopes(
            client, ws, by_prefix, cursor=cursor, size=1, include_session_count=True
        )
        seen |= _counts(response)
        cursor = response.json()["next_page"]
        assert len(seen) <= len(scopes)
    assert seen == scopes


def test_scope_list_name_filter(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """`id` filters on the unprefixed name, never on the reserved prefix."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    # Hex after a letter, so the tag can never spell a substring the filters
    # below match on ("sup", "t2", "billing", "scope")
    tag = "x" + str(generate_nanoid("0123456789abcdef", 8))
    support, support_t2, billing = f"support{tag}", f"support{tag}t2", f"billing{tag}"
    # Matches none of the filters below, so an unfiltered total would differ
    unrelated = str(generate_nanoid(size=8)).replace("-", "_")
    for scope_name in (support, support_t2, billing, unrelated):
        _create_scope(client, ws, scope_name)

    def ids(filters: dict[str, Any]) -> set[str]:
        """IDs of the scopes the filter returns in this workspace."""
        response = _list_scopes(client, ws, {"filters": filters})
        assert response.status_code == 200, response.text
        return {item["id"] for item in response.json()["items"]}

    def ours(filters: dict[str, Any]) -> set[str]:
        """Like ids, but restricted to this test's tagged scopes."""
        return ids({"AND": [{"id": {"icontains": tag}}, filters]})

    assert ours({"id": {"icontains": "SUP"}}) == {support, support_t2}
    assert ours({"id": {"contains": "billing"}}) == {billing}
    assert ids({"id": billing}) == {billing}
    assert ids({"id": {"in": [support, billing]}}) == {support, billing}
    assert ours({"NOT": [{"id": {"icontains": "sup"}}]}) == {billing}
    assert ours({"OR": [{"id": billing}, {"id": {"icontains": "t2"}}]}) == {
        billing,
        support_t2,
    }
    # The reserved prefix is not part of the name
    assert ours({"id": {"icontains": "scope"}}) == set()
    assert ours({"id": {"icontains": "."}}) == set()
    assert ids({"id": f"scope.{billing}"}) == set()

    response = _list_scopes(client, ws, {"filters": {"id": {"icontains": tag}}})
    assert response.json()["total"] == 3
    assert _list_scopes(client, ws).json()["total"] >= 4


def test_scope_list_rejects_unknown_filter_fields(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Fields a scope can't be filtered on, including internal ones, are a 422."""
    test_workspace, _ = sample_data
    ws = test_workspace.name
    for field in ("peer_id", "session_id", "configuration", "internal_metadata"):
        response = _list_scopes(client, ws, {"filters": {field: "x"}})
        assert response.status_code == 422, (field, response.text)

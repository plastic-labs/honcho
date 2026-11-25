"""The conclusion graph routes answer for the caller's workspace."""

import datetime
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from arro3.core import Table
from arro3.io import read_ipc_stream
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt
from src.telemetry import prometheus_metrics
from src.utils.graph_arrow import MEDIA_TYPE
from src.utils.scopes import scope_peer_name

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
TEAM = scope_peer_name("team")
MISSING_ID = "z" * 21


async def _seed(db: AsyncSession, workspace: Workspace, alice: Peer) -> dict[str, str]:
    """Seed one graph and return each conclusion's ID by name."""
    ws, me = workspace.name, alice.name
    db.add_all(
        [
            models.Peer(name="bob", workspace_name=ws),
            models.Peer(
                name=TEAM, workspace_name=ws, internal_metadata={"kind": "scope"}
            ),
            models.Session(name="s1", workspace_name=ws),
        ]
    )
    await db.flush()
    db.add_all(
        [
            models.Collection(workspace_name=ws, observer=me, observed="bob"),
            models.Collection(workspace_name=ws, observer=me, observed=me),
            models.Collection(workspace_name=ws, observer=TEAM, observed="bob"),
        ]
    )
    await db.flush()

    def doc(minutes: int, **fields: Any) -> models.Document:
        # workspace_name comes first: the source_ids setter reads it
        values: dict[str, Any] = {
            "workspace_name": ws,
            "observer": me,
            "observed": "bob",
            "content": f"conclusion at minute {minutes}",
            "created_at": T0 + datetime.timedelta(minutes=minutes),
        }
        return models.Document(**(values | fields))

    root = doc(0, session_name="s1")
    other = doc(1, observed=me)
    deleted = doc(2, deleted_at=T0)
    db.add_all([root, other, deleted])
    await db.flush()
    derived = doc(3, level="deductive", times_derived=2, source_ids=[root.id, other.id])
    partial = doc(4, level="inductive", source_ids=[deleted.id, MISSING_ID])
    lonely = doc(5, level="deductive")
    copy = doc(6, observer=TEAM, internal_metadata={"copied_from": root.id})
    db.add_all([derived, partial, lonely, copy])
    await db.commit()
    named = {
        "root": root,
        "other": other,
        "deleted": deleted,
        "derived": derived,
        "partial": partial,
        "lonely": lonely,
        "copy": copy,
    }
    return {name: d.id for name, d in named.items()}


def _values(column: Any) -> list[Any]:
    # arro3's stub says to_pylist returns an ndarray; it returns a list
    values: object = column.to_pylist()
    assert isinstance(values, list)
    return cast("list[Any]", values)


def _decode(body: bytes) -> dict[str, list[Any]]:
    table = Table.from_batches(list(read_ipc_stream(body)))
    return {name: _values(table.column(name)) for name in table.column_names}


def _names(graph: dict[str, Any], ids: dict[str, str]) -> list[str]:
    by_id = {doc_id: name for name, doc_id in ids.items()}
    return [by_id[doc_id] for doc_id in graph["id"]]


def _use_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, params: JWTParams
) -> None:
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    client.headers["Authorization"] = f"Bearer {create_jwt(params)}"


async def test_whole_workspace_graph(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    ids = await _seed(db_session, workspace, alice)

    response = client.get(f"/v3/workspaces/{workspace.name}/graph")
    assert response.status_code == 200
    assert response.headers["content-type"] == MEDIA_TYPE
    graph = _decode(response.content)

    order = _names(graph, ids)
    assert order == ["root", "other", "derived", "partial", "lonely", "copy"]
    at = {name: k for k, name in enumerate(order)}
    assert graph["created_at"][at["root"]] == T0
    assert graph["session"][at["root"]] == "s1"
    assert graph["session"][at["other"]] is None
    assert graph["observed"][at["other"]] == alice.name
    assert graph["observer"][at["copy"]] == TEAM
    assert graph["observer_kind"][at["copy"]] == "scope"
    assert graph["observer_kind"][at["root"]] == "peer"
    assert graph["times_derived"][at["derived"]] == 2

    assert graph["sources"][at["derived"]] == [at["root"], at["other"]]
    assert graph["lineage"][at["derived"]] == "complete"

    p = at["partial"]
    unresolved = (
        graph["unresolved_outside"][p],
        graph["unresolved_deleted"][p],
        graph["unresolved_unknown"][p],
    )
    assert unresolved == (0, 1, 1)
    assert graph["lineage"][p] == "partial"
    assert graph["lineage"][at["lonely"]] == "unknown"
    assert graph["lineage"][at["root"]] == "not_applicable"

    assert graph["is_copy"][at["copy"]] is True
    assert graph["copied_from"][at["copy"]] == at["root"]
    assert MISSING_ID.encode() not in response.content
    assert ids["deleted"].encode() not in response.content


async def test_sources_outside_the_filter_count_as_outside(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    ids = await _seed(db_session, workspace, alice)

    response = client.get(
        f"/v3/workspaces/{workspace.name}/graph", params={"observed": "bob"}
    )
    graph = _decode(response.content)

    order = _names(graph, ids)
    assert "other" not in order
    d = order.index("derived")
    unresolved = (
        graph["unresolved_outside"][d],
        graph["unresolved_deleted"][d],
        graph["unresolved_unknown"][d],
    )
    assert unresolved == (1, 0, 0)
    assert graph["lineage"][d] == "partial"
    assert ids["other"].encode() not in response.content


async def test_a_copy_without_its_original_has_no_copied_from(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    ids = await _seed(db_session, workspace, alice)

    response = client.get(
        f"/v3/workspaces/{workspace.name}/graph", params={"observer": TEAM}
    )
    graph = _decode(response.content)

    assert _names(graph, ids) == ["copy"]
    assert graph["is_copy"] == [True]
    assert graph["copied_from"] == [None]
    assert graph["observer_kind"] == ["scope"]


async def test_matching_etags_return_304(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    await _seed(db_session, workspace, alice)
    url = f"/v3/workspaces/{workspace.name}/graph"
    etag = client.get(url).headers["etag"]

    for header in (etag, f"W/{etag}", f'"other", {etag}'):
        response = client.get(url, headers={"If-None-Match": header})
        assert response.status_code == 304, header
        assert response.content == b""
    assert client.get(url, headers={"If-None-Match": '"other"'}).status_code == 200


async def test_a_graph_over_the_limit_is_refused(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, alice = sample_data
    await _seed(db_session, workspace, alice)
    url = f"/v3/workspaces/{workspace.name}/graph"
    built, refused = MagicMock(), MagicMock()
    monkeypatch.setattr(prometheus_metrics, "record_graph_built", built)
    monkeypatch.setattr(prometheus_metrics, "record_graph_refused", refused)

    # the seed has six live conclusions
    monkeypatch.setattr(settings.GRAPH, "MAX_CONCLUSIONS", 6)
    assert client.get(url).status_code == 200
    assert built.call_args.kwargs["conclusions"] == 6
    assert set(built.call_args.kwargs["stages"]) == {
        "probe",
        "wait",
        "db",
        "build",
        "encode",
    }
    refused.assert_not_called()

    monkeypatch.setattr(settings.GRAPH, "MAX_CONCLUSIONS", 5)
    response = client.get(url)
    assert response.status_code == 422
    assert "more than 5 conclusions" in response.json()["detail"]
    refused.assert_called_once_with()
    assert client.get(url, params={"observer": TEAM}).status_code == 200


async def test_overview_counts_live_conclusions_per_perspective(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    await _seed(db_session, workspace, alice)

    response = client.get(f"/v3/workspaces/{workspace.name}/graph/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["conclusions"] == 6
    counts = {
        (p["observer"], p["observed"]): (p["conclusions"], p["derived"])
        for p in body["perspectives"]
    }
    assert counts == {
        (alice.name, "bob"): (4, 3),
        (alice.name, alice.name): (1, 0),
        (TEAM, "bob"): (1, 0),
    }
    assert body["perspectives"][0]["observed"] == "bob"
    assert "version" not in body


async def test_text_returns_live_conclusions_only(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    ids = await _seed(db_session, workspace, alice)

    response = client.post(
        f"/v3/workspaces/{workspace.name}/graph/text",
        json={"ids": [ids["root"], ids["deleted"], MISSING_ID]},
    )
    assert response.status_code == 200
    assert response.json()["items"] == [
        {"id": ids["root"], "content": "conclusion at minute 0", "session_id": "s1"}
    ]


def _call(client: TestClient, route: str, workspace: str, **params: str):
    base = f"/v3/workspaces/{workspace}/graph"
    if route == "graph":
        return client.get(base, params=params)
    if route == "overview":
        return client.get(f"{base}/overview")
    return client.post(f"{base}/text", json={"ids": [MISSING_ID]})


ROUTES = ["graph", "overview", "text"]


@pytest.mark.parametrize("route", ROUTES)
def test_a_missing_workspace_is_404(client: TestClient, route: str):
    assert _call(client, route, "no-such-workspace").status_code == 404


@pytest.mark.parametrize("position", ["observer", "observed"])
def test_a_missing_peer_is_404(
    client: TestClient, sample_data: tuple[Workspace, Peer], position: str
):
    workspace, _ = sample_data
    response = _call(client, "graph", workspace.name, **{position: "no-such-peer"})
    assert response.status_code == 404
    assert "no-such-peer" in response.json()["detail"]


@pytest.mark.parametrize("route", ROUTES)
async def test_a_workspace_key_reads_every_route(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    route: str,
):
    workspace, alice = sample_data
    await _seed(db_session, workspace, alice)
    _use_key(client, monkeypatch, JWTParams(w=workspace.name))

    assert _call(client, route, workspace.name).status_code == 200


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("scope", ["peer", "other workspace"])
def test_keys_below_the_workspace_are_refused(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    route: str,
):
    workspace, alice = sample_data
    params = (
        JWTParams(w=workspace.name, p=alice.name)
        if scope == "peer"
        else JWTParams(w="another-workspace")
    )
    _use_key(client, monkeypatch, params)

    assert _call(client, route, workspace.name).status_code == 401

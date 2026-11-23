"""The conclusion graph route answers with HONCHOGR v2 bytes for the caller's workspace."""

import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt
from src.utils.graph_wire import CONTENT_TYPE
from src.utils.honchogr_reference import NO_COPIED_FROM
from src.utils.scopes import scope_peer_name
from tests.utils.test_graph_wire import decode

T0 = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
T0_US = 1_704_067_200_000_000
TEAM = scope_peer_name("team")
MISSING_ID = "z" * 21
NOT_APPLICABLE, COMPLETE, PARTIAL, LINEAGE_UNKNOWN = range(4)


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


def _names(graph: dict[str, Any], ids: dict[str, str]) -> list[str]:
    by_id = {doc_id: name for name, doc_id in ids.items()}
    return [by_id[doc_id] for doc_id in graph["ids"]]


async def test_whole_workspace_graph(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    ids = await _seed(db_session, workspace, alice)

    response = client.get(f"/v3/workspaces/{workspace.name}/graph")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE
    graph = decode(response.content)

    order = _names(graph, ids)
    assert order == ["root", "other", "derived", "partial", "lonely", "copy"]
    at = {name: k for k, name in enumerate(order)}
    assert graph["created_us"][at["root"]] == T0_US
    assert graph["sessions"] == ["s1"]
    assert graph["node_session"][at["root"]] == 0
    expected = [
        (alice.name, alice.name, "peer"),
        (alice.name, "bob", "peer"),
        (TEAM, "bob", "scope"),
    ]
    assert graph["perspectives"] == sorted(
        expected, key=lambda p: (p[0].encode(), p[1].encode())
    )
    assert graph["times_derived"][at["derived"]] == 2

    start, end = graph["offsets"][at["derived"]], graph["offsets"][at["derived"] + 1]
    assert graph["targets"][start:end] == [at["root"], at["other"]]
    assert graph["lineage"][at["derived"]] == COMPLETE

    p = at["partial"]
    assert (graph["outside"][p], graph["deleted"][p], graph["unknown"][p]) == (0, 1, 1)
    assert graph["lineage"][p] == PARTIAL
    assert graph["lineage"][at["lonely"]] == LINEAGE_UNKNOWN

    assert graph["flags"][at["copy"]] == 1
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
    graph = decode(response.content)

    order = _names(graph, ids)
    assert "other" not in order
    d = order.index("derived")
    assert (graph["outside"][d], graph["deleted"][d], graph["unknown"][d]) == (1, 0, 0)
    assert graph["lineage"][d] == PARTIAL
    assert ids["other"].encode() not in response.content


async def test_a_copy_without_its_original_has_no_copied_from(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    ids = await _seed(db_session, workspace, alice)

    response = client.get(
        f"/v3/workspaces/{workspace.name}/graph", params={"observer": TEAM}
    )
    graph = decode(response.content)

    assert _names(graph, ids) == ["copy"]
    assert graph["flags"] == [1]
    assert graph["copied_from"] == [NO_COPIED_FROM]
    assert graph["perspectives"] == [(TEAM, "bob", "scope")]


async def test_matching_etag_returns_304(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, alice = sample_data
    await _seed(db_session, workspace, alice)
    url = f"/v3/workspaces/{workspace.name}/graph"

    first = client.get(url)
    second = client.get(url, headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 304
    assert second.content == b""


@pytest.mark.parametrize("scope", ["peer", "other workspace"])
def test_keys_below_the_workspace_are_refused(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
):
    workspace, alice = sample_data
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    params = (
        JWTParams(w=workspace.name, p=alice.name)
        if scope == "peer"
        else JWTParams(w="another-workspace")
    )
    client.headers["Authorization"] = f"Bearer {create_jwt(params)}"

    response = client.get(f"/v3/workspaces/{workspace.name}/graph")
    assert response.status_code == 401

"""Named-scope keys enforce a projection while permitting message ingestion."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.message import resolve_session_scope
from src.exceptions import AuthenticationException
from src.security import create_admin_jwt, verify_jwt
from src.utils.agent_tools import (
    ToolContext,
    _handle_get_reasoning_chain,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.scopes import scope_peer_name
from tests.routes.test_scope_reads import (
    _seed_documents,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def scope_key(
    client: TestClient,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, dict[str, str]]:
    workspace, peer = sample_data
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    client.headers["Authorization"] = f"Bearer {create_admin_jwt()}"
    root = f"/v3/workspaces/{workspace.name}"
    for scope in ("private", "shared"):
        assert client.post(f"{root}/scopes", json={"id": scope}).status_code == 201
    for session, scopes in (("member", ["private"]), ("outside", ["shared"])):
        response = client.post(
            f"{root}/sessions",
            json={"id": session, "scopes": scopes, "peers": {peer.name: {}}},
        )
        assert response.status_code == 201, response.text
    response = client.post(
        "/v3/keys", params={"workspace_id": workspace.name, "scope_id": "private"}
    )
    assert response.status_code == 200, response.text
    assert verify_jwt(response.json()["key"]).sc == "private"
    return root, {"Authorization": f"Bearer {response.json()['key']}"}


def test_create_ingest_read_and_update(
    client: TestClient,
    scope_key: tuple[str, dict[str, str]],
    sample_data: tuple[models.Workspace, models.Peer],
):
    root, headers = scope_key
    _, peer = sample_data
    response = client.post(f"{root}/sessions", json={"id": "new"}, headers=headers)
    assert response.status_code == 201, response.text
    members = client.post(
        f"{root}/scopes/private/sessions/list", headers=headers
    ).json()["items"]
    assert {s["id"] for s in members} == {"member", "new"}
    response = client.post(
        f"{root}/sessions/new/messages",
        json={"messages": [{"peer_id": peer.name, "content": "scope-key ingestion"}]},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    message_id = response.json()[0]["id"]
    path = f"{root}/sessions/new/messages/{message_id}"
    assert client.get(path, headers=headers).json()["content"] == "scope-key ingestion"
    assert (
        client.put(
            path, json={"metadata": {"source": "centaur"}}, headers=headers
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"{root}/sessions/new/messages/list", json={}, headers=headers
        ).status_code
        == 200
    )
    assert (
        client.get(f"{root}/sessions/new/context", headers=headers).status_code == 200
    )
    assert (
        client.get(f"{root}/sessions/new/summaries", headers=headers).status_code == 200
    )
    assert client.get(f"{root}/scopes/private", headers=headers).status_code == 200
    assert (
        client.get(f"{root}/scopes/private/status", headers=headers).status_code == 200
    )
    # Reopening a member session must not re-enroll it or enqueue a backfill.
    assert (
        client.post(
            f"{root}/sessions",
            json={"id": "new", "scopes": ["private"]},
            headers=headers,
        ).status_code
        == 200
    )


@pytest.mark.parametrize(
    "body",
    [
        {"id": "outside"},
        {"id": "outside", "scopes": ["private"]},
        {"id": "new", "scopes": ["shared"]},
        {"id": "new", "scopes": ["private", "shared"]},
        {"id": "new", "scopes": []},
    ],
)
def test_cannot_enroll_unrelated_sessions(
    client: TestClient, scope_key: tuple[str, dict[str, str]], body: dict[str, Any]
):
    root, headers = scope_key
    assert (
        client.post(f"{root}/sessions", json=body, headers=headers).status_code == 401
    )
    members = client.post(
        f"{root}/scopes/private/sessions/list", headers=headers
    ).json()["items"]
    assert [s["id"] for s in members] == ["member"]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        (
            "POST",
            "/sessions/outside/messages",
            {"messages": [{"peer_id": "alice", "content": "forbidden"}]},
        ),
        ("POST", "/sessions/outside/messages/list", {}),
        ("GET", "/sessions/outside/context", None),
        ("POST", "/scopes/private/sessions", {"session_ids": ["outside"]}),
        ("DELETE", "/scopes/private/sessions/member", None),
        ("GET", "/scopes/shared", None),
        ("POST", "/scopes/shared/sessions/list", {}),
        ("GET", "/scopes/shared/status", None),
        ("POST", "/scopes/list", {}),
        ("POST", "/scopes", {"id": "new-scope"}),
        ("POST", "/peers", {"id": "alice"}),
        ("POST", "/peers/list", {}),
        ("GET", "/peers/alice/context", None),
        ("GET", "/peers/alice/card", None),
        ("POST", "/sessions/list", {}),
        ("POST", "/chat", {"query": "all company secrets"}),
        ("POST", "/search", {"query": "secret"}),
        ("POST", "/conclusions/list", {}),
        ("GET", "/sessions/member/peers", None),
        ("DELETE", "/sessions/member", None),
    ],
)
def test_unsupported_or_outside_routes_fail_closed(
    client: TestClient,
    scope_key: tuple[str, dict[str, str]],
    method: str,
    path: str,
    body: dict[str, Any] | None,
):
    root, headers = scope_key
    assert (
        client.request(method, root + path, json=body, headers=headers).status_code
        == 401
    )


@pytest.mark.parametrize(
    "options",
    [
        {"scope": "shared"},
        {"scope": ["private"]},
        {"scope": ["private", "shared"]},
        {"session_id": "outside"},
        {"filters": {"session_id": "outside"}},
    ],
)
@pytest.mark.parametrize("endpoint", ["chat", "representation"])
def test_recall_cannot_override_scope(
    client: TestClient,
    scope_key: tuple[str, dict[str, str]],
    options: dict[str, Any],
    endpoint: str,
):
    root, headers = scope_key
    body = {"query": "What do you know?", **options} if endpoint == "chat" else options
    assert client.post(
        f"{root}/peers/alice/{endpoint}", json=body, headers=headers
    ).status_code in (401, 422)


@pytest.mark.parametrize(
    "params",
    [
        {"scope": "shared"},
        {"peer_perspective": "alice", "peer_target": "alice"},
        {"sessions": ["outside"], "peer_target": "alice"},
        {"limit_to_session": "true", "peer_target": "alice"},
    ],
)
def test_context_cannot_override_scope(
    client: TestClient, scope_key: tuple[str, dict[str, str]], params: dict[str, Any]
):
    root, headers = scope_key
    assert (
        client.get(
            f"{root}/sessions/member/context", params=params, headers=headers
        ).status_code
        == 401
    )


async def test_implicit_recall_excludes_global_representation_and_card(
    client: TestClient,
    scope_key: tuple[str, dict[str, str]],
    sample_data: tuple[models.Workspace, models.Peer],
    db_session: AsyncSession,
):
    root, headers = scope_key
    workspace, peer = sample_data
    for observer, fact in (
        (peer.name, "GLOBAL SECRET"),
        (scope_peer_name("private"), "PRIVATE FACT"),
    ):
        await _seed_documents(
            db_session,
            workspace.name,
            observer=observer,
            observed=peer.name,
            contents=[(fact, "member")],
        )
        await crud.set_peer_card(
            db_session,
            workspace.name,
            peer_card=[fact + " CARD"],
            observer=observer,
            observed=peer.name,
        )
    await db_session.commit()
    response = client.post(
        f"{root}/peers/{peer.name}/representation", json={}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert "PRIVATE FACT" in response.text and "GLOBAL SECRET" not in response.text
    response = client.get(
        f"{root}/sessions/member/context",
        params={"peer_target": peer.name},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["peer_card"] == ["PRIVATE FACT CARD"]
    assert "GLOBAL SECRET" not in response.text
    # Existing workspace keys retain the global view when scope is omitted.
    response = client.get(
        f"{root}/sessions/member/context", params={"peer_target": peer.name}
    )
    assert response.json()["peer_card"] == ["GLOBAL SECRET CARD"]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_binds_scope_before_dialectic(
    client: TestClient,
    scope_key: tuple[str, dict[str, str]],
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
):
    root, headers = scope_key
    _, peer = sample_data
    captured: dict[str, Any] = {}

    async def chat(**kwargs: Any):
        captured.update(kwargs)
        return "scoped answer"

    async def streaming(**kwargs: Any):
        captured.update(kwargs)
        yield "scoped answer"

    monkeypatch.setattr("src.routers.peers.agentic_chat", chat)
    monkeypatch.setattr("src.routers.peers.agentic_chat_stream", streaming)
    response = client.post(
        f"{root}/peers/{peer.name}/chat",
        json={"query": "Who am I?", "stream": stream},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert captured["observer"] == scope_peer_name("private")
    assert captured["observed"] == peer.name


def test_removal_revokes_session_access(
    client: TestClient, scope_key: tuple[str, dict[str, str]]
):
    root, headers = scope_key
    assert client.delete(f"{root}/scopes/private/sessions/member").status_code == 204
    for path in ("context", "summaries"):
        assert (
            client.get(f"{root}/sessions/member/{path}", headers=headers).status_code
            == 401
        )
    assert (
        client.post(
            f"{root}/sessions/member/messages",
            json={"messages": [{"peer_id": "alice", "content": "no"}]},
            headers=headers,
        ).status_code
        == 401
    )
    assert (
        client.post(
            f"{root}/sessions", json={"id": "member"}, headers=headers
        ).status_code
        == 401
    )


def test_workspace_and_key_delegation_denied(
    client: TestClient, scope_key: tuple[str, dict[str, str]]
):
    _, headers = scope_key
    assert (
        client.post(
            "/v3/keys", params={"workspace_id": "elsewhere"}, headers=headers
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v3/workspaces", json={"id": "elsewhere"}, headers=headers
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v3/workspaces/elsewhere/sessions", json={"id": "new"}, headers=headers
        ).status_code
        == 401
    )


@pytest.mark.parametrize(
    "params",
    [
        {"scope_id": "private"},
        {"workspace_id": "ws", "scope_id": ""},
        {"workspace_id": "ws", "scope_id": "scope.private"},
        {"workspace_id": "ws", "scope_id": "private", "peer_id": "alice"},
        {"workspace_id": "ws", "scope_id": "private", "session_id": "member"},
    ],
)
@pytest.mark.usefixtures("scope_key")
def test_invalid_key_shapes_rejected(client: TestClient, params: dict[str, str]):
    assert client.post("/v3/keys", params=params).status_code == 422


@pytest.mark.usefixtures("scope_key")
async def test_create_race_cannot_enroll_competing_session(
    sample_data: tuple[models.Workspace, models.Peer],
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
):
    """A unique conflict retries authorization against the winning session."""
    from src.crud import session as session_crud

    workspace, _ = sample_data
    original = session_crud._fetch_session  # pyright: ignore[reportPrivateUsage]
    fetch = AsyncMock(
        side_effect=[None, await original(db_session, workspace.name, "outside")]
    )
    monkeypatch.setattr(session_crud, "_fetch_session", fetch)
    with pytest.raises(AuthenticationException, match="JWT not permissioned"):
        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(name="outside", scopes=["private"]),
            workspace.name,
            acting_scope="private",
        )
    assert fetch.call_count == 2


async def test_message_recall_excludes_removed_and_unrelated_sessions(
    client: TestClient,
    scope_key: tuple[str, dict[str, str]],
    sample_data: tuple[models.Workspace, models.Peer],
    db_session: AsyncSession,
):
    root, _ = scope_key
    workspace, peer = sample_data
    for session in ("member", "outside"):
        assert (
            client.post(
                f"{root}/sessions/{session}/messages",
                json={
                    "messages": [
                        {"peer_id": peer.name, "content": session + " evidence"}
                    ]
                },
            ).status_code
            == 201
        )
    observer = scope_peer_name("private")
    messages = await crud.get_messages_by_date_range(
        db_session, workspace.name, None, observer=observer
    )
    assert [m.content for m in messages] == ["member evidence"]
    assert client.delete(f"{root}/scopes/private/sessions/member").status_code == 204
    for session in (None, "member", "outside"):
        # SQL (grep/date/context) and external-vector-store paths must agree.
        messages = await crud.get_messages_by_date_range(
            db_session, workspace.name, session, observer=observer
        )
        assert messages == []
        _, deny = await resolve_session_scope(
            db_session, workspace.name, session, None, observer
        )
        assert deny
    # Existing ordinary peer recall keeps its historical membership behavior.
    assert (
        len(
            await crud.get_messages_by_date_range(
                db_session, workspace.name, None, observer=peer.name
            )
        )
        == 2
    )


@pytest.mark.usefixtures("scope_key")
async def test_reasoning_chain_cannot_follow_foreign_ids(
    sample_data: tuple[models.Workspace, models.Peer],
    db_session: AsyncSession,
):
    workspace, peer = sample_data
    observer = scope_peer_name("private")
    for source, fact in ((peer.name, "GLOBAL SECRET"), (observer, "SCOPED PREMISE")):
        await _seed_documents(
            db_session,
            workspace.name,
            observer=source,
            observed=peer.name,
            contents=[(fact, "member")],
        )
    docs = list((await db_session.scalars(select(models.Document))).all())
    foreign = next(doc for doc in docs if doc.observer == peer.name)
    scoped = [doc for doc in docs if doc.observer == observer]
    derived = models.Document(
        workspace_name=workspace.name,
        observer=observer,
        observed=peer.name,
        content="SCOPED DERIVED",
        level="deductive",
        source_ids=[foreign.id, scoped[0].id],
    )
    db_session.add(derived)
    await db_session.commit()
    ctx = ToolContext(
        workspace_name=workspace.name,
        observer=observer,
        observed=peer.name,
        session_name=None,
        current_messages=None,
        include_observation_ids=True,
        db_lock=asyncio.Lock(),
    )
    guessed = await _handle_get_reasoning_chain(ctx, {"observation_id": foreign.id})
    assert "not found" in guessed and "GLOBAL SECRET" not in guessed
    chain = await _handle_get_reasoning_chain(ctx, {"observation_id": derived.id})
    assert "SCOPED DERIVED" in chain and "SCOPED PREMISE" in chain
    assert "GLOBAL SECRET" not in chain

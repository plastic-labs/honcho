"""Tests for the `scope` option on the read routes (DEV-1998).

A single scope swaps the observer to the scope peer, so recall is confined to
the (scope, observed) collection and the scope's member sessions by existing
observer semantics. A list of scopes keeps the path peer as observer and
restricts recall to the union of the scopes' member sessions (the DEV-1995
session-allowlist arm).
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt
from src.utils.scopes import scope_peer_name


def _create_scope(client: TestClient, workspace_name: str, scope_name: str):
    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes", json={"id": scope_name}
    )
    assert response.status_code in [200, 201]
    return response


def _create_session(
    client: TestClient,
    workspace_name: str,
    session_name: str | None = None,
    **extra: Any,
) -> str:
    session_name = session_name or str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions",
        json={"id": session_name, **extra},
    )
    assert response.status_code in [200, 201]
    return session_name


def _add_sessions_to_scope(
    client: TestClient, workspace_name: str, scope_name: str, session_names: list[str]
) -> None:
    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/sessions",
        json={"session_ids": session_names},
    )
    assert response.status_code == 200


async def _seed_documents(
    db_session: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    contents: list[tuple[str, str | None]],
) -> None:
    """Seed a collection plus documents for an (observer, observed) pair.

    ``contents`` is a list of (content, session_name) tuples.
    """
    collection = models.Collection(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )
    db_session.add(collection)
    await db_session.flush()
    db_session.add_all(
        [
            models.Document(
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=content,
                session_name=session_name,
            )
            for content, session_name in contents
        ]
    )
    await db_session.commit()


async def _seed_legacy_collision_peer(
    db_session: AsyncSession, workspace_name: str, scope_name: str
) -> None:
    """Create a plain peer squatting on a scope's reserved internal name."""
    db_session.add(
        models.Peer(
            workspace_name=workspace_name,
            name=scope_peer_name(scope_name),
        )
    )
    await db_session.commit()


class TestScopeReadValidation:
    """4xx paths shared by chat and representation.

    Chat validation happens before any LLM work, so these are safe to exercise.
    """

    def _chat(
        self, client: TestClient, workspace: Workspace, peer: Peer, body: dict[str, Any]
    ):
        return client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
            json={"query": "what do you know?", **body},
        )

    def _representation(
        self, client: TestClient, workspace: Workspace, peer: Peer, body: dict[str, Any]
    ):
        return client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/representation",
            json=body,
        )

    def test_unknown_scope_404(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        unknown = str(generate_nanoid())
        assert (
            self._chat(client, workspace, peer, {"scope": unknown}).status_code == 404
        )
        assert (
            self._representation(
                client, workspace, peer, {"scope": unknown}
            ).status_code
            == 404
        )

    def test_unknown_scope_in_list_404(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        resp = self._representation(
            client, workspace, peer, {"scope": [scope_name, str(generate_nanoid())]}
        )
        assert resp.status_code == 404

    async def test_non_scope_peer_as_scope_422(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """A peer squatting on the reserved name without the kind flag is not a scope."""
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        await _seed_legacy_collision_peer(db_session, workspace.name, scope_name)

        assert (
            self._chat(client, workspace, peer, {"scope": scope_name}).status_code
            == 422
        )
        assert (
            self._representation(
                client, workspace, peer, {"scope": scope_name}
            ).status_code
            == 422
        )

    def test_scope_plus_filters_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        body = {"scope": scope_name, "filters": {"session_id": ["s1"]}}
        assert self._chat(client, workspace, peer, body).status_code == 422
        assert self._representation(client, workspace, peer, body).status_code == 422

    def test_scope_plus_session_id_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        body = {"scope": scope_name, "session_id": "s1"}
        assert self._chat(client, workspace, peer, body).status_code == 422
        assert self._representation(client, workspace, peer, body).status_code == 422

    def test_peer_scoped_jwt_403(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A scope's sessions may exceed the peer's own membership: workspace/admin only."""
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)

        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=workspace.name, p=peer.name))}"
        )

        assert (
            self._chat(client, workspace, peer, {"scope": scope_name}).status_code
            == 403
        )
        assert (
            self._representation(
                client, workspace, peer, {"scope": scope_name}
            ).status_code
            == 403
        )

        # A workspace-level key is allowed through validation (404 here only
        # if the scope were unknown; representation of an empty scope is 200).
        client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=workspace.name))}"
        )
        assert (
            self._representation(
                client, workspace, peer, {"scope": scope_name}
            ).status_code
            == 200
        )

    def test_scope_union_cap_422(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_a = _create_session(client, workspace.name)
        session_b = _create_session(client, workspace.name)
        _add_sessions_to_scope(
            client, workspace.name, scope_name, [session_a, session_b]
        )

        monkeypatch.setattr("src.routers.peers.MAX_SESSION_ALLOWLIST_ENTRIES", 1)
        resp = self._representation(client, workspace, peer, {"scope": [scope_name]})
        assert resp.status_code == 422
        assert "maximum" in resp.json()["detail"]


class TestRepresentationWithScope:
    async def test_single_scope_reads_scope_collection(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """A single scope swaps the observer: only the (scope, peer) collection is read."""
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_a = _create_session(client, workspace.name)
        session_b = _create_session(client, workspace.name)
        _add_sessions_to_scope(client, workspace.name, scope_name, [session_a])

        # Conclusions the scope observed (session A) ...
        await _seed_documents(
            db_session,
            workspace.name,
            observer=scope_peer_name(scope_name),
            observed=peer.name,
            contents=[("scoped fact about hiking", session_a)],
        )
        # ... and global self-observations from another session
        await _seed_documents(
            db_session,
            workspace.name,
            observer=peer.name,
            observed=peer.name,
            contents=[("global fact about cooking", session_b)],
        )

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/representation",
            json={"scope": scope_name},
        )
        assert resp.status_code == 200
        representation = resp.json()["representation"]
        assert "scoped fact about hiking" in representation
        assert "global fact about cooking" not in representation

    async def test_scope_list_unions_member_sessions(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """A scope list keeps the global observer and applies the union allowlist."""
        workspace, peer = sample_data
        scope_a = str(generate_nanoid())
        scope_b = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_a)
        _create_scope(client, workspace.name, scope_b)
        session_a = _create_session(client, workspace.name)
        session_b = _create_session(client, workspace.name)
        session_c = _create_session(client, workspace.name)
        _add_sessions_to_scope(client, workspace.name, scope_a, [session_a])
        _add_sessions_to_scope(client, workspace.name, scope_b, [session_b])

        # All conclusions live in the GLOBAL (peer, peer) collection: only the
        # union session-allowlist can explain the filtering below (this is the
        # dynamic DEV-1995 arm, not the observer swap).
        await _seed_documents(
            db_session,
            workspace.name,
            observer=peer.name,
            observed=peer.name,
            contents=[
                ("fact from session a", session_a),
                ("fact from session b", session_b),
                ("fact from session c", session_c),
                ("sessionless dream fact", None),
            ],
        )

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/representation",
            json={"scope": [scope_a, scope_b]},
        )
        assert resp.status_code == 200
        representation = resp.json()["representation"]
        assert "fact from session a" in representation
        assert "fact from session b" in representation
        assert "fact from session c" not in representation
        assert "sessionless dream fact" not in representation

    def test_empty_scope_list_fails_closed(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        """A scope with no member sessions yields an empty representation."""
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/representation",
            json={"scope": [scope_name]},
        )
        assert resp.status_code == 200
        assert "fact" not in resp.json()["representation"]


class TestChatWithScope:
    """Verify what the chat route hands the dialectic, without real LLM work.

    ``agentic_chat`` is mocked in conftest (``mock_llm_call_functions``); the
    scoped peer-card fetch happens inside it and is covered end-to-end by the
    session-context test. Here we assert the route passes the right observer /
    observed / session_names — the wiring that keys the card fetch.
    """

    def test_single_scope_swaps_observer(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        mock_llm_call_functions: dict[str, Any],
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
            json={"query": "what do you know?", "scope": scope_name},
        )
        assert resp.status_code == 200

        kwargs = mock_llm_call_functions["agentic_chat"].await_args.kwargs
        # The scope peer is the observer; the path peer stays the observed
        assert kwargs["observer"] == scope_peer_name(scope_name)
        assert kwargs["observed"] == peer.name
        # Single-scope confinement rides on observer semantics, not an allowlist
        assert kwargs["session_names"] is None

    def test_scope_list_passes_union_allowlist(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        mock_llm_call_functions: dict[str, Any],
    ):
        workspace, peer = sample_data
        scope_a = str(generate_nanoid())
        scope_b = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_a)
        _create_scope(client, workspace.name, scope_b)
        session_a = _create_session(client, workspace.name)
        session_b = _create_session(client, workspace.name)
        _add_sessions_to_scope(client, workspace.name, scope_a, [session_a])
        _add_sessions_to_scope(client, workspace.name, scope_b, [session_b])

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
            json={"query": "what do you know?", "scope": [scope_a, scope_b]},
        )
        assert resp.status_code == 200

        kwargs = mock_llm_call_functions["agentic_chat"].await_args.kwargs
        # Union path: the path peer stays the observer, the allowlist is the union
        assert kwargs["observer"] == peer.name
        assert kwargs["observed"] == peer.name
        assert set(kwargs["session_names"]) == {session_a, session_b}


class TestWorkspaceSearchWithScope:
    def _seed_message(
        self, client: TestClient, workspace_name: str, session_name: str, peer: Peer
    ) -> None:
        resp = client.post(
            f"/v3/workspaces/{workspace_name}/sessions/{session_name}/messages",
            json={
                "messages": [{"peer_id": peer.name, "content": "needle in haystack"}]
            },
        )
        assert resp.status_code == 201

    def test_search_restricted_to_scope_sessions(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_a = _create_session(client, workspace.name, peers={peer.name: {}})
        session_b = _create_session(client, workspace.name, peers={peer.name: {}})
        _add_sessions_to_scope(client, workspace.name, scope_name, [session_a])
        self._seed_message(client, workspace.name, session_a, peer)
        self._seed_message(client, workspace.name, session_b, peer)

        # Unscoped: both sessions' messages match
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/search",
            json={"query": "needle"},
        )
        assert resp.status_code == 200
        assert {m["session_id"] for m in resp.json()} == {session_a, session_b}

        # Scoped: only the scope's member session
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/search",
            json={"query": "needle", "scope": scope_name},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert results
        assert {m["session_id"] for m in results} == {session_a}

    def test_empty_scope_returns_no_results(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_a = _create_session(client, workspace.name, peers={peer.name: {}})
        self._seed_message(client, workspace.name, session_a, peer)

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/search",
            json={"query": "needle", "scope": scope_name},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_unknown_scope_404(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, _ = sample_data
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/search",
            json={"query": "needle", "scope": str(generate_nanoid())},
        )
        assert resp.status_code == 404

    def test_scope_plus_session_id_filter_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, _ = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/search",
            json={
                "query": "needle",
                "scope": scope_name,
                "filters": {"session_id": "s1"},
            },
        )
        assert resp.status_code == 422


class TestSessionContextWithScope:
    async def test_scope_swaps_perspective_source(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """`scope` reads the scope's collection and the scoped peer card."""
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})
        _add_sessions_to_scope(client, workspace.name, scope_name, [session_name])

        await _seed_documents(
            db_session,
            workspace.name,
            observer=scope_peer_name(scope_name),
            observed=peer.name,
            contents=[("scoped fact about hiking", session_name)],
        )
        await _seed_documents(
            db_session,
            workspace.name,
            observer=peer.name,
            observed=peer.name,
            contents=[("global fact about cooking", session_name)],
        )
        await crud.set_peer_card(
            db_session,
            workspace.name,
            peer_card=["SCOPED CARD"],
            observer=scope_peer_name(scope_name),
            observed=peer.name,
        )
        await crud.set_peer_card(
            db_session,
            workspace.name,
            peer_card=["GLOBAL CARD"],
            observer=peer.name,
            observed=peer.name,
        )
        await db_session.commit()

        # Without scope: the global (self) perspective
        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={"peer_target": peer.name},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "global fact about cooking" in data["peer_representation"]
        assert data["peer_card"] == ["GLOBAL CARD"]

        # With scope: the scope's perspective
        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={"peer_target": peer.name, "scope": scope_name},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "scoped fact about hiking" in data["peer_representation"]
        assert "global fact about cooking" not in data["peer_representation"]
        assert data["peer_card"] == ["SCOPED CARD"]

    def test_scope_requires_peer_target(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})

        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={"scope": scope_name},
        )
        assert resp.status_code == 422

    def test_scope_and_peer_perspective_mutually_exclusive(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})

        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={
                "peer_target": peer.name,
                "peer_perspective": peer.name,
                "scope": scope_name,
            },
        )
        assert resp.status_code == 422

    def test_unknown_scope_404(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})

        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={"peer_target": peer.name, "scope": str(generate_nanoid())},
        )
        assert resp.status_code == 404

    def test_narrow_keys_rejected_403(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Peer- and session-scoped keys may not widen reads through a scope."""
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})
        _add_sessions_to_scope(client, workspace.name, scope_name, [session_name])

        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        url = f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context"
        params = {"peer_target": peer.name, "scope": scope_name}

        # Peer-scoped key (member read grants access to the route, not to scope)
        client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=workspace.name, p=peer.name))}"
        )
        assert client.get(url, params=params).status_code == 403

        # Session-scoped key
        client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=workspace.name, s=session_name))}"
        )
        assert client.get(url, params=params).status_code == 403

        # Workspace-scoped key is allowed
        client.headers["Authorization"] = (
            f"Bearer {create_jwt(JWTParams(w=workspace.name))}"
        )
        assert client.get(url, params=params).status_code == 200


class TestScopePeerGuardrailClosure:
    """Scope peers are rejected on the generic perspective/context surfaces."""

    def test_session_context_rejects_scope_peer_target(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})

        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={"peer_target": scope_peer_name(scope_name)},
        )
        assert resp.status_code == 422

    def test_session_context_rejects_scope_peer_perspective(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)
        session_name = _create_session(client, workspace.name, peers={peer.name: {}})

        resp = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{session_name}/context",
            params={
                "peer_target": peer.name,
                "peer_perspective": scope_peer_name(scope_name),
            },
        )
        assert resp.status_code == 422

    def test_peer_context_rejects_scope_peer(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        scope_name = str(generate_nanoid())
        _create_scope(client, workspace.name, scope_name)

        # As the path-level peer
        resp = client.get(
            f"/v3/workspaces/{workspace.name}/peers/{scope_peer_name(scope_name)}/context"
        )
        assert resp.status_code == 422

        # As the target
        resp = client.get(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/context",
            params={"target": scope_peer_name(scope_name)},
        )
        assert resp.status_code == 422

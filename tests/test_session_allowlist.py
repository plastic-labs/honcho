"""
Tests for the session allowlist (DEV-1995).

Covers the constrained `filters` surface on dialectic/representation
(extract_session_allowlist), fail-closed conclusion recall (search_memory),
and the strict allowlist ∩ membership intersection in message cruds.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.crud.message import resolve_session_scope
from src.exceptions import FilterError
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt
from src.utils.agent_tools import search_memory
from src.utils.filter import (
    MAX_SESSION_ALLOWLIST_ENTRIES,
    extract_session_allowlist,
)


class TestExtractSessionAllowlist:
    def test_none_passthrough(self):
        assert extract_session_allowlist(None) is None

    def test_single_id(self):
        assert extract_session_allowlist({"session_id": "s1"}) == ["s1"]

    def test_bare_list(self):
        assert extract_session_allowlist({"session_id": ["s1", "s2"]}) == ["s1", "s2"]

    def test_in_operator(self):
        assert extract_session_allowlist({"session_id": {"in": ["s1"]}}) == ["s1"]

    def test_dedupes_preserving_order(self):
        assert extract_session_allowlist({"session_id": ["s2", "s1", "s2"]}) == [
            "s2",
            "s1",
        ]

    def test_empty_list_preserved_for_fail_closed(self):
        assert extract_session_allowlist({"session_id": []}) == []

    def test_unsupported_key_rejected(self):
        with pytest.raises(FilterError, match="Unsupported filter key"):
            extract_session_allowlist({"peer_id": ["a"], "session_id": ["s1"]})

    def test_missing_session_id_rejected(self):
        with pytest.raises(FilterError, match="must contain"):
            extract_session_allowlist({})

    def test_bad_shapes_rejected(self):
        for bad in [123, {"gte": "x"}, {"in": "s1"}, [1, 2], [""], None]:
            with pytest.raises(FilterError):
                extract_session_allowlist({"session_id": bad})

    @pytest.mark.parametrize(
        "filters",
        [
            {"session_id": "*"},
            {"session_id": ["s1", "*"]},
            {"session_id": {"in": ["*"]}},
        ],
    )
    def test_wildcard_rejected(self, filters: dict[str, Any]):
        """A wildcard means two different things depending on which consumer
        receives the allowlist: the filter DSL drops the condition entirely
        (matching every session), while the direct `IN` and Python membership
        paths treat "*" as a literal session name (matching none). It is not
        part of this endpoint's contract, so it is rejected outright.

        The mixed list is the case that matters most — it looks narrowed.
        """
        with pytest.raises(FilterError, match="Invalid session id"):
            extract_session_allowlist(filters)

    @pytest.mark.parametrize("name", ["a b", "a/b", "a.b", "a%b", "s1;drop"])
    def test_malformed_session_ids_rejected(self, name: str):
        with pytest.raises(FilterError, match="Invalid session id"):
            extract_session_allowlist({"session_id": name})

    def test_valid_id_characters_still_accepted(self):
        """The pattern must not be stricter than the ids the API actually
        issues, which include underscores and hyphens."""
        assert extract_session_allowlist({"session_id": "Valid_name-123"}) == [
            "Valid_name-123"
        ]

    def test_cap_enforced(self):
        too_many = [f"s{i}" for i in range(MAX_SESSION_ALLOWLIST_ENTRIES + 1)]
        with pytest.raises(FilterError, match="at most"):
            extract_session_allowlist({"session_id": too_many})

    def test_must_include_satisfied(self):
        assert extract_session_allowlist(
            {"session_id": ["s1", "s2"]}, must_include="s2"
        ) == ["s1", "s2"]

    def test_must_include_missing_rejected(self):
        with pytest.raises(FilterError, match="must be included"):
            extract_session_allowlist({"session_id": ["s1"]}, must_include="s2")

    def test_must_include_ignored_without_filters(self):
        assert extract_session_allowlist(None, must_include="s1") is None

    def test_must_include_none_is_no_constraint(self):
        assert extract_session_allowlist({"session_id": ["s1"]}, must_include=None) == [
            "s1"
        ]


class TestSearchMemoryAllowlist:
    @pytest.mark.asyncio
    async def test_allowlist_pushed_down_as_filters(self):
        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            await search_memory(
                workspace_name="w",
                observer="o",
                observed="o",
                query="q",
                limit=5,
                levels=["explicit"],
                embedding=[0.1],
                session_allowlist=["s1", "s2"],
            )
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] == {
            "level": {"in": ["explicit"]},
            "session_name": {"in": ["s1", "s2"]},
        }

    @pytest.mark.asyncio
    async def test_allowlist_narrows_levels_to_allowlist_safe(self):
        """Only levels with a trustworthy session stamp survive scoping.

        Dream-derived levels are stamped with one session but synthesized
        across many (DEV-2201), so they can't be served under an allowlist.
        """
        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            await search_memory(
                workspace_name="w",
                observer="o",
                observed="o",
                query="q",
                limit=5,
                levels=["explicit", "inductive"],
                embedding=[0.1],
                session_allowlist=["s1"],
            )
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"]["level"] == {"in": ["explicit"]}

    @pytest.mark.asyncio
    async def test_allowlist_defaults_to_explicit_when_no_levels_requested(self):
        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            await search_memory(
                workspace_name="w",
                observer="o",
                observed="o",
                query="q",
                limit=5,
                embedding=[0.1],
                session_allowlist=["s1"],
            )
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"]["level"] == {"in": ["explicit"]}

    @pytest.mark.asyncio
    async def test_derived_only_request_under_allowlist_returns_empty(self):
        """The dialectic's derived prefetch short-circuits instead of querying."""
        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            result = await search_memory(
                workspace_name="w",
                observer="o",
                observed="o",
                query="q",
                limit=5,
                levels=["deductive", "inductive", "contradiction"],
                embedding=[0.1],
                session_allowlist=["s1"],
            )
        mock_query.assert_not_awaited()
        assert result.is_empty()

    @pytest.mark.asyncio
    async def test_levels_untouched_without_allowlist(self):
        """No allowlist means no level narrowing — unscoped recall is unchanged."""
        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            await search_memory(
                workspace_name="w",
                observer="o",
                observed="o",
                query="q",
                limit=5,
                levels=["deductive", "inductive"],
                embedding=[0.1],
            )
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] == {
            "level": {"in": ["deductive", "inductive"]}
        }

    @pytest.mark.asyncio
    async def test_empty_allowlist_fails_closed_without_querying(self):
        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            result = await search_memory(
                workspace_name="w",
                observer="o",
                observed="o",
                query="q",
                limit=5,
                embedding=[0.1],
                session_allowlist=[],
            )
        mock_query.assert_not_awaited()
        assert result.is_empty()


class TestMessageCrudAllowlistIntersection:
    """allowlist ∩ observer-membership, fail-closed on empty intersection."""

    async def _setup_two_sessions(
        self,
        client: TestClient,
        workspace: Workspace,
        peer: Peer,
    ) -> tuple[str, str]:
        ids: list[str] = []
        for marker in ("alpha", "beta"):
            session_id = str(generate_nanoid())
            resp = client.post(
                f"/v3/workspaces/{workspace.name}/sessions",
                json={"id": session_id, "peer_names": {peer.name: {}}},
            )
            assert resp.status_code == 201
            resp = client.post(
                f"/v3/workspaces/{workspace.name}/sessions/{session_id}/messages",
                json={
                    "messages": [
                        {
                            "content": f"needle in {marker}",
                            "peer_id": peer.name,
                        }
                    ]
                },
            )
            assert resp.status_code == 201
            ids.append(session_id)
        return ids[0], ids[1]

    @pytest.mark.asyncio
    async def test_grep_messages_intersects_allowlist(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, peer = sample_data
        session_a, session_b = await self._setup_two_sessions(client, workspace, peer)

        snippets = await crud.grep_messages(
            workspace_name=workspace.name,
            session_name=None,
            text="needle",
            observer=peer.name,
            session_allowlist=[session_a],
        )
        contents = [m.content for matches, _ in snippets for m in matches]
        assert contents == ["needle in alpha"]

        # A session the observer is NOT a member of contributes nothing,
        # even when allowlisted (strict intersection).
        foreign = str(generate_nanoid())
        snippets = await crud.grep_messages(
            workspace_name=workspace.name,
            session_name=None,
            text="needle",
            observer=peer.name,
            session_allowlist=[foreign],
        )
        assert snippets == []

        # Both sessions allowlisted -> both found
        snippets = await crud.grep_messages(
            workspace_name=workspace.name,
            session_name=None,
            text="needle",
            observer=peer.name,
            session_allowlist=[session_a, session_b],
        )
        assert len(snippets) == 2

    @pytest.mark.asyncio
    async def test_get_messages_by_date_range_intersects_allowlist(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, peer = sample_data
        session_a, _session_b = await self._setup_two_sessions(client, workspace, peer)

        messages = await crud.get_messages_by_date_range(
            db_session,
            workspace_name=workspace.name,
            session_name=None,
            observer=peer.name,
            session_allowlist=[session_a],
        )
        assert [m.content for m in messages] == ["needle in alpha"]

        # Empty allowlist fails closed
        messages = await crud.get_messages_by_date_range(
            db_session,
            workspace_name=workspace.name,
            session_name=None,
            observer=peer.name,
            session_allowlist=[],
        )
        assert messages == []


class TestPeerScopedJWTAllowlistGate:
    """A peer-scoped key may only allowlist sessions its peer actively belongs to.

    The gate uses `active_only=True` so it agrees with the `is_peer_in_session`
    check on `options.session_id` — a peer that has left a session is denied by
    both, not just one.
    """

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")

    def _chat_as(
        self,
        client: TestClient,
        workspace: Workspace,
        peer: Peer,
        token: str,
        body: dict[str, Any],
    ):
        return client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
            json={"query": "what do you know?", **body},
            headers={"Authorization": f"Bearer {token}"},
        )

    async def _session_with(
        self, client: TestClient, workspace: Workspace, peer: Peer
    ) -> str:
        session_id = str(generate_nanoid())
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/sessions",
            json={"id": session_id, "peer_names": {peer.name: {}}},
        )
        assert resp.status_code == 201
        return session_id

    @pytest.mark.asyncio
    async def test_member_sessions_allowed(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        session_id = await self._session_with(client, workspace, peer)
        token = create_jwt(JWTParams(w=workspace.name, p=peer.name))

        with patch(
            "src.routers.peers.agentic_chat", new=AsyncMock(return_value="ok")
        ) as mock_chat:
            resp = self._chat_as(
                client,
                workspace,
                peer,
                token,
                {"filters": {"session_id": [session_id]}},
            )
        assert resp.status_code == 200
        # The allowlist reaches the agent rather than being dropped at the gate.
        assert mock_chat.await_args is not None
        assert mock_chat.await_args.kwargs["session_allowlist"] == [session_id]

    @pytest.mark.asyncio
    async def test_non_member_session_denied(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        session_id = await self._session_with(client, workspace, peer)
        token = create_jwt(JWTParams(w=workspace.name, p=peer.name))

        # One allowlisted session the peer belongs to, one it doesn't:
        # membership must hold for *every* entry.
        resp = self._chat_as(
            client,
            workspace,
            peer,
            token,
            {"filters": {"session_id": [session_id, str(generate_nanoid())]}},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_left_session_denied(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """The regression this gate's `active_only` flag exists to prevent.

        With the loose membership definition a peer that left a session still
        passed here, while the adjacent `session_id` check rejected it.
        """
        workspace, peer = sample_data
        session_id = await self._session_with(client, workspace, peer)
        token = create_jwt(JWTParams(w=workspace.name, p=peer.name))

        await crud.remove_peers_from_session(
            db_session,
            workspace_name=workspace.name,
            session_name=session_id,
            peer_names={peer.name},
        )
        await db_session.commit()

        resp = self._chat_as(
            client, workspace, peer, token, {"filters": {"session_id": [session_id]}}
        )
        assert resp.status_code == 401

        # ...and the single-session gate agrees, which is the whole point.
        resp = self._chat_as(client, workspace, peer, token, {"session_id": session_id})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_workspace_scoped_key_bypasses_gate(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        """Workspace keys are trusted callers — the allowlist passes as given."""
        workspace, peer = sample_data
        token = create_jwt(JWTParams(w=workspace.name))
        foreign = str(generate_nanoid())

        with patch("src.routers.peers.agentic_chat", new=AsyncMock(return_value="ok")):
            resp = self._chat_as(
                client, workspace, peer, token, {"filters": {"session_id": [foreign]}}
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_allowlist_still_gated(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        """`filters={"session_id": []}` is a real allowlist, not an absent one.

        It must reach the gate (and pass trivially, since the empty set is a
        subset of anything) rather than being skipped by a truthiness check.
        """
        workspace, peer = sample_data
        token = create_jwt(JWTParams(w=workspace.name, p=peer.name))

        with patch(
            "src.routers.peers.agentic_chat", new=AsyncMock(return_value="ok")
        ) as mock_chat:
            resp = self._chat_as(
                client, workspace, peer, token, {"filters": {"session_id": []}}
            )
        assert resp.status_code == 200
        assert mock_chat.await_args is not None
        assert mock_chat.await_args.kwargs["session_allowlist"] == []


class TestResolveSessionScope:
    """The tri-state contract the four message-crud call sites depend on."""

    @pytest.mark.asyncio
    async def test_unrestricted_when_no_observer_and_no_allowlist(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ):
        workspace, _ = sample_data
        assert await resolve_session_scope(
            db_session, workspace.name, None, None, None
        ) == (None, False)

    @pytest.mark.asyncio
    async def test_pinned_session_inside_allowlist_passes_through(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ):
        workspace, _ = sample_data
        # None (not [s1]) — the query filters on session_name directly.
        assert await resolve_session_scope(
            db_session, workspace.name, "s1", ["s1", "s2"], None
        ) == (None, False)

    @pytest.mark.asyncio
    async def test_pinned_session_outside_allowlist_denies(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ):
        workspace, _ = sample_data
        assert await resolve_session_scope(
            db_session, workspace.name, "s3", ["s1", "s2"], None
        ) == (None, True)

    @pytest.mark.asyncio
    async def test_empty_allowlist_denies_rather_than_returning_empty_list(
        self, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
    ):
        """Never returns [] — downstream stores drop an empty IN clause."""
        workspace, _ = sample_data
        allowed, deny = await resolve_session_scope(
            db_session, workspace.name, None, [], None
        )
        assert (allowed, deny) == (None, True)

    @pytest.mark.asyncio
    async def test_no_db_touched_when_no_observer_lookup_needed(self):
        """Callers pass db=None on the external-vector-store path.

        The helper must not open a session of its own unless it actually needs
        an observer lookup, or the external semantic lookup stops being the
        first thing that happens (see
        tests/integration/test_message_embeddings.py).
        """
        with patch("src.crud.message.tracked_db") as mock_tracked_db:
            # No observer: pinned session, unrestricted, and plain allowlist.
            assert await resolve_session_scope(None, "w", "s1", None, None) == (
                None,
                False,
            )
            assert await resolve_session_scope(None, "w", None, None, None) == (
                None,
                False,
            )
            assert await resolve_session_scope(None, "w", None, ["s1"], None) == (
                ["s1"],
                False,
            )
        mock_tracked_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_observer_scope_intersected_with_allowlist(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, peer = sample_data
        session_id = str(generate_nanoid())
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/sessions",
            json={"id": session_id, "peer_names": {peer.name: {}}},
        )
        assert resp.status_code == 201

        allowed, deny = await resolve_session_scope(
            db_session, workspace.name, None, [session_id], peer.name
        )
        assert (allowed, deny) == ([session_id], False)

        # Allowlisting only a session the observer isn't in denies outright.
        allowed, deny = await resolve_session_scope(
            db_session, workspace.name, None, [str(generate_nanoid())], peer.name
        )
        assert (allowed, deny) == (None, True)


class TestChatRouteFilterValidation:
    """Filter validation happens before any LLM work — safe to exercise."""

    def _chat(
        self,
        client: TestClient,
        workspace: Workspace,
        peer: Peer,
        body: dict[str, Any],
    ):
        return client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/chat",
            json={"query": "what do you know?", **body},
        )

    def test_unsupported_filter_key_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        resp = self._chat(client, workspace, peer, {"filters": {"peer_id": ["x"]}})
        assert resp.status_code == 422

    def test_bad_filter_shape_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        resp = self._chat(client, workspace, peer, {"filters": {"session_id": 42}})
        assert resp.status_code == 422

    def test_session_id_not_in_allowlist_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        resp = self._chat(
            client,
            workspace,
            peer,
            {"session_id": "s-outside", "filters": {"session_id": ["s1", "s2"]}},
        )
        assert resp.status_code == 422

    def test_allowlist_cap_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        too_many = [f"s{i}" for i in range(MAX_SESSION_ALLOWLIST_ENTRIES + 1)]
        resp = self._chat(
            client, workspace, peer, {"filters": {"session_id": too_many}}
        )
        assert resp.status_code == 422


class TestRepresentationRouteFilters:
    @pytest.mark.asyncio
    async def test_representation_scoped_by_filters(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, peer = sample_data

        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add_all([session_a, session_b])
        await db_session.flush()

        collection = models.Collection(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
        )
        db_session.add(collection)
        await db_session.flush()

        db_session.add_all(
            [
                models.Document(
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                    content="fact from session a",
                    session_name=session_a.name,
                ),
                models.Document(
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                    content="fact from session b",
                    session_name=session_b.name,
                ),
                models.Document(
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                    content="sessionless dream fact",
                    session_name=None,
                ),
            ]
        )
        await db_session.commit()

        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/representation",
            json={"filters": {"session_id": [session_a.name]}},
        )
        assert resp.status_code == 200
        representation = resp.json()["representation"]
        assert "fact from session a" in representation
        assert "fact from session b" not in representation
        assert "sessionless dream fact" not in representation

    def test_session_id_not_in_allowlist_422(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        workspace, peer = sample_data
        resp = client.post(
            f"/v3/workspaces/{workspace.name}/peers/{peer.name}/representation",
            json={"session_id": "s-out", "filters": {"session_id": ["s-in"]}},
        )
        assert resp.status_code == 422

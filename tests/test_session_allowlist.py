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
from src.exceptions import FilterError
from src.models import Peer, Workspace
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

    def test_cap_enforced(self):
        too_many = [f"s{i}" for i in range(MAX_SESSION_ALLOWLIST_ENTRIES + 1)]
        with pytest.raises(FilterError, match="at most"):
            extract_session_allowlist({"session_id": too_many})


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
                session_names=["s1", "s2"],
            )
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] == {
            "level": {"in": ["explicit"]},
            "session_name": {"in": ["s1", "s2"]},
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
                session_names=[],
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
            session_names=[session_a],
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
            session_names=[foreign],
        )
        assert snippets == []

        # Both sessions allowlisted -> both found
        snippets = await crud.grep_messages(
            workspace_name=workspace.name,
            session_name=None,
            text="needle",
            observer=peer.name,
            session_names=[session_a, session_b],
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
            session_names=[session_a],
        )
        assert [m.content for m in messages] == ["needle in alpha"]

        # Empty allowlist fails closed
        messages = await crud.get_messages_by_date_range(
            db_session,
            workspace_name=workspace.name,
            session_name=None,
            observer=peer.name,
            session_names=[],
        )
        assert messages == []


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

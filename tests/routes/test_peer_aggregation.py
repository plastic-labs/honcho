"""Read-time aggregation over peer members.

A read that names a target expands it into the target plus its members, down
to `depth`, and reads each one's collection. These tests drive the read routes
against seeded conclusions; the dialectic's own recall is covered in
tests/dialectic/test_aggregation.py.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.models import Peer, Workspace
from src.security import JWTParams, create_jwt
from src.utils.scopes import scope_peer_name
from tests.conftest import (
    _content_to_embedding,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(autouse=True)
def _pgvector_recall(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    # Documents are seeded straight into postgres, so semantic recall has to go
    # through pgvector rather than a migrated external store.
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)


def _peer(client: TestClient, ws: str, name: str) -> None:
    response = client.post(f"/v3/workspaces/{ws}/peers", json={"id": name})
    assert response.status_code in (200, 201), response.text


def _member(client: TestClient, ws: str, parent: str, member: str) -> None:
    response = client.post(
        f"/v3/workspaces/{ws}/peers/{parent}/members", json={"peer_id": member}
    )
    assert response.status_code in (200, 201), response.text


async def _seed(
    db_session: AsyncSession,
    ws: str,
    *,
    observer: str,
    observed: str,
    contents: list[str],
    level: str = "explicit",
    session_name: str | None = None,
) -> list[models.Document]:
    """Write conclusions into one (observer, observed) collection."""
    exists = await db_session.scalar(
        select(models.Collection.id).where(
            models.Collection.workspace_name == ws,
            models.Collection.observer == observer,
            models.Collection.observed == observed,
        )
    )
    if exists is None:
        db_session.add(
            models.Collection(workspace_name=ws, observer=observer, observed=observed)
        )
        await db_session.flush()
    documents = [
        models.Document(
            workspace_name=ws,
            observer=observer,
            observed=observed,
            content=content,
            level=level,
            session_name=session_name,
            embedding=_content_to_embedding(content),
        )
        for content in contents
    ]
    db_session.add_all(documents)
    await db_session.commit()
    return documents


def _representation(
    client: TestClient, ws: str, peer: str, **body: Any
) -> tuple[int, str]:
    response = client.post(
        f"/v3/workspaces/{ws}/peers/{peer}/representation", json=body
    )
    return response.status_code, (
        response.json().get("representation", "")
        if response.status_code == 200
        else response.text
    )


def _headings(text: str) -> list[str]:
    """The `# <peer>` section headings of an aggregated read, in order."""
    return [line[2:] for line in text.splitlines() if line.startswith("# ")]


@pytest.fixture
async def team(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> str:
    """engineering-team with alice, bob and charlie, each holding their own facts."""
    workspace, _ = sample_data
    ws = workspace.name
    for name in ("engineering-team", "alice", "bob", "charlie"):
        _peer(client, ws, name)
    await _seed(
        db_session,
        ws,
        observer="engineering-team",
        observed="engineering-team",
        contents=["The team ships on Fridays"],
    )
    for name, fact in (
        ("alice", "Alice leads the rewrite"),
        ("bob", "Bob owns the billing service"),
        ("charlie", "Charlie is on call this week"),
    ):
        await _seed(db_session, ws, observer=name, observed=name, contents=[fact])
    return ws


class TestComposition:
    async def test_parent_read_includes_each_member_attributed(
        self, client: TestClient, team: str
    ):
        for name in ("alice", "bob", "charlie"):
            _member(client, team, "engineering-team", name)

        status, text = _representation(client, team, "engineering-team")
        assert status == 200
        for heading, fact in (
            ("# engineering-team", "The team ships on Fridays"),
            ("# alice", "Alice leads the rewrite"),
            ("# bob", "Bob owns the billing service"),
            ("# charlie", "Charlie is on call this week"),
        ):
            assert heading in text
            assert fact in text
        # The target's own section comes first.
        assert text.index("# engineering-team") < text.index("# alice")
        # Each fact sits under its own peer's heading.
        assert text.index("Alice leads") < text.index("# bob")

    async def test_a_member_is_attributed_even_when_it_alone_has_results(
        self, client: TestClient, team: str
    ):
        _peer(client, team, "empty-team")
        _member(client, team, "empty-team", "alice")

        _, text = _representation(client, team, "empty-team")
        assert text.startswith("# alice")
        assert "Alice leads the rewrite" in text

    async def test_member_read_sees_only_the_member(
        self, client: TestClient, team: str
    ):
        for name in ("alice", "bob"):
            _member(client, team, "engineering-team", name)

        status, text = _representation(client, team, "alice")
        assert status == 200
        assert "Alice leads the rewrite" in text
        assert "ships on Fridays" not in text
        assert "Bob owns" not in text
        assert "# alice" not in text  # nothing aggregated, so no headings

    async def test_duplicates_across_members_are_attributed_not_merged(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        shared = "We chose Rust for the rewrite"
        await _seed(
            db_session, team, observer="alice", observed="alice", contents=[shared]
        )
        await _seed(db_session, team, observer="bob", observed="bob", contents=[shared])
        # Same text and timestamp: only the peer tells them apart.
        await db_session.execute(
            update(models.Document)
            .where(models.Document.content == shared)
            .values(internal_metadata={"message_created_at": "2026-01-01T00:00:00Z"})
        )
        await db_session.commit()
        for name in ("alice", "bob"):
            _member(client, team, "engineering-team", name)

        _, text = _representation(client, team, "engineering-team")
        assert text.count(shared) == 2
        alice_section = text[text.index("# alice") : text.index("# bob")]
        assert shared in alice_section
        assert shared in text[text.index("# bob") :]


class TestDepthAndBounds:
    async def test_depth_zero_equals_the_pre_membership_response(
        self, client: TestClient, team: str
    ):
        before = _representation(client, team, "engineering-team")
        for name in ("alice", "bob"):
            _member(client, team, "engineering-team", name)

        assert _representation(client, team, "engineering-team", depth=0) == before
        # A peer with no members reads the same at every depth.
        alice = _representation(client, team, "alice", depth=0)
        assert _representation(client, team, "alice", depth=3) == alice

    async def test_depth_two_reaches_members_of_members(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        _peer(client, team, "platform")
        await _seed(
            db_session,
            team,
            observer="platform",
            observed="platform",
            contents=["Platform runs the deploy pipeline"],
        )
        _member(client, team, "platform", "engineering-team")
        _member(client, team, "engineering-team", "alice")

        _, depth_one = _representation(client, team, "platform", depth=1)
        assert "Platform runs" in depth_one
        assert "ships on Fridays" in depth_one
        assert "Alice leads" not in depth_one

        _, depth_two = _representation(client, team, "platform", depth=2)
        assert "Alice leads" in depth_two

    async def test_depth_above_the_maximum_is_rejected(
        self, client: TestClient, team: str
    ):
        status, body = _representation(
            client,
            team,
            "engineering-team",
            depth=settings.PEER.AGGREGATION_MAX_DEPTH + 1,
        )
        assert status == 422
        assert "Lower the depth" in body

    async def test_fanout_cap_is_enforced(
        self, client: TestClient, team: str, monkeypatch: pytest.MonkeyPatch
    ):
        for name in ("alice", "bob", "charlie"):
            _member(client, team, "engineering-team", name)
        monkeypatch.setattr(settings.PEER, "AGGREGATION_MAX_FANOUT", 3)

        status, body = _representation(client, team, "engineering-team")
        assert status == 422
        assert "more than 3 peers" in body
        # Querying a member directly still works.
        assert _representation(client, team, "alice")[0] == 200


class TestObserverPairs:
    async def test_a_directional_read_keeps_its_observer_for_every_member(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        for observed, fact in (
            ("engineering-team", "Bob thinks the team is behind"),
            ("alice", "Bob thinks Alice is overloaded"),
        ):
            await _seed(
                db_session, team, observer="bob", observed=observed, contents=[fact]
            )
        _member(client, team, "engineering-team", "alice")

        status, text = _representation(client, team, "bob", target="engineering-team")
        assert status == 200
        assert _headings(text) == ["engineering-team", "alice"]
        assert "Alice is overloaded" in text
        assert "Alice leads the rewrite" not in text  # alice's own memory
        assert "ships on Fridays" not in text  # the team's own memory

    async def test_a_self_read_uses_each_members_own_memory_not_the_parents_view(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        await _seed(
            db_session,
            team,
            observer="engineering-team",
            observed="alice",
            contents=["The team's view of Alice"],
        )
        _member(client, team, "engineering-team", "alice")

        _, text = _representation(client, team, "engineering-team")
        assert "Alice leads the rewrite" in text
        assert "team's view of Alice" not in text

    async def test_a_session_allowlist_applies_to_every_member(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        kept, dropped = str(generate_nanoid()), str(generate_nanoid())
        for session in (kept, dropped):
            assert client.post(
                f"/v3/workspaces/{team}/sessions", json={"id": session}
            ).status_code in (200, 201)
        for name in ("alice", "bob"):
            for session, verb in ((kept, "kept"), (dropped, "dropped")):
                await _seed(
                    db_session,
                    team,
                    observer=name,
                    observed=name,
                    contents=[f"{name} {verb} this"],
                    session_name=session,
                )
            _member(client, team, "engineering-team", name)

        status, text = _representation(
            client, team, "engineering-team", filters={"session_id": [kept]}
        )
        assert status == 200
        assert "alice kept this" in text
        assert "bob kept this" in text
        assert "dropped" not in text

    async def test_a_parent_key_reads_its_members_through_the_parent(
        self, client: TestClient, team: str, monkeypatch: pytest.MonkeyPatch
    ):
        _member(client, team, "engineering-team", "alice")
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        team_key = create_jwt(JWTParams(w=team, p="engineering-team"))
        client.headers["Authorization"] = f"Bearer {team_key}"

        status, text = _representation(client, team, "engineering-team")
        assert status == 200
        assert "Alice leads the rewrite" in text
        # Membership grants the read through the parent, not of the member itself.
        assert _representation(client, team, "alice")[0] == 401


class TestScopes:
    async def test_single_scope_reads_the_scopes_view_of_every_member(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        assert client.post(
            f"/v3/workspaces/{team}/scopes", json={"id": "q3-planning"}
        ).status_code in (200, 201)
        scope = scope_peer_name("q3-planning")
        await _seed(
            db_session,
            team,
            observer=scope,
            observed="engineering-team",
            contents=["In planning the team committed to the Q3 launch"],
        )
        await _seed(
            db_session,
            team,
            observer=scope,
            observed="alice",
            contents=["In planning Alice took the migration"],
            level="inductive",
        )
        _member(client, team, "engineering-team", "alice")

        status, text = _representation(
            client, team, "engineering-team", scope="q3-planning"
        )
        assert status == 200
        assert "committed to the Q3 launch" in text
        # Scope-level dreams (derived levels) come through for members too.
        assert "Alice took the migration" in text
        # Global conclusions stay outside the scope.
        assert "Alice leads the rewrite" not in text
        assert "ships on Fridays" not in text

    async def test_scope_list_admits_only_explicit_across_members(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        in_scope = str(generate_nanoid())
        out_of_scope = str(generate_nanoid())
        for session in (in_scope, out_of_scope):
            assert client.post(
                f"/v3/workspaces/{team}/sessions", json={"id": session}
            ).status_code in (200, 201)
        for scope_name in ("q3", "q4"):
            client.post(f"/v3/workspaces/{team}/scopes", json={"id": scope_name})
        assert (
            client.post(
                f"/v3/workspaces/{team}/scopes/q3/sessions",
                json={"session_ids": [in_scope]},
            ).status_code
            == 204
        )
        await _seed(
            db_session,
            team,
            observer="alice",
            observed="alice",
            contents=["Alice said this in a scoped session"],
            session_name=in_scope,
        )
        await _seed(
            db_session,
            team,
            observer="alice",
            observed="alice",
            contents=["Alice deduced this from many sessions"],
            level="deductive",
            session_name=in_scope,
        )
        await _seed(
            db_session,
            team,
            observer="alice",
            observed="alice",
            contents=["Alice said this somewhere else"],
            session_name=out_of_scope,
        )
        _member(client, team, "engineering-team", "alice")

        status, text = _representation(
            client, team, "engineering-team", scope=["q3", "q4"]
        )
        assert status == 200
        assert "Alice said this in a scoped session" in text
        assert "deduced" not in text
        assert "somewhere else" not in text

    async def test_a_scope_peer_never_joins_an_expanded_set(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        client.post(f"/v3/workspaces/{team}/scopes", json={"id": "forced"})
        scope = scope_peer_name("forced")
        await _seed(
            db_session,
            team,
            observer=scope,
            observed=scope,
            contents=["A scope never has a representation of its own"],
        )
        rows = await db_session.execute(
            select(models.Peer.name, models.Peer.id).where(
                models.Peer.workspace_name == team,
                models.Peer.name.in_(["engineering-team", scope]),
            )
        )
        ids: dict[str, str] = {name: peer_id for name, peer_id in rows.all()}
        # Bypass the route's guard: the read side must hold on its own.
        db_session.add(
            models.PeerMembership(
                workspace_name=team,
                parent_id=ids["engineering-team"],
                member_id=ids[scope],
            )
        )
        await db_session.commit()

        assert await crud.expand_observed(
            db_session, team, "engineering-team", depth=3
        ) == ["engineering-team"]
        _, text = _representation(client, team, "engineering-team")
        assert "never has a representation" not in text


class TestIsolation:
    async def test_identical_names_in_another_workspace_never_aggregate(
        self,
        client: TestClient,
        db_session: AsyncSession,
        team: str,
    ):
        other = str(generate_nanoid())
        assert client.post("/v3/workspaces", json={"id": other}).status_code in (
            200,
            201,
        )
        for name in ("engineering-team", "alice"):
            _peer(client, other, name)
        await _seed(
            db_session,
            other,
            observer="alice",
            observed="alice",
            contents=["Other-workspace Alice is a secret"],
        )
        _member(client, team, "engineering-team", "alice")

        _, text = _representation(client, team, "engineering-team")
        assert "Alice leads the rewrite" in text
        assert "secret" not in text
        # And the membership does not exist over there.
        _, other_text = _representation(client, other, "engineering-team")
        assert "secret" not in other_text


class TestDeletion:
    async def test_a_tombstoned_member_conclusion_leaves_the_parent_read(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        [doomed] = await _seed(
            db_session,
            team,
            observer="alice",
            observed="alice",
            contents=["Alice prefers tabs"],
        )
        _member(client, team, "engineering-team", "alice")
        assert (
            "Alice prefers tabs" in _representation(client, team, "engineering-team")[1]
        )

        response = client.delete(f"/v3/workspaces/{team}/conclusions/{doomed.id}")
        assert response.status_code in (200, 204), response.text

        _, text = _representation(client, team, "engineering-team")
        assert "Alice prefers tabs" not in text
        assert "Alice leads the rewrite" in text

    async def test_deleting_a_peer_removes_its_memberships_on_both_sides(
        self, client: TestClient, db_session: AsyncSession, team: str
    ):
        _peer(client, team, "dave")
        _member(client, team, "engineering-team", "dave")
        _member(client, team, "dave", "alice")

        await db_session.execute(
            delete(models.Peer).where(
                models.Peer.workspace_name == team, models.Peer.name == "dave"
            )
        )
        await db_session.commit()

        remaining = await db_session.scalars(
            select(models.PeerMembership).where(
                models.PeerMembership.workspace_name == team
            )
        )
        assert remaining.all() == []


class TestCardContextAndSearch:
    async def test_card_and_context_carry_member_cards_only_when_aggregated(
        self, client: TestClient, team: str
    ):
        for name, card in (("alice", ["Name: Alice"]), ("engineering-team", ["Team"])):
            assert (
                client.put(
                    f"/v3/workspaces/{team}/peers/{name}/card",
                    json={"peer_card": card},
                ).status_code
                == 200
            )
        _member(client, team, "engineering-team", "alice")
        _member(client, team, "engineering-team", "bob")

        card = client.get(f"/v3/workspaces/{team}/peers/engineering-team/card").json()
        assert card == {
            "peer_card": ["Team"],
            "member_cards": {"alice": ["Name: Alice"], "bob": None},
        }
        flat = client.get(
            f"/v3/workspaces/{team}/peers/engineering-team/card?depth=0"
        ).json()
        assert flat == {"peer_card": ["Team"]}

        context = client.get(
            f"/v3/workspaces/{team}/peers/engineering-team/context"
        ).json()
        assert context["member_cards"] == {"alice": ["Name: Alice"], "bob": None}
        assert "# alice" in context["representation"]
        assert "Alice leads the rewrite" in context["representation"]

        alone = client.get(f"/v3/workspaces/{team}/peers/alice/context").json()
        assert "member_cards" not in alone

    async def test_search_matches_messages_written_by_members(
        self, client: TestClient, team: str
    ):
        session = str(generate_nanoid())
        assert client.post(
            f"/v3/workspaces/{team}/sessions",
            json={"id": session, "peers": {"alice": {}, "engineering-team": {}}},
        ).status_code in (200, 201)
        assert client.post(
            f"/v3/workspaces/{team}/sessions/{session}/messages",
            json={
                "messages": [
                    {"peer_id": "alice", "content": "the zeppelin migration is done"}
                ]
            },
        ).status_code in (200, 201)
        _member(client, team, "engineering-team", "alice")

        def search(depth: int | None) -> list[str]:
            body: dict[str, Any] = {"query": "zeppelin"}
            if depth is not None:
                body["depth"] = depth
            response = client.post(
                f"/v3/workspaces/{team}/peers/engineering-team/search", json=body
            )
            assert response.status_code == 200, response.text
            return [m["peer_id"] for m in response.json()]

        assert search(None) == ["alice"]
        assert search(0) == []


# org
# |-- platform -- ana, ben, cal, sre -- hal
# |-- product --- cal, dee, eve
# |-- research -- fay, gus
# `-- gus           (also a direct member, added last)
#
# Membership rows in creation order: each level of an expansion is ordered by
# when its memberships were created. org -> gus comes last, so only its level
# explains gus being read at depth 1.
ORG_EDGES = [
    ("org", "platform"),
    ("org", "product"),
    ("org", "research"),
    ("platform", "ana"),
    ("platform", "ben"),
    ("platform", "cal"),
    ("platform", "sre"),
    ("product", "cal"),
    ("product", "dee"),
    ("product", "eve"),
    ("research", "fay"),
    ("research", "gus"),
    ("sre", "hal"),
    ("org", "gus"),
]
ORG_LEVELS = [
    ["org"],
    ["platform", "product", "research", "gus"],
    ["ana", "ben", "cal", "sre", "dee", "eve", "fay"],
    ["hal"],
]


def _org_fact(peer: str) -> str:
    return f"{peer} holds fact {peer}-1"


def _org_expected(depth: int) -> list[str]:
    return [peer for level in ORG_LEVELS[: depth + 1] for peer in level]


@pytest.fixture
async def org(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> str:
    """The org chart above, every peer holding one fact of its own."""
    workspace, _ = sample_data
    ws = workspace.name
    for peer in _org_expected(3):
        _peer(client, ws, peer)
        await _seed(
            db_session, ws, observer=peer, observed=peer, contents=[_org_fact(peer)]
        )
    for parent, member in ORG_EDGES:
        _member(client, ws, parent, member)
    return ws


class TestOrgChart:
    @pytest.mark.parametrize("depth", [1, 2, 3])
    async def test_each_depth_adds_exactly_one_level(
        self, client: TestClient, org: str, depth: int
    ):
        status, text = _representation(
            client, org, "org", depth=depth, max_conclusions=50
        )
        assert status == 200
        assert _headings(text) == _org_expected(depth)
        for peer in _org_expected(depth):
            assert _org_fact(peer) in text, peer

    async def test_a_peer_on_several_paths_is_read_once_at_its_shallowest_level(
        self, client: TestClient, org: str, monkeypatch: pytest.MonkeyPatch
    ):
        # A cap of exactly the distinct peers: counting cal or gus twice trips it.
        monkeypatch.setattr(
            settings.PEER, "AGGREGATION_MAX_FANOUT", len(_org_expected(3))
        )
        status, text = _representation(client, org, "org", depth=3, max_conclusions=50)
        assert status == 200, text
        headings = _headings(text)
        # cal is under platform and product; gus under research and org itself.
        for peer in ("cal", "gus"):
            assert headings.count(peer) == 1, peer
            assert text.count(_org_fact(peer)) == 1, peer
        assert headings.index("gus") < headings.index("ana")

    async def test_removing_a_middle_link_drops_its_subtree(
        self, client: TestClient, org: str
    ):
        response = client.delete(f"/v3/workspaces/{org}/peers/org/members/platform")
        assert response.status_code in (200, 204), response.text

        _, text = _representation(client, org, "org", depth=3, max_conclusions=50)
        headings = _headings(text)
        for gone in ("platform", "ana", "ben", "sre", "hal"):
            assert gone not in headings, gone
        assert "cal" in headings  # still reached through product
        _, subtree = _representation(client, org, "platform", depth=2)
        assert _headings(subtree) == ["platform", "ana", "ben", "cal", "sre", "hal"]

    async def test_the_fanout_cap_counts_only_peers_within_the_requested_depth(
        self, client: TestClient, org: str, monkeypatch: pytest.MonkeyPatch
    ):
        # Exactly the depth-1 set; the eight peers below must not count toward it.
        monkeypatch.setattr(
            settings.PEER, "AGGREGATION_MAX_FANOUT", len(_org_expected(1))
        )

        assert _representation(client, org, "org", depth=1)[0] == 200
        status, body = _representation(client, org, "org", depth=2)
        assert status == 422
        assert "more than 5 peers" in body

    async def test_member_cards_are_one_flat_map_over_every_level(
        self, client: TestClient, org: str
    ):
        for peer in ("platform", "cal", "hal"):
            assert (
                client.put(
                    f"/v3/workspaces/{org}/peers/{peer}/card",
                    json={"peer_card": [f"Name: {peer}"]},
                ).status_code
                == 200
            )

        cards = client.get(f"/v3/workspaces/{org}/peers/org/card?depth=3").json()[
            "member_cards"
        ]
        assert list(cards) == _org_expected(3)[1:]
        assert cards["hal"] == ["Name: hal"]  # not nested under sre
        assert cards["ben"] is None
        shallower = client.get(f"/v3/workspaces/{org}/peers/org/card?depth=2").json()
        assert "hal" not in shallower["member_cards"]

    async def test_a_directional_read_keeps_its_observer_at_every_level(
        self, client: TestClient, db_session: AsyncSession, org: str
    ):
        _peer(client, org, "zed")
        for observed in ("org", "platform", "ana", "hal"):
            await _seed(
                db_session,
                org,
                observer="zed",
                observed=observed,
                contents=[f"zed's view of {observed}"],
            )

        _, text = _representation(client, org, "zed", target="org", depth=3)
        assert _headings(text) == ["org", "platform", "ana", "hal"]
        assert "zed's view of hal" in text
        assert " holds fact " not in text  # no peer's own memory

    async def test_search_reaches_the_deepest_level_only_at_that_depth(
        self, client: TestClient, org: str
    ):
        session = str(generate_nanoid())
        assert client.post(
            f"/v3/workspaces/{org}/sessions",
            json={"id": session, "peers": {"hal": {}}},
        ).status_code in (200, 201)
        assert client.post(
            f"/v3/workspaces/{org}/sessions/{session}/messages",
            json={
                "messages": [{"peer_id": "hal", "content": "the quokka drill passed"}]
            },
        ).status_code in (200, 201)

        def search(depth: int) -> list[str]:
            response = client.post(
                f"/v3/workspaces/{org}/peers/org/search",
                json={"query": "quokka", "depth": depth},
            )
            assert response.status_code == 200, response.text
            return [m["peer_id"] for m in response.json()]

        assert search(3) == ["hal"]
        assert search(2) == []

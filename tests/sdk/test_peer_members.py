"""SDK tests for composable peers: membership management and member aggregation."""

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import patch

import pytest

from sdks.python.src.honcho.api_types import PeerMembership
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.conclusions import ConclusionCreateParams
from sdks.python.src.honcho.http.exceptions import (
    ConflictError,
    UnprocessableEntityError,
)
from sdks.python.src.honcho.pagination import AsyncPage, SyncPage


@pytest.mark.asyncio
async def test_members_add_list_remove(client_fixture: tuple[Honcho, str]):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("members-team")
    bob = honcho_client.peer("members-bob")

    if client_type == "async":
        alice_membership = await team.members.aio.add(
            "members-alice", metadata={"role": "lead"}
        )
        await team.aio.members.add(bob)
        page = await team.members.aio.list()
        assert isinstance(page, AsyncPage)
        listed = [m.peer_id for m in page.items]
        await team.members.aio.remove("members-alice")
        # Removing a peer that is not a member is a no-op.
        await team.members.aio.remove("members-alice")
        remaining = [m.peer_id for m in (await team.members.aio.list()).items]
    else:
        alice_membership = team.members.add("members-alice", metadata={"role": "lead"})
        team.members.add(bob)
        page = team.members.list()
        assert isinstance(page, SyncPage)
        listed = [m.peer_id for m in page]
        team.members.remove("members-alice")
        team.members.remove("members-alice")
        remaining = [m.peer_id for m in team.members.list().items]

    assert isinstance(alice_membership, PeerMembership)
    assert alice_membership.parent_id == team.id
    assert alice_membership.peer_id == "members-alice"
    assert alice_membership.workspace_id == honcho_client.workspace_id
    assert alice_membership.metadata == {"role": "lead"}
    # Oldest membership first.
    assert listed == ["members-alice", bob.id]
    assert remaining == [bob.id]


@pytest.mark.asyncio
async def test_members_list_filters_on_the_member_peer(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("filter-team")
    honcho_client.peer("filter-alice", metadata={"kind": "human"})
    honcho_client.peer("filter-bot", metadata={"kind": "agent"})
    team.members.add("filter-alice")
    team.members.add("filter-bot")

    filters = {"metadata": {"kind": "agent"}}
    if client_type == "async":
        page = await team.members.aio.list(filters)
    else:
        page = team.members.list(filters)

    assert [m.peer_id for m in page.items] == ["filter-bot"]


@pytest.mark.asyncio
async def test_parents_lists_the_peers_a_peer_belongs_to(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture
    alice = honcho_client.peer("parents-alice")
    honcho_client.peer("parents-team").members.add(alice)
    honcho_client.peer("parents-guild").members.add(alice)

    if client_type == "async":
        page = await alice.aio.parents()
    else:
        page = alice.parents()

    assert [m.parent_id for m in page.items] == ["parents-team", "parents-guild"]
    assert all(m.peer_id == alice.id for m in page.items)


@pytest.mark.asyncio
async def test_re_adding_a_member_returns_the_existing_membership(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("readd-team")

    if client_type == "async":
        first = await team.members.aio.add("readd-alice", metadata={"role": "ic"})
        again = await team.members.aio.add("readd-alice")
        replaced = await team.members.aio.add("readd-alice", metadata={"role": "lead"})
        count = len((await team.members.aio.list()).items)
    else:
        first = team.members.add("readd-alice", metadata={"role": "ic"})
        again = team.members.add("readd-alice")
        replaced = team.members.add("readd-alice", metadata={"role": "lead"})
        count = len(team.members.list().items)

    assert again.created_at == first.created_at
    # Metadata is kept when none is given, and replaced when it is.
    assert again.metadata == {"role": "ic"}
    assert replaced.metadata == {"role": "lead"}
    assert count == 1


@pytest.mark.asyncio
async def test_a_cycle_raises_conflict_error(client_fixture: tuple[Honcho, str]):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("cycle-team")
    alice = honcho_client.peer("cycle-alice")
    team.members.add(alice)

    with pytest.raises(ConflictError):
        if client_type == "async":
            await alice.members.aio.add(team)
        else:
            alice.members.add(team)


@pytest.mark.asyncio
async def test_a_peer_cannot_be_its_own_member(client_fixture: tuple[Honcho, str]):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("self-team")

    with pytest.raises(UnprocessableEntityError):
        if client_type == "async":
            await team.members.aio.add(team)
        else:
            team.members.add(team)


def test_remove_rejects_a_malformed_member_id(honcho_sync_test_client: Honcho):
    # The ID lands in the request path, so `alice?typo` would remove `alice`.
    team = honcho_sync_test_client.peer("malformed-team")
    with pytest.raises(ValueError, match="must match pattern"):
        team.members.remove("alice?typo")


@pytest.mark.asyncio
async def test_card_and_context_surface_member_cards_only_when_aggregated(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("cards-team")
    alice = honcho_client.peer("cards-alice")
    team.set_card(["The team ships on Fridays"])
    alice.set_card(["Alice leads the rewrite"])
    team.members.add(alice)

    if client_type == "async":
        card = await team.aio.get_card()
        aggregated = await team.aio.context()
        alone = await team.aio.context(depth=0)
    else:
        card = team.get_card()
        aggregated = team.context()
        alone = team.context(depth=0)

    # The card getter returns the target's own card; members' come via context.
    assert card == ["The team ships on Fridays"]
    assert aggregated.peer_card == ["The team ships on Fridays"]
    assert aggregated.member_cards == {alice.id: ["Alice leads the rewrite"]}
    assert alone.member_cards is None


@pytest.mark.asyncio
async def test_representation_aggregates_member_conclusions_by_depth(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("rep-team")
    alice = honcho_client.peer("rep-alice")
    alice.conclusions.create(
        [ConclusionCreateParams(content="Alice leads the rewrite")]
    )
    team.members.add(alice)

    if client_type == "async":
        aggregated = await team.aio.representation()
        alone = await team.aio.representation(depth=0)
    else:
        aggregated = team.representation()
        alone = team.representation(depth=0)

    assert "Alice leads the rewrite" in aggregated
    assert "Alice leads the rewrite" not in alone


@pytest.mark.asyncio
async def test_search_matches_member_messages_by_depth(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture
    team = honcho_client.peer("search-team")
    alice = honcho_client.peer("search-alice")
    session = honcho_client.session("search-members-session")
    content = "the rewrite will use a unique storage engine"
    session.add_messages([alice.message(content)])
    team.members.add(alice)

    if client_type == "async":
        aggregated = await team.aio.search(content)
        alone = await team.aio.search(content, depth=0)
    else:
        aggregated = team.search(content)
        alone = team.search(content, depth=0)

    assert [m.peer_id for m in aggregated] == [alice.id]
    assert alone == []


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [None, 0, 2])
async def test_chat_sends_depth_only_when_given(
    client_fixture: tuple[Honcho, str], depth: int | None
):
    honcho_client, client_type = client_fixture
    peer = honcho_client.peer(f"chat-depth-{client_type}")

    if client_type == "async":

        async def mock_post(*_args: object, **_kwargs: object) -> dict[str, str]:
            return {"content": "ok"}

        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "post",
            side_effect=mock_post,
        ) as mock:
            await peer.aio.chat("what did we decide?", depth=depth)
    else:
        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "post",
            return_value={"content": "ok"},
        ) as mock:
            peer.chat("what did we decide?", depth=depth)

    body: dict[str, Any] = mock.call_args.kwargs["body"]
    if depth is None:
        assert "depth" not in body
    else:
        assert body["depth"] == depth


@pytest.mark.asyncio
async def test_chat_stream_sends_depth(client_fixture: tuple[Honcho, str]):
    honcho_client, client_type = client_fixture
    peer = honcho_client.peer(f"chat-stream-depth-{client_type}")

    if client_type == "async":

        async def mock_astream(
            *_args: object, **_kwargs: object
        ) -> AsyncIterator[bytes]:
            yield b'data: {"done": true}\n'

        with patch.object(
            peer._honcho._async_http_client,  # pyright: ignore[reportPrivateUsage]
            "stream",
            side_effect=mock_astream,
        ) as mock:
            result = await peer.aio.chat_stream("q", depth=2)
            _ = [chunk async for chunk in result]
    else:

        def mock_stream(*_args: object, **_kwargs: object) -> Iterator[bytes]:
            yield b'data: {"done": true}\n'

        with patch.object(
            peer._honcho._http,  # pyright: ignore[reportPrivateUsage]
            "stream",
            side_effect=mock_stream,
        ) as mock:
            _ = list(peer.chat_stream("q", depth=2))

    assert mock.call_args.kwargs["body"]["depth"] == 2


def test_negative_depth_is_rejected_before_the_request(
    honcho_sync_test_client: Honcho,
):
    peer = honcho_sync_test_client.peer("negative-depth")
    with pytest.raises(ValueError):
        peer.representation(depth=-1)

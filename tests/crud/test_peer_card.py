import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.peer_card import construct_peer_card_label, get_peer_card, set_peer_card


@pytest.mark.asyncio
async def test_peer_card_get_set_roundtrip(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Roundtrip set/get for peer card on a valid peer."""
    workspace, peer = sample_data

    # Initially absent
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer.name, observed=peer.name
        )
        is None
    )

    # Set and read back
    value_1 = ["Initial peer card text"]
    await set_peer_card(
        db_session,
        workspace.name,
        value_1,
        observer=peer.name,
        observed=peer.name,
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer.name, observed=peer.name
        )
        == value_1
    )

    # Update and read back
    value_2 = ["Updated peer card text", "Another line"]
    await set_peer_card(
        db_session, workspace.name, value_2, observer=peer.name, observed=peer.name
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer.name, observed=peer.name
        )
        == value_2
    )


@pytest.mark.asyncio
async def test_get_peer_card_missing_peer_returns_none(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Getting a peer card for a non-existent peer should return None."""
    workspace, _existing_peer = sample_data
    result = await get_peer_card(
        db_session,
        workspace.name,
        observer="missing-peer",
        observed="missing-peer",
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_peer_card_missing_workspace_returns_none(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Getting a peer card for a non-existent workspace should return None."""
    _workspace, peer = sample_data
    result = await get_peer_card(
        db_session,
        "missing-workspace",
        observer=peer.name,
        observed=peer.name,
    )
    assert result is None


@pytest.mark.asyncio
async def test_peer_card_empty_list(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Setting and getting an empty peer card should work."""
    workspace, peer = sample_data

    empty_card: list[str] = []
    await set_peer_card(
        db_session,
        workspace.name,
        empty_card,
        observer=peer.name,
        observed=peer.name,
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer.name, observed=peer.name
        )
        == empty_card
    )


@pytest.mark.asyncio
async def test_peer_card_multiple_lines(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Peer card should support multiple lines."""
    workspace, peer = sample_data

    multi_line_card = [
        "First observation about the peer",
        "Second observation about the peer",
        "Third observation about the peer",
        "Fourth observation about the peer",
    ]
    await set_peer_card(
        db_session,
        workspace.name,
        multi_line_card,
        observer=peer.name,
        observed=peer.name,
    )
    result = await get_peer_card(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert result is not None
    assert result == multi_line_card
    assert len(result) == 4


@pytest.mark.asyncio
async def test_peer_card_different_observer_observed(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Test peer card with different observer and observed peers."""
    workspace, peer1 = sample_data

    # Create second peer
    peer2 = models.Peer(name="test-peer-2", workspace_name=workspace.name)
    db_session.add(peer2)
    await db_session.flush()

    # Peer1 observes peer2
    card_1_to_2 = ["Peer 1's observation of peer 2"]
    await set_peer_card(
        db_session,
        workspace.name,
        card_1_to_2,
        observer=peer1.name,
        observed=peer2.name,
    )

    # Peer2 observes peer1
    card_2_to_1 = ["Peer 2's observation of peer 1"]
    await set_peer_card(
        db_session,
        workspace.name,
        card_2_to_1,
        observer=peer2.name,
        observed=peer1.name,
    )

    # Verify each card is independent
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer1.name, observed=peer2.name
        )
        == card_1_to_2
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer2.name, observed=peer1.name
        )
        == card_2_to_1
    )


@pytest.mark.asyncio
async def test_peer_card_self_and_other_observations(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Test that a peer can have both a self-observation and observations of others."""
    workspace, peer1 = sample_data

    # Create second peer
    peer2 = models.Peer(name="test-peer-2", workspace_name=workspace.name)
    db_session.add(peer2)
    await db_session.flush()

    # Peer1's self-observation
    self_card = ["Self observation"]
    await set_peer_card(
        db_session,
        workspace.name,
        self_card,
        observer=peer1.name,
        observed=peer1.name,
    )

    # Peer1's observation of peer2
    other_card = ["Observation of other peer"]
    await set_peer_card(
        db_session,
        workspace.name,
        other_card,
        observer=peer1.name,
        observed=peer2.name,
    )

    # Both should be retrievable independently
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer1.name, observed=peer1.name
        )
        == self_card
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer1.name, observed=peer2.name
        )
        == other_card
    )


@pytest.mark.asyncio
async def test_peer_card_multiple_observers_same_observed(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Test that multiple peers can observe the same peer with different cards."""
    workspace, peer1 = sample_data

    # Create two more peers
    peer2 = models.Peer(name="test-peer-2", workspace_name=workspace.name)
    peer3 = models.Peer(name="test-peer-3", workspace_name=workspace.name)
    db_session.add_all([peer2, peer3])
    await db_session.flush()

    # Both peer2 and peer3 observe peer1
    card_2_to_1 = ["Peer 2's view of peer 1"]
    card_3_to_1 = ["Peer 3's view of peer 1"]

    await set_peer_card(
        db_session,
        workspace.name,
        card_2_to_1,
        observer=peer2.name,
        observed=peer1.name,
    )
    await set_peer_card(
        db_session,
        workspace.name,
        card_3_to_1,
        observer=peer3.name,
        observed=peer1.name,
    )

    # Each observer should have their own independent card
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer2.name, observed=peer1.name
        )
        == card_2_to_1
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer3.name, observed=peer1.name
        )
        == card_3_to_1
    )


@pytest.mark.asyncio
async def test_peer_card_update_does_not_affect_others(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Test that updating one peer card doesn't affect other peer cards."""
    workspace, peer1 = sample_data

    # Create second peer
    peer2 = models.Peer(name="test-peer-2", workspace_name=workspace.name)
    db_session.add(peer2)
    await db_session.flush()

    # Set initial cards
    card_1 = ["Peer 1 self observation"]
    card_2 = ["Peer 1's observation of peer 2"]

    await set_peer_card(
        db_session,
        workspace.name,
        card_1,
        observer=peer1.name,
        observed=peer1.name,
    )
    await set_peer_card(
        db_session,
        workspace.name,
        card_2,
        observer=peer1.name,
        observed=peer2.name,
    )

    # Update one card
    new_card_1 = ["Updated self observation"]
    await set_peer_card(
        db_session,
        workspace.name,
        new_card_1,
        observer=peer1.name,
        observed=peer1.name,
    )

    # Verify only the updated card changed
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer1.name, observed=peer1.name
        )
        == new_card_1
    )
    assert (
        await get_peer_card(
            db_session, workspace.name, observer=peer1.name, observed=peer2.name
        )
        == card_2
    )


@pytest.mark.asyncio
async def test_peer_card_special_characters(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Test that peer cards handle special characters correctly."""
    workspace, peer = sample_data

    special_card = [
        "Line with special chars: @#$%^&*()",
        "Line with unicode: 你好世界 🌍",
        "Line with quotes: \"double\" and 'single'",
        "Line with newlines embedded\\n",
    ]
    await set_peer_card(
        db_session,
        workspace.name,
        special_card,
        observer=peer.name,
        observed=peer.name,
    )
    result = await get_peer_card(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert result == special_card


@pytest.mark.asyncio
async def test_peer_card_large_content(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Test that peer cards can handle large amounts of content."""
    workspace, peer = sample_data

    # Create a large peer card with many observations
    large_card = [f"Observation number {i}" for i in range(100)]
    await set_peer_card(
        db_session,
        workspace.name,
        large_card,
        observer=peer.name,
        observed=peer.name,
    )
    result = await get_peer_card(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert result == large_card
    assert result is not None
    assert len(result) == 100


def test_construct_helpers_labels():
    """Helper label constructors return expected values."""
    assert construct_peer_card_label(observer="a", observed="a") == "peer_card"
    assert construct_peer_card_label(observer="a", observed="b") == "b_peer_card"


def test_construct_peer_card_label_with_special_chars():
    """Test label construction with special characters in peer names."""
    # Test with dashes and underscores
    assert (
        construct_peer_card_label(observer="peer-1", observed="peer-2")
        == "peer-2_peer_card"
    )
    assert (
        construct_peer_card_label(observer="peer_1", observed="peer_2")
        == "peer_2_peer_card"
    )

    # Test with same observer and observed with special chars
    assert (
        construct_peer_card_label(observer="peer-1", observed="peer-1") == "peer_card"
    )


def test_construct_peer_card_label_edge_cases():
    """Test label construction edge cases."""
    # Test with empty strings (though this shouldn't happen in practice)
    assert construct_peer_card_label(observer="", observed="") == "peer_card"
    assert construct_peer_card_label(observer="a", observed="") == "_peer_card"

    # Test with very long peer names
    long_name = "a" * 100
    assert (
        construct_peer_card_label(observer=long_name, observed=long_name) == "peer_card"
    )
    assert (
        construct_peer_card_label(observer="a", observed=long_name)
        == f"{long_name}_peer_card"
    )

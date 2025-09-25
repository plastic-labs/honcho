import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.representation import (
    construct_collection_name,
    construct_peer_card_label,
    get_peer_card,
    set_peer_card,
)
from src.exceptions import ResourceNotFoundException
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    PromptDeductiveObservation,
    PromptRepresentation,
    Representation,
)


@pytest.mark.asyncio
async def test_peer_card_get_set_roundtrip(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Roundtrip set/get for peer card on a valid peer."""
    workspace, peer = sample_data

    # Initially absent
    assert await get_peer_card(db_session, workspace.name, peer.name, peer.name) is None

    # Set and read back
    value_1 = ["Initial peer card text"]
    await set_peer_card(db_session, workspace.name, peer.name, peer.name, value_1)
    assert (
        await get_peer_card(db_session, workspace.name, peer.name, peer.name) == value_1
    )

    # Update and read back
    value_2 = ["Updated peer card text", "Another line"]
    await set_peer_card(db_session, workspace.name, peer.name, peer.name, value_2)
    assert (
        await get_peer_card(db_session, workspace.name, peer.name, peer.name) == value_2
    )


@pytest.mark.asyncio
async def test_set_peer_card_missing_peer_raises(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Setting a peer card for a non-existent peer should raise ResourceNotFoundException."""
    workspace, _existing_peer = sample_data
    with pytest.raises(ResourceNotFoundException):
        await set_peer_card(
            db_session, workspace.name, "missing-peer", "missing-peer", ["card"]
        )


def test_construct_helpers_labels():
    """Helper label constructors return expected values."""
    assert construct_peer_card_label(observer="a", observed="a") == "peer_card"
    assert construct_peer_card_label(observer="a", observed="b") == "b_peer_card"
    assert construct_collection_name(observer="obs", observed="obj") == "obs_obj"


def test_representation_is_empty_and_diff():
    """is_empty and diff_representation behave per the new definitions."""
    now = datetime.datetime.now(datetime.timezone.utc)
    shared_time = now - datetime.timedelta(seconds=10)
    exp_shared_1 = ExplicitObservation(
        content="A",
        created_at=shared_time,
        message_id=1,
        session_name="s",
    )
    exp_shared_2 = ExplicitObservation(
        content="B",
        created_at=shared_time,
        message_id=1,
        session_name="s",
    )
    rep1 = Representation(explicit=[exp_shared_1], deductive=[])
    rep2 = Representation(
        explicit=[
            ExplicitObservation(
                content="A",
                created_at=shared_time,
                message_id=1,
                session_name="s",
            ),
            exp_shared_2,
        ]
    )

    assert not rep1.is_empty()
    assert Representation().is_empty()

    diff = rep1.diff_representation(rep2)
    assert [e.content for e in diff.explicit] == ["B"]
    assert diff.deductive == []


def test_representation_formatting_methods():
    """__str__ and format_as_markdown produce expected section headers and content."""
    now = datetime.datetime.now(datetime.timezone.utc)
    e = ExplicitObservation(
        content="has a dog",
        created_at=now,
        message_id=1,
        session_name="s",
    )
    d = DeductiveObservation(
        created_at=now,
        message_id=1,
        session_name="s",
        conclusion="owns a pet",
        premises=[e.content],
    )
    rep = Representation(explicit=[e], deductive=[d])

    s = str(rep)
    assert "EXPLICIT:" in s
    assert "DEDUCTIVE:" in s
    assert "owns a pet" in s

    md = rep.format_as_markdown()
    assert "## Explicit Observations" in md
    assert "## Deductive Observations" in md
    assert "**Conclusion**: owns a pet" in md


def test_prompt_representation_conversion():
    """PromptRepresentation.to_representation maps strings to observation objects."""
    pr = PromptRepresentation(
        explicit=["A"],
        deductive=[PromptDeductiveObservation(conclusion="C", premises=["P1"])],
    )
    rep = pr.to_representation(message_id=1, session_name="s")
    assert isinstance(rep, Representation)
    assert [e.content for e in rep.explicit] == ["A"]
    assert rep.deductive[0].conclusion == "C"
    assert rep.deductive[0].premises == ["P1"]

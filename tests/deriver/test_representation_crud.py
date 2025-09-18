import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.crud.representation import (
    construct_collection_name,
    construct_peer_card_label,
    get_peer_card,
    get_working_representation,
    set_peer_card,
    set_working_representation,
)
from src.deriver.queue_payload import RepresentationPayload
from src.exceptions import ResourceNotFoundException
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    PromptDeductiveObservation,
    PromptRepresentation,
    Representation,
    StoredRepresentation,
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


async def _create_session_with_peers(
    db_session: AsyncSession, workspace: models.Workspace
) -> tuple[models.Session, models.Peer, models.Peer]:
    """Create a session with two peers and return (session, observer, observed)."""
    observer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([observer, observed])
    await db_session.flush()

    session_name = str(generate_nanoid())
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=session_name,
            peers={
                observer.name: schemas.SessionPeerConfig(),
                observed.name: schemas.SessionPeerConfig(),
            },
        ),
        workspace.name,
    )
    await db_session.commit()
    return session, observer, observed


def _make_representation(*, session_name: str) -> Representation:
    """Create a small Representation with two explicit and one deductive observation."""
    now = datetime.datetime.now(datetime.timezone.utc)
    e1 = ExplicitObservation(
        content="likes pizza",
        created_at=now - datetime.timedelta(seconds=5),
        message_id="m1",
        session_name=session_name,
    )
    e2 = ExplicitObservation(
        content="runs daily",
        created_at=now - datetime.timedelta(seconds=4),
        message_id="m2",
        session_name=session_name,
    )
    d1 = DeductiveObservation(
        created_at=now - datetime.timedelta(seconds=3),
        message_id="m3",
        session_name=session_name,
        conclusion="is healthy",
        premises=[e2],
    )
    return Representation(explicit=[e1, e2], deductive=[d1])


@pytest.mark.asyncio
async def test_working_representation_self_roundtrip(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Initial set/get roundtrip for a self working representation using new models."""
    workspace, peer = sample_data

    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    representation = _make_representation(session_name=session.name)
    payload = RepresentationPayload(
        task_type="representation",
        workspace_name=workspace.name,
        session_name=session.name,
        message_id=1,
        content="hello",
        sender_name=peer.name,
        target_name=peer.name,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    await set_working_representation(db_session, representation, payload)

    stored = await get_working_representation(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(stored, StoredRepresentation)
    assert len(stored.explicit) == 2
    assert len(stored.deductive) == 1
    assert any(e.content == "likes pizza" for e in stored.explicit)
    assert any(e.content == "runs daily" for e in stored.explicit)
    assert stored.deductive[0].conclusion == "is healthy"

    formatted = str(stored)
    assert "EXPLICIT:" in formatted
    assert "DEDUCTIVE:" in formatted
    assert "is healthy" in formatted


@pytest.mark.asyncio
async def test_working_representation_directional_storage_key(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Directional working representations are stored under observer_observed key."""
    workspace, _ = sample_data
    session, observer, observed = await _create_session_with_peers(
        db_session, workspace
    )

    representation = _make_representation(session_name=session.name)
    payload = RepresentationPayload(
        task_type="representation",
        workspace_name=workspace.name,
        session_name=session.name,
        message_id=2,
        content="hello",
        sender_name=observed.name,
        target_name=observer.name,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    await set_working_representation(db_session, representation, payload)

    derived_key = construct_collection_name(
        observer=observer.name, observed=observed.name
    )
    stmt = select(models.SessionPeer).where(
        models.SessionPeer.peer_name == observer.name,
        models.SessionPeer.session_name == session.name,
        models.SessionPeer.workspace_name == workspace.name,
    )
    result = await db_session.execute(stmt)
    sp = result.scalar_one()
    assert derived_key in sp.internal_metadata


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
        message_id="m",
        session_name="s",
    )
    exp_shared_2 = ExplicitObservation(
        content="B",
        created_at=shared_time,
        message_id="m",
        session_name="s",
    )
    rep1 = Representation(explicit=[exp_shared_1], deductive=[])
    rep2 = Representation(
        explicit=[
            ExplicitObservation(
                content="A",
                created_at=shared_time,
                message_id="m",
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
        message_id="m",
        session_name="s",
    )
    d = DeductiveObservation(
        created_at=now,
        message_id="m",
        session_name="s",
        conclusion="owns a pet",
        premises=[e],
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
    rep = pr.to_representation()
    assert isinstance(rep, Representation)
    assert [e.content for e in rep.explicit] == ["A"]
    assert rep.deductive[0].conclusion == "C"
    assert [p.content for p in rep.deductive[0].premises] == ["P1"]


def test_stored_representation_add_and_trim():
    """add_single_observation appends and trims per max_observations for both kinds."""
    now = datetime.datetime.now(datetime.timezone.utc)
    sr = StoredRepresentation(
        created_at=now,
        message_id="m",
        explicit=[],
        deductive=[],
        max_observations=2,
    )

    # Add three explicit, expect only last two kept
    for i in range(3):
        sr.add_single_observation(
            ExplicitObservation(
                content=f"E{i}",
                created_at=now + datetime.timedelta(seconds=i),
                message_id=f"me{i}",
                session_name="s",
            )
        )
    assert [e.content for e in sr.explicit] == ["E1", "E2"]

    # Add three deductive, expect only last two kept
    for i in range(3):
        sr.add_single_observation(
            DeductiveObservation(
                created_at=now + datetime.timedelta(seconds=10 + i),
                message_id=f"md{i}",
                session_name="s",
                conclusion=f"D{i}",
                premises=[],
            )
        )
    assert [d.conclusion for d in sr.deductive] == ["D1", "D2"]


def test_observation_strs():
    """ExplicitObservation and DeductiveObservation string formatting include timestamps and content."""
    when = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    e = ExplicitObservation(
        content="alpha",
        created_at=when,
        message_id="m",
        session_name="s",
    )
    d = DeductiveObservation(
        created_at=when,
        message_id="m",
        session_name="s",
        conclusion="beta",
        premises=[e],
    )
    assert str(e).startswith("[2025-01-01") and "alpha" in str(e)
    s = str(d)
    assert "beta" in s and "- [2025-01-01" in s

import datetime
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.representation import (
    get_peer_card,
    get_working_representation,
    get_working_representation_data,
    set_peer_card,
    set_working_representation,
)
from src.exceptions import ResourceNotFoundException


@pytest.mark.asyncio
async def test_peer_card_get_set_roundtrip(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Roundtrip set/get for peer card on a valid peer."""
    workspace, peer = sample_data

    # Initially absent
    assert await get_peer_card(db_session, workspace.name, peer.name) is None

    # Set and read back
    value_1 = "Initial peer card text"
    await set_peer_card(db_session, workspace.name, peer.name, value_1)
    assert await get_peer_card(db_session, workspace.name, peer.name) == value_1

    # Update and read back
    value_2 = "Updated peer card text"
    await set_peer_card(db_session, workspace.name, peer.name, value_2)
    assert await get_peer_card(db_session, workspace.name, peer.name) == value_2


@pytest.mark.asyncio
async def test_set_peer_card_missing_peer_raises(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Setting a peer card for a non-existent peer should raise ResourceNotFoundException."""
    workspace, _existing_peer = sample_data
    with pytest.raises(ResourceNotFoundException):
        await set_peer_card(db_session, workspace.name, "missing-peer", "card")


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


def _make_wr_payload(
    *,
    explicit: list[str],
    deductive: list[dict[str, Any]] | None = None,
    thinking: str | None = None,
    message_id: str = "m-new",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a structured working representation dict payload."""
    return {
        "final_observations": {
            "explicit": explicit,
            "deductive": deductive or [],
            "thinking": thinking,
        },
        "message_id": message_id,
        "created_at": created_at
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@pytest.mark.asyncio
async def test_working_representation_self_merge_and_trim(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Merging a self working representation appends and trims to 25, and updates metadata fields."""
    workspace, peer = sample_data

    # Align limit with current implementation and future config usage
    monkeypatch.setattr(
        settings.DERIVER, "WORKING_REPRESENTATION_MAX_OBSERVATIONS", 25, raising=False
    )
    LIMIT = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS

    # Create a session with the single peer
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    # Existing explicit: 24 items 0..23
    existing = _make_wr_payload(
        explicit=[f"E{i}" for i in range(24)],
        deductive=[{"conclusion": f"D{i}", "premises": [f"P{i}"]} for i in range(10)],
        thinking="old-think",
        message_id="m-old",
        created_at=datetime.datetime(
            2024, 1, 1, tzinfo=datetime.timezone.utc
        ).isoformat(),
    )
    await set_working_representation(
        db_session, existing, workspace.name, peer.name, peer.name, session.name
    )

    # New explicit: 3 items 24..26, deductive two new items
    new_payload = _make_wr_payload(
        explicit=["E24", "E25", "E26"],
        deductive=[
            {"conclusion": "D_new1", "premises": []},
            {"conclusion": "D_new2", "premises": ["PX"]},
        ],
        thinking="new-think",
        message_id="m-new",
        created_at=datetime.datetime(
            2025, 1, 1, tzinfo=datetime.timezone.utc
        ).isoformat(),
    )
    await set_working_representation(
        db_session, new_payload, workspace.name, peer.name, peer.name, session.name
    )

    # Verify merged raw data
    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(raw, dict)
    final = raw["final_observations"]

    # Explicit should be last LIMIT of 24+3 (drop oldest overflow)
    explicit = final["explicit"]
    total_explicit = 24 + 3
    expected_len = min(LIMIT, total_explicit)
    dropped = max(0, total_explicit - LIMIT)
    assert len(explicit) == expected_len
    assert explicit[0] == f"E{dropped}"
    assert explicit[-1] == "E26"

    # Deductive should be appended and capped to LIMIT
    deductive = final["deductive"]
    assert len(deductive) == min(LIMIT, 12)
    assert deductive[-2]["conclusion"] == "D_new1"
    assert deductive[-1]["conclusion"] == "D_new2"

    # Thinking and metadata should reflect latest
    assert final.get("thinking") == "new-think"
    assert raw.get("message_id") == "m-new"
    created_at_value = raw.get("created_at")
    assert created_at_value is not None
    assert created_at_value.startswith("2025-01-01")

    # Formatted string getter returns sections and bullets
    formatted = await get_working_representation(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert "EXPLICIT OBSERVATIONS:" in formatted
    assert "DEDUCTIVE OBSERVATIONS:" in formatted
    assert "- E26" in formatted


@pytest.mark.asyncio
async def test_working_representation_directional_merge_and_keys(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Directional working representations are stored under observer_observed key and merge correctly."""
    workspace, _ = sample_data
    session, observer, observed = await _create_session_with_peers(
        db_session, workspace
    )

    # Store initial
    first = _make_wr_payload(explicit=["A", "B"], thinking="first")
    await set_working_representation(
        db_session, first, workspace.name, observer.name, observed.name, session.name
    )

    # Merge second
    second = _make_wr_payload(
        explicit=["C"],
        deductive=[{"conclusion": "Z", "premises": ["p1", "p2"]}],
        thinking="second",
    )
    await set_working_representation(
        db_session, second, workspace.name, observer.name, observed.name, session.name
    )

    # Fetch raw and assert structure
    raw = await get_working_representation_data(
        db_session, workspace.name, observer.name, observed.name, session.name
    )
    assert isinstance(raw, dict)
    final = raw["final_observations"]
    assert final["explicit"] == ["A", "B", "C"]
    assert final["deductive"][-1]["conclusion"] == "Z"
    assert final.get("thinking") == "second"

    # Validate it's stored under the derived key in SessionPeer.internal_metadata
    derived_key = f"{observer.name}_{observed.name}"
    stmt = select(models.SessionPeer).where(
        models.SessionPeer.peer_name == observer.name,
        models.SessionPeer.session_name == session.name,
        models.SessionPeer.workspace_name == workspace.name,
    )
    result = await db_session.execute(stmt)
    sp = result.scalar_one()
    assert derived_key in sp.internal_metadata
    assert sp.internal_metadata[derived_key]["final_observations"]["explicit"] == [
        "A",
        "B",
        "C",
    ]


@pytest.mark.asyncio
async def test_wr_string_roundtrip_then_structured_overrides(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """String WR stores as-is and later structured merge does not incorporate old string content."""
    workspace, peer = sample_data
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    # Store legacy string representation first
    raw_string = "legacy working rep text"
    await set_working_representation(
        db_session, raw_string, workspace.name, peer.name, peer.name, session.name
    )

    # Ensure get returns the same string
    assert (
        await get_working_representation(
            db_session, workspace.name, peer.name, peer.name, session.name
        )
        == raw_string
    )

    # Now store structured representation; merge should ignore old string and just store new structured
    structured = _make_wr_payload(explicit=["X", "Y"], deductive=[])
    await set_working_representation(
        db_session, structured, workspace.name, peer.name, peer.name, session.name
    )

    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(raw, dict)
    assert raw["final_observations"]["explicit"] == ["X", "Y"]


@pytest.mark.asyncio
async def test_wr_missing_levels_and_empty_new_lists(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Merge handles missing levels and empty new lists; result formatting is empty string."""
    workspace, peer = sample_data
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    # Existing has only deductive; explicit missing
    existing = {
        "final_observations": {
            "deductive": [{"conclusion": "old", "premises": ["p"]}],
        },
        "message_id": "m-old",
        "created_at": datetime.datetime(
            2024, 1, 2, tzinfo=datetime.timezone.utc
        ).isoformat(),
    }
    await set_working_representation(
        db_session, existing, workspace.name, peer.name, peer.name, session.name
    )

    # New has empty lists and no thinking
    new_payload = _make_wr_payload(
        explicit=[], deductive=[], thinking=None, message_id="m-new"
    )
    await set_working_representation(
        db_session, new_payload, workspace.name, peer.name, peer.name, session.name
    )

    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(raw, dict)
    final = raw["final_observations"]
    # Deductive remains as old (no additions), explicit remains missing/empty and thinking None
    assert final["deductive"] == [{"conclusion": "old", "premises": ["p"]}]
    assert final["explicit"] == []
    assert final.get("thinking") is None
    # Formatter includes the remaining deductive observation
    formatted = await get_working_representation(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert "DEDUCTIVE OBSERVATIONS:" in formatted
    assert "- old (based on: p)" in formatted


@pytest.mark.asyncio
async def test_wr_trim_boundary_exact_25_plus_one(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Exactly 25 existing + 1 new yields last 25, preserving order and including the new last element."""
    workspace, peer = sample_data
    monkeypatch.setattr(
        settings.DERIVER, "WORKING_REPRESENTATION_MAX_OBSERVATIONS", 25, raising=False
    )
    LIMIT = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    existing = _make_wr_payload(
        explicit=[f"E{i}" for i in range(LIMIT)],
        deductive=[{"conclusion": f"D{i}", "premises": []} for i in range(LIMIT)],
    )
    await set_working_representation(
        db_session, existing, workspace.name, peer.name, peer.name, session.name
    )

    new_payload = _make_wr_payload(
        explicit=[f"E{LIMIT}"],
        deductive=[{"conclusion": f"D{LIMIT}", "premises": []}],
        thinking="t2",
        message_id="m2",
    )
    await set_working_representation(
        db_session, new_payload, workspace.name, peer.name, peer.name, session.name
    )

    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(raw, dict)
    final = raw["final_observations"]
    assert final["explicit"] == [f"E{i}" for i in range(1, LIMIT + 1)]
    assert final["explicit"][-1] == f"E{LIMIT}"
    assert len(final["deductive"]) == LIMIT
    assert final["deductive"][-1]["conclusion"] == f"D{LIMIT}"


@pytest.mark.asyncio
async def test_wr_formatting_rules_mixed_observation_types(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Formatting includes section headers, bullets, premise display, and content fallback."""
    workspace, peer = sample_data
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    payload = {
        "final_observations": {
            "explicit": ["likes pizza", {"content": "runs daily"}],
            "deductive": [
                {
                    "conclusion": "is healthy",
                    "premises": ["runs daily", "eats veggies"],
                },
                {"content": "fallback without conclusion"},
            ],
            "thinking": "t",
        },
        "message_id": "m1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    await set_working_representation(
        db_session, payload, workspace.name, peer.name, peer.name, session.name
    )

    formatted = await get_working_representation(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    # Section headers present
    assert formatted.splitlines()[0] == "EXPLICIT OBSERVATIONS:"
    assert "DEDUCTIVE OBSERVATIONS:" in formatted
    # Bullets present for strings and dict content fallback
    assert "- likes pizza" in formatted
    assert "- runs daily" in formatted
    # Deductive with premises shows based on
    assert "is healthy (based on: runs daily; eats veggies)" in formatted
    # Deductive dict without conclusion falls back to content
    assert "- fallback without conclusion" in formatted


@pytest.mark.asyncio
async def test_wr_legacy_key_fallback_for_self_representation(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """If only the legacy key is present, data retrieval falls back appropriately."""
    workspace, peer = sample_data
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    # Manually write legacy key on SessionPeer
    stmt = select(models.SessionPeer).where(
        models.SessionPeer.peer_name == peer.name,
        models.SessionPeer.session_name == session.name,
        models.SessionPeer.workspace_name == workspace.name,
    )
    result = await db_session.execute(stmt)
    sp: models.SessionPeer = result.scalar_one()
    # Assign a new dict so JSON mutation is tracked & persisted
    sp.internal_metadata = {
        **(sp.internal_metadata or {}),
        "global_representation": "legacy-global",
    }
    await db_session.commit()

    # Retrieval should see legacy value
    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert raw == "legacy-global"


@pytest.mark.asyncio
async def test_wr_empty_both_levels_formats_empty_string(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """When both explicit and deductive are empty after merge, formatted string is empty."""
    workspace, peer = sample_data
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    empty_payload = _make_wr_payload(explicit=[], deductive=[], thinking=None)
    await set_working_representation(
        db_session, empty_payload, workspace.name, peer.name, peer.name, session.name
    )

    formatted = await get_working_representation(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert formatted == ""


@pytest.mark.asyncio
async def test_wr_existing_dict_without_final_observations(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    """Existing dict lacking final_observations merges cleanly with new structured payload."""
    workspace, peer = sample_data
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    await set_working_representation(
        db_session,
        {"some": "field"},
        workspace.name,
        peer.name,
        peer.name,
        session.name,
    )
    new_payload = _make_wr_payload(
        explicit=["n1"], deductive=[{"conclusion": "c1", "premises": []}]
    )
    await set_working_representation(
        db_session, new_payload, workspace.name, peer.name, peer.name, session.name
    )

    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(raw, dict)
    final = raw["final_observations"]
    assert final["explicit"] == ["n1"]
    assert final["deductive"][0]["conclusion"] == "c1"


@pytest.mark.asyncio
async def test_wr_deductive_trim_when_no_new_items(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """If existing deductive > 25 and no new entries, merge still trims to last 25."""
    workspace, peer = sample_data
    monkeypatch.setattr(
        settings.DERIVER, "WORKING_REPRESENTATION_MAX_OBSERVATIONS", 25, raising=False
    )
    LIMIT = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()), peers={peer.name: schemas.SessionPeerConfig()}
        ),
        workspace.name,
    )
    await db_session.commit()

    over = LIMIT + 5
    existing = _make_wr_payload(
        explicit=[],
        deductive=[{"conclusion": f"D{i}", "premises": []} for i in range(over)],
    )
    await set_working_representation(
        db_session, existing, workspace.name, peer.name, peer.name, session.name
    )

    # Merge empty new; should trim to last 25 existing
    new_payload = _make_wr_payload(explicit=[], deductive=[])
    await set_working_representation(
        db_session, new_payload, workspace.name, peer.name, peer.name, session.name
    )

    raw = await get_working_representation_data(
        db_session, workspace.name, peer.name, peer.name, session.name
    )
    assert isinstance(raw, dict)
    final = raw["final_observations"]
    assert len(final["deductive"]) == LIMIT
    # Oldest retained index is over - LIMIT
    oldest_kept = over - LIMIT
    assert final["deductive"][0]["conclusion"] == f"D{oldest_kept}"
    assert final["deductive"][-1]["conclusion"] == f"D{over - 1}"

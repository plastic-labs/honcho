"""How `get_context` decides whether to serve a session summary.

Two paths, and which one runs depends on `peer_target`:

- Without it, `summarizer.get_session_context` gives the summary 40% of the
  requested `tokens`.
- With it, `sessions._select_summary_for_context` gives it the same 40%, and
  the peer card and representation are fitted into what remains, so an
  observer with many observations can no longer starve a valid summary.

Either way a summary that does not fit leaves the caller with `summary: null`,
indistinguishable from a session that has none, which is why both paths log
when they drop one.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.models import Peer, Workspace
from src.routers.sessions import (
    _fit_representation_to_budget,  # pyright: ignore[reportPrivateUsage]
    _select_summary_for_context,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)
from src.utils.summarizer import (
    Summary,
    SummaryType,
    _save_summary,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.tokens import estimate_tokens
from tests.conftest import _content_to_embedding  # pyright: ignore[reportPrivateUsage]

# Measured from CI run 33779689337: 12 messages from one peer produce 12
# explicit observations costing ~1176 tokens. Only the `peer_target` path pays
# this, and the unified `config_summary` fixtures do not take that path.
_FIXTURE_REPRESENTATION_TOKENS = 1176
_SHORT_SUMMARY_CAP = 1000  # SUMMARY.MAX_TOKENS_SHORT default


def _summary_schema(token_count: int) -> schemas.Summary:
    return schemas.Summary(
        content="A summary of the conversation so far.",
        message_id=1,
        summary_type="short",
        created_at=dt.datetime.now(dt.UTC).isoformat(),
        token_count=token_count,
        message_public_id="msg_public",
    )


def _stored_summary(token_count: int) -> Summary:
    return Summary(
        content="A summary of the conversation so far. " * 5,
        message_id=1,
        summary_type=SummaryType.SHORT.value,
        created_at=dt.datetime.now(dt.UTC).isoformat(),
        token_count=token_count,
        message_public_id="msg_public",
    )


def test_representation_can_exhaust_the_budget_entirely() -> None:
    """A large representation can leave a negative budget on the observer path."""
    adjusted = 400 - _FIXTURE_REPRESENTATION_TOKENS
    assert adjusted < 0
    chosen, _, _ = _select_summary_for_context(
        _summary_schema(99), None, adjusted, True
    )
    assert chosen is None


def test_a_conforming_summary_can_still_be_dropped() -> None:
    """With that representation, 2500 leaves 529 — under `SUMMARY.MAX_TOKENS_SHORT`."""
    adjusted = 2500 - _FIXTURE_REPRESENTATION_TOKENS
    chosen, _, _ = _select_summary_for_context(
        _summary_schema(_SHORT_SUMMARY_CAP), None, adjusted, True
    )
    assert chosen is None


def test_fixture_limit_fits_any_conforming_summary() -> None:
    """4000 leaves room even when a representation is subtracted."""
    adjusted = 4000 - _FIXTURE_REPRESENTATION_TOKENS
    assert int(adjusted * 0.4) >= _SHORT_SUMMARY_CAP
    chosen, _, _ = _select_summary_for_context(
        _summary_schema(_SHORT_SUMMARY_CAP), None, adjusted, True
    )
    assert chosen is not None


def test_zero_token_summary_is_never_served() -> None:
    chosen, _, _ = _select_summary_for_context(_summary_schema(0), None, 4000, True)
    assert chosen is None


def test_dropped_summary_is_logged_not_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`summary: null` is indistinguishable from 'no summary exists' otherwise."""
    with caplog.at_level("INFO", logger="src.routers.sessions"):
        _select_summary_for_context(_summary_schema(900), None, 1000, True)
    assert "Summary dropped" in caplog.text


def test_no_log_when_the_session_simply_has_no_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="src.routers.sessions"):
        _select_summary_for_context(None, None, 1000, True)
    assert "Summary dropped" not in caplog.text


@pytest.mark.parametrize("with_observer", [False, True])
async def test_a_stored_summary_is_served(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
    with_observer: bool,
) -> None:
    """Retrieval itself works: a saved summary comes back through the route."""
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _save_summary(db_session, _stored_summary(99), workspace.name, session_id)
    await db_session.commit()

    url = (
        f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
        "?summary=true&tokens=4000"
    )
    if with_observer:
        url += f"&peer_target={peer.name}"

    data = client.get(url).json()
    assert data["summary"] is not None
    assert data["summary"]["token_count"] == 99


async def test_fixture_path_ignores_representation_budget(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    """Without `peer_target`, the summary gets 40% of `tokens` outright.

    The unified `config_summary` fixtures set `observer_peer_id`, but the runner
    does not forward it to `get_context`, so this is the path they exercise.
    """
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _save_summary(db_session, _stored_summary(99), workspace.name, session_id)
    await db_session.commit()

    url = f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
    data = client.get(f"{url}?summary=true&tokens=2500").json()

    assert data["summary"] is not None
    assert data.get("peer_representation") is None


async def _seed_observations(
    db_session: AsyncSession,
    workspace: Workspace,
    peer: Peer,
    session_id: str,
    count: int,
) -> None:
    """Store `count` explicit observations of `peer`, oldest first."""
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    db_session.add_all(
        [
            models.Document(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                session_name=session_id,
                content=(
                    f"Observation number {i}: the peer described a long and "
                    "detailed preference about their tooling and daily schedule"
                ),
                created_at=base + dt.timedelta(minutes=i),
            )
            for i in range(count)
        ]
    )
    await db_session.commit()


def _context_tokens(data: dict[str, Any]) -> int:
    """Tokens of everything `get_context` returned, measured as served."""
    total = estimate_tokens(data.get("peer_representation"))
    total += estimate_tokens(data.get("peer_card"))
    if data.get("summary"):
        total += data["summary"]["token_count"]
    return total + sum(m["token_count"] for m in data["messages"])


async def test_tokens_bounds_the_representation_and_keeps_the_summary(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    """`tokens` covers the representation too, so it cannot starve the summary.

    60 observations render to well over 1000 tokens. The whole response must
    still fit in `tokens=1000`, the summary must keep its 40% share, and the
    representation keeps the newest observations rather than the oldest.
    """
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _seed_observations(db_session, workspace, peer, session_id, 60)
    await _save_summary(db_session, _stored_summary(99), workspace.name, session_id)
    await db_session.commit()

    url = f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
    unbounded = client.get(f"{url}?peer_target={peer.name}").json()
    assert estimate_tokens(unbounded["peer_representation"]) > 1000

    data = client.get(f"{url}?peer_target={peer.name}&tokens=1000").json()

    assert _context_tokens(data) <= 1000
    assert data["summary"] is not None
    representation = data["peer_representation"]
    assert "Observation number 59:" in representation
    assert "Observation number 0:" not in representation


async def test_representation_within_budget_is_served_whole(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    """A representation that already fits is not trimmed."""
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    await _seed_observations(db_session, workspace, peer, session_id, 5)
    await _save_summary(db_session, _stored_summary(99), workspace.name, session_id)
    await db_session.commit()

    url = f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
    data = client.get(f"{url}?peer_target={peer.name}&tokens=4000").json()

    assert data["summary"] is not None
    for i in range(5):
        assert f"Observation number {i}:" in data["peer_representation"]


async def test_tight_budget_keeps_the_search_match_over_recent_filler(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    """With `search_query`, trimming drops recent filler before the match.

    The one semantic match is older than ten unrelated observations that only
    fill the rest of the representation. A budget that fits a couple of
    observations must still serve the match, or `search_query` has no effect.
    """
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    match = "QUERY MATCH: the peer's favorite color is green"
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    contents = [match] + [
        (
            f"Unrelated observation {i}: the peer mentioned a long and detailed "
            "preference about their tooling and daily schedule"
        )
        for i in range(10)
    ]
    db_session.add_all(
        [
            models.Document(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                session_name=session_id,
                content=content,
                level="explicit",
                embedding=_content_to_embedding(content),
                created_at=base + dt.timedelta(minutes=i),
            )
            for i, content in enumerate(contents)
        ]
    )
    await db_session.commit()

    url = f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
    params = {"peer_target": peer.name, "search_query": match, "search_top_k": 1}
    unbounded = client.get(url, params=params).json()["peer_representation"]
    assert match in unbounded
    assert "Unrelated observation 9:" in unbounded

    data = client.get(url, params={**params, "tokens": 60}).json()

    assert _context_tokens(data) <= 60
    representation = data["peer_representation"]
    assert match in representation
    assert "Unrelated observation 0:" not in representation


async def test_peer_card_is_served_when_it_fits_and_dropped_when_not(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    db_session: AsyncSession,
) -> None:
    """The peer card counts toward `tokens` and is omitted, not overrun."""
    workspace, peer = sample_data
    session_id = str(generate_nanoid())
    client.post(
        f"/v3/workspaces/{workspace.name}/sessions",
        json={"id": session_id, "peers": {peer.name: {}}},
    )
    card = [f"Card fact {i}: the peer works on distributed systems" for i in range(8)]
    await crud.set_peer_card(
        db_session, workspace.name, card, observer=peer.name, observed=peer.name
    )
    await db_session.commit()
    card_tokens = estimate_tokens(card)

    url = f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context"
    fits = client.get(f"{url}?peer_target={peer.name}&tokens={card_tokens}").json()
    assert fits["peer_card"] == card
    assert _context_tokens(fits) <= card_tokens

    tight = client.get(f"{url}?peer_target={peer.name}&tokens={card_tokens - 1}")
    data = tight.json()
    assert data["peer_card"] is None
    assert _context_tokens(data) <= card_tokens - 1


def _mixed_representation() -> Representation:
    """Explicit and deductive observations interleaved in time."""
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    return Representation(
        explicit=[
            ExplicitObservation(
                id=f"e{i}",
                content=f"explicit {i}",
                created_at=base + dt.timedelta(minutes=2 * i),
                message_ids=[i],
            )
            for i in range(3)
        ],
        deductive=[
            DeductiveObservation(
                id=f"d{i}",
                conclusion=f"deductive {i}",
                premises=[f"explicit {i}"],
                created_at=base + dt.timedelta(minutes=2 * i + 1),
                message_ids=[i],
            )
            for i in range(3)
        ],
    )


def test_keep_top_counts_across_levels() -> None:
    kept = _mixed_representation().keep_top(3)
    assert [o.content for o in kept.explicit] == ["explicit 2"]
    assert [o.conclusion for o in kept.deductive] == ["deductive 1", "deductive 2"]


def test_keep_top_puts_ranked_ids_ahead_of_recency() -> None:
    representation = _mixed_representation()
    kept = representation.keep_top(2, ranked_ids=["e0"])
    assert [o.content for o in kept.explicit] == ["explicit 0"]
    assert [o.conclusion for o in kept.deductive] == ["deductive 2"]


def test_fit_representation_keeps_the_newest_that_fit() -> None:
    representation = _mixed_representation()
    full = estimate_tokens(representation.format_as_markdown())

    assert _fit_representation_to_budget(representation, full) is representation

    trimmed = _fit_representation_to_budget(representation, full - 1)
    assert 0 < trimmed.len() < representation.len()
    assert estimate_tokens(trimmed.format_as_markdown()) <= full - 1
    assert trimmed.deductive[-1].conclusion == "deductive 2"

    assert _fit_representation_to_budget(representation, 0).is_empty()

    ranked = _fit_representation_to_budget(representation, full - 1, ["e0"])
    assert ranked.explicit[0].content == "explicit 0"

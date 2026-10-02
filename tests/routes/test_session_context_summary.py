"""How `get_context` decides whether to serve a session summary.

Two paths, and which one runs depends on `peer_target`:

- Without it, `summarizer.get_session_context` gives the summary 40% of the
  requested `tokens`.
- With it, `sessions._select_summary_for_context` gives it 40% of what remains
  *after* the peer representation and card are subtracted, so an observer with
  many observations can starve a perfectly valid summary.

Either way the caller just sees `summary: null`, indistinguishable from a
session that has none, which is why both paths now log when they drop one.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.models import Peer, Workspace
from src.routers.sessions import (
    _select_summary_for_context,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.summarizer import (
    Summary,
    SummaryType,
    _save_summary,  # pyright: ignore[reportPrivateUsage]
)

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

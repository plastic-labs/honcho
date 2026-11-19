"""Suggested links: refresh, review, accept (a link) or dismiss, across a reset."""

import datetime
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.organizer.run import refresh_suggestions

ALEX = {"id": "u1", "name": "Alex", "handle": "alex_d"}
SAM = {"id": "u2", "name": "Sam", "handle": "samwise"}
ALEX_GITHUB = {"id": "g1", "name": "alexdev", "handle": "alexdev"}
# Shares Sam's Discord handle on GitHub, but nothing else says it's Sam.
SAM_GITHUB = {"id": "g2", "name": "samwise", "handle": "samwise"}


def _workspace(client: TestClient, *, auto: bool = True) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name, "auto": auto})
    assert response.status_code in (200, 201), response.text
    return name


def _record(
    n: int, author: dict[str, Any], content: str, source: str = "discord"
) -> dict[str, Any]:
    return {
        "source": source,
        "source_instance": "acme",
        "source_id": f"{source}-{n}",
        "created_at": f"2026-11-04T10:{n:02d}:00Z",
        "container": {"id": f"{source}-chan", "name": f"{source} channel"},
        "author": author,
        "content": content,
    }


RECORDS = [
    _record(1, ALEX, "Morning! I'm alexdev on GitHub if you need reviews."),
    _record(2, SAM, "Pricing copy is done."),
    _record(3, ALEX_GITHUB, "PR: fix the onboarding link", "github"),
    _record(4, SAM_GITHUB, "PR: bump the parser", "github"),
]


def _import(client: TestClient, workspace: str) -> None:
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"derive": "later", "records": RECORDS},
    )
    assert response.status_code == 200, response.text


def _suggestions(client: TestClient, workspace: str, **params: str) -> list[Any]:
    response = client.get(
        f"/v3/workspaces/{workspace}/ingest/suggestions", params=params
    )
    assert response.status_code == 200, response.text
    return response.json()["items"]


async def _refresh(client: TestClient, workspace: str) -> None:
    """Queue a refresh, then run it as the deriver would."""
    response = client.post(f"/v3/workspaces/{workspace}/ingest/suggestions/refresh")
    assert response.status_code == 202, response.text
    await refresh_suggestions(workspace)


async def _reset(client: TestClient, db: AsyncSession, workspace: str) -> None:
    response = client.post(f"/v3/workspaces/{workspace}/ingest/reset")
    assert response.status_code == 202, response.text
    for name in response.json()["sessions"]:
        await crud.delete_session(db, workspace, name)
    await db.commit()


@pytest.mark.asyncio
async def test_a_suggestion_accepted_after_a_reset_links_the_accounts(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    await _refresh(client, workspace)

    [high] = _suggestions(client, workspace, confidence="high")
    assert {(a["source"], a["handle"]) for a in high["accounts"]} == {
        ("discord", "alex_d"),
        ("github", "alexdev"),
    }
    assert high["evidence"][0]["signal"] == "self_reference"
    assert high["linked"] is False
    # A handle shared across platforms is only worth a look.
    [low] = _suggestions(client, workspace, confidence="low")
    assert {a["handle"] for a in low["accounts"]} == {"samwise"}
    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()
    assert status["suggestions"]["high"] == 1
    assert status["suggestions"]["low"] == 1

    # Accepting is linking: refused while the people hold imported messages.
    url = f"/v3/workspaces/{workspace}/ingest/suggestions/{high['id']}/accept"
    refused = client.post(url)
    assert refused.status_code == 409, refused.text
    assert "reset" in refused.json()["detail"].lower()

    await _reset(client, db_session, workspace)
    accepted = client.post(url)
    assert accepted.status_code == 200, accepted.text
    body = accepted.json()
    assert body["suggestion"]["status"] == "accepted"
    assert body["suggestion"]["linked"] is True
    peers = {a["peer_id"] for a in high["accounts"]}
    kept = body["link"]["peer_id"]
    assert kept in peers
    assert body["link"]["merged"] == sorted(peers - {kept})

    _import(client, workspace)
    people = client.get(f"/v3/workspaces/{workspace}/ingest/people").json()["items"]
    alex = next(p for p in people if p["peer_id"] == kept)
    assert {a["source"] for a in alex["accounts"]} == {"discord", "github"}

    # A refresh never reopens what a person decided.
    await _refresh(client, workspace)
    assert _suggestions(client, workspace, confidence="high") == []
    [kept_accepted] = _suggestions(client, workspace, status="accepted")
    assert kept_accepted["id"] == high["id"]


@pytest.mark.asyncio
async def test_a_dismissed_suggestion_stays_dismissed(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    await _refresh(client, workspace)
    [low] = _suggestions(client, workspace, confidence="low")
    base = f"/v3/workspaces/{workspace}/ingest/suggestions/{low['id']}"

    dismissed = client.post(f"{base}/dismiss")
    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["status"] == "dismissed"
    await _refresh(client, workspace)
    assert _suggestions(client, workspace, confidence="low") == []
    assert client.post(f"{base}/accept").status_code == 409
    missing = client.post(f"/v3/workspaces/{workspace}/ingest/suggestions/nope/dismiss")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_keep_chooses_the_person_and_must_be_one_of_them(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    await _refresh(client, workspace)
    [high] = _suggestions(client, workspace, confidence="high")
    await _reset(client, db_session, workspace)
    url = f"/v3/workspaces/{workspace}/ingest/suggestions/{high['id']}/accept"

    assert client.post(url, json={"keep": "someone-else"}).status_code == 409
    github = {a["source"]: a["peer_id"] for a in high["accounts"]}["github"]
    accepted = client.post(url, json={"keep": github})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["link"]["peer_id"] == github


@pytest.mark.asyncio
async def test_suggestions_need_an_auto_workspace(client: TestClient) -> None:
    workspace = _workspace(client, auto=False)
    base = f"/v3/workspaces/{workspace}/ingest/suggestions"
    assert client.post(f"{base}/refresh").status_code == 409
    assert client.get(base).status_code == 409


@pytest.mark.asyncio
async def test_one_waiting_refresh_covers_the_next_request(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    url = f"/v3/workspaces/{workspace}/ingest/suggestions/refresh"
    assert client.post(url).json() == {"queued": True}
    assert client.post(url).json() == {"queued": False}
    status = client.get(f"/v3/workspaces/{workspace}/ingest/status").json()
    assert status["suggestions"]["refreshing"] is True


@pytest.mark.asyncio
async def test_deleting_the_workspace_deletes_its_suggestions(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    await _refresh(client, workspace)
    assert _suggestions(client, workspace)

    await crud.delete_workspace(db_session, workspace)
    await db_session.commit()
    left = await db_session.scalars(
        select(models.IngestSuggestion).where(
            models.IngestSuggestion.workspace_name == workspace
        )
    )
    assert list(left) == []


@pytest.mark.asyncio
async def test_a_participant_only_account_is_matched_on_its_email(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    records = [
        # Meeting notes name an attendee by email; Slack has the same person.
        {
            **_record(1, {"id": "notes", "name": "Notes"}, "Sync notes", "granola"),
            "participants": [
                {"id": "attendee-1", "name": "Riley O", "email": "Riley@Team.example"}
            ],
        },
        _record(
            2,
            {"id": "s1", "name": "Riley", "email": "riley@team.example"},
            "Shipping today",
            "slack",
        ),  # fmt: skip
    ]
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"derive": "later", "records": records},
    )
    assert response.status_code == 200, response.text
    await _refresh(client, workspace)

    [high] = _suggestions(client, workspace, confidence="high")
    assert {a["id"] for a in high["accounts"]} == {"attendee-1", "s1"}
    assert high["evidence"][0] == {
        "signal": "email",
        "level": "strong",
        "detail": "same email address",
    }
    # The digest is never shown.
    assert "team.example" not in json.dumps(high).casefold()


@pytest.mark.asyncio
async def test_an_explicit_peer_id_matches_a_username_exactly(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    explicit = {
        **_record(1, ALEX, "From the app", "app"),
        "author": None,
        "peer_id": "alex-nguyen-4f2a",
    }
    records = [
        explicit,
        _record(2, {"id": "u9", "name": "Alex", "handle": "alex-nguyen-4f2a"}, "hi"),
    ]
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"derive": "later", "records": records},
    )
    assert response.status_code == 200, response.text
    await _refresh(client, workspace)

    [high] = _suggestions(client, workspace, confidence="high")
    assert {a["source"] for a in high["accounts"]} == {"explicit", "discord"}
    await _reset(client, db_session, workspace)
    accepted = client.post(
        f"/v3/workspaces/{workspace}/ingest/suggestions/{high['id']}/accept"
    )
    assert accepted.status_code == 200, accepted.text
    # The explicit peer can't be merged away: it is kept.
    assert accepted.json()["link"]["peer_id"] == "alex-nguyen-4f2a"


@pytest.mark.asyncio
async def test_a_refresh_writes_nothing_to_a_workspace_deleted_meanwhile(
    client: TestClient, db_session: AsyncSession
) -> None:
    from src.crud.ingest_suggestions import load_inputs, save_suggestions
    from src.organizer.identity import suggest

    workspace = _workspace(client)
    _import(client, workspace)
    old_id = await db_session.scalar(
        select(models.Workspace.id).where(models.Workspace.name == workspace)
    )
    inputs = await load_inputs(db_session, workspace)
    found = suggest(inputs.accounts, inputs.posts)
    assert found

    # Deleted, then made again under the same name, after the refresh read it.
    await crud.delete_workspace(db_session, workspace)
    await db_session.commit()
    _workspace_named(client, workspace)
    written = await save_suggestions(
        db_session, workspace, old_id or "", found, inputs.snapshots,
        datetime.datetime.now(datetime.UTC),
    )  # fmt: skip
    await db_session.commit()
    assert written == 0
    assert _suggestions(client, workspace, status="all") == []


def _workspace_named(client: TestClient, name: str) -> None:
    response = client.post("/v3/workspaces", json={"id": name, "auto": True})
    assert response.status_code in (200, 201), response.text


@pytest.mark.asyncio
async def test_a_refresh_requested_while_one_runs_is_queued_behind_it(
    client: TestClient, db_session: AsyncSession
) -> None:
    from src.deriver.enqueue import insert_organize_task
    from src.utils.queue_payload import create_organize_payload
    from src.utils.work_unit import construct_work_unit_key

    workspace = _workspace(client)
    assert await insert_organize_task(db_session, workspace) is True
    assert await insert_organize_task(db_session, workspace) is False
    # A worker claims it: the running refresh may have read too early.
    key = construct_work_unit_key(workspace, create_organize_payload())
    db_session.add(models.ActiveQueueSession(work_unit_key=key))
    await db_session.flush()
    assert await insert_organize_task(db_session, workspace) is True
    assert await insert_organize_task(db_session, workspace) is False
    await db_session.rollback()


@pytest.mark.asyncio
async def test_status_all_lists_every_state(client: TestClient) -> None:
    workspace = _workspace(client)
    _import(client, workspace)
    await _refresh(client, workspace)
    [low] = _suggestions(client, workspace, confidence="low")
    client.post(f"/v3/workspaces/{workspace}/ingest/suggestions/{low['id']}/dismiss")

    assert {s["status"] for s in _suggestions(client, workspace, status="all")} == {
        "open",
        "dismissed",
    }
    assert len(_suggestions(client, workspace)) == 1  # open, the default


@pytest.mark.asyncio
async def test_accept_keeps_a_peer_link_cant_merge_away(
    client: TestClient, db_session: AsyncSession
) -> None:
    """An account already linked into an explicit peer keeps that peer."""
    workspace = _workspace(client)
    discord_alex = {"id": "u9", "name": "Alex", "handle": "alexn"}
    github_alex = {"id": "g9", "name": "alexn", "handle": "alexn"}
    explicit = {**_record(1, ALEX, "From the app", "app"), "author": None}
    explicit["peer_id"] = "alex-nguyen"
    records = [
        explicit,
        _record(2, discord_alex, "I'm alexn on GitHub, by the way."),
        *(_record(10 + n, github_alex, f"PR {n}", "github") for n in range(3)),
    ]

    def send() -> None:
        response = client.post(
            f"/v3/workspaces/{workspace}/ingest",
            json={"derive": "later", "records": records},
        )
        assert response.status_code == 200, response.text

    send()
    discord_peer = next(
        p["peer_id"]
        for p in client.get(f"/v3/workspaces/{workspace}/ingest/people").json()["items"]
        if any(a["source"] == "discord" for a in p["accounts"])
    )
    await _reset(client, db_session, workspace)
    linked = client.post(
        f"/v3/workspaces/{workspace}/ingest/link",
        json={"peers": ["alex-nguyen", discord_peer]},
    )
    assert linked.status_code == 200, linked.text
    send()
    await _refresh(client, workspace)
    [high] = _suggestions(client, workspace, confidence="high")
    await _reset(client, db_session, workspace)

    # GitHub has more records, but the explicit peer can't be merged away.
    accepted = client.post(
        f"/v3/workspaces/{workspace}/ingest/suggestions/{high['id']}/accept"
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["link"]["peer_id"] == "alex-nguyen"

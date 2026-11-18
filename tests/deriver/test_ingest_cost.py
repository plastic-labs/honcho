"""What ingested history costs to derive: summaries and bots."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.ingest_enqueue import (
    _last_summaries,  # pyright: ignore[reportPrivateUsage]
)


def _workspace(client: TestClient) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name, "auto": True})
    assert response.status_code in (200, 201), response.text
    return name


def _record(n: int, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": "discord",
        "source_instance": "guild1",
        "source_id": f"msg-{n}",
        "created_at": f"2026-11-04T{10 + n // 60:02d}:{n % 60:02d}:00Z",
        "container": {"id": "chan-1", "name": "dev"},
        "author": {"id": "user-1", "name": "Jessica"},
        "content": f"message {n}",
    }
    record.update(overrides)
    return record


def _ingest(client: TestClient, workspace: str, records: list[dict[str, Any]]) -> None:
    for start in range(0, len(records), 100):
        response = client.post(
            f"/v3/workspaces/{workspace}/ingest",
            json={"records": records[start : start + 100]},
        )
        assert response.status_code == 200, response.text
        assert {r["outcome"] for r in response.json()["records"]} == {"created"}


async def _queue_payloads(
    db: AsyncSession, workspace: str, task_type: str
) -> list[dict[str, Any]]:
    rows = await db.scalars(
        select(models.QueueItem.payload).where(
            models.QueueItem.workspace_name == workspace,
            models.QueueItem.task_type == task_type,
        )
    )
    return list(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _summary_task(seq: int, short: int = 20, long: int = 60) -> dict[str, Any]:
    return {
        "task_type": "summary",
        "payload": {
            "message_seq_in_session": seq,
            "configuration": {
                "summary": {
                    "messages_per_short_summary": short,
                    "messages_per_long_summary": long,
                }
            },
        },
    }


def test_a_batch_keeps_only_its_last_long_and_last_short_summary() -> None:
    representation: dict[str, Any] = {"task_type": "representation", "payload": {}}
    tasks = [
        representation,
        *(_summary_task(seq) for seq in (20, 40, 60, 80, 100)),
        representation,
    ]

    kept = _last_summaries(tasks)

    seqs = [
        t["payload"]["message_seq_in_session"]
        for t in kept
        if t["task_type"] == "summary"
    ]
    assert seqs == [60, 100]
    assert [t for t in kept if t["task_type"] == "representation"] == [
        representation,
        representation,
    ]


def test_one_summary_is_kept_when_the_last_is_also_long() -> None:
    kept = _last_summaries([_summary_task(seq) for seq in (100, 120)])
    assert [t["payload"]["message_seq_in_session"] for t in kept] == [120]


def test_intervals_that_dont_divide_keep_both_summaries() -> None:
    """Short every 20, long every 30: a batch ending at 30 still needs 20's short."""
    kept = _last_summaries([_summary_task(seq, long=30) for seq in (20, 30)])
    assert [t["payload"]["message_seq_in_session"] for t in kept] == [20, 30]


@pytest.mark.asyncio
async def test_ingested_history_queues_only_the_last_summaries(
    client: TestClient, db_session: AsyncSession
) -> None:
    """110 messages cross six summary intervals; two summaries are queued."""
    workspace = _workspace(client)
    _ingest(client, workspace, [_record(n) for n in range(1, 111)])

    summaries = await _queue_payloads(db_session, workspace, "summary")

    assert sorted(p["message_seq_in_session"] for p in summaries) == [60, 100]
    # Representation work is untouched: one task per message.
    assert len(await _queue_payloads(db_session, workspace, "representation")) == 110


# ---------------------------------------------------------------------------
# Bots
# ---------------------------------------------------------------------------


async def _peer_configurations(db: AsyncSession, workspace: str) -> dict[str, Any]:
    rows = await db.execute(
        select(models.Peer.name, models.Peer.configuration).where(
            models.Peer.workspace_name == workspace
        )
    )
    return {name: configuration for name, configuration in rows.all()}


@pytest.mark.asyncio
async def test_a_bot_gets_a_peer_that_isnt_observed(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(
        client,
        workspace,
        [
            _record(
                1,
                author={"id": "sentry", "name": "Sentry", "bot": True},
                participants=[{"id": "user-1", "name": "Jessica"}],
            ),
            _record(
                2,
                participants=[{"id": "hook", "name": "Deploys", "bot": True}],
            ),
        ],
    )

    configurations = await _peer_configurations(db_session, workspace)
    observed = {
        p["observed"]
        for p in await _queue_payloads(db_session, workspace, "representation")
    }

    by_name = {name.split("-")[0]: c for name, c in configurations.items()}
    assert by_name["sentry"] == {"observe_me": False}
    assert by_name["deploys"] == {"observe_me": False}
    assert by_name["jessica"] == {}
    contexts = list(
        await db_session.scalars(
            select(models.IngestRecord.source_context).where(
                models.IngestRecord.workspace_name == workspace
            )
        )
    )
    # Kept as sent: the flag where the source set it, nowhere else.
    assert {"id": "sentry", "name": "Sentry", "bot": True} in [
        c["author"] for c in contexts if c
    ]
    assert {"id": "user-1", "name": "Jessica"} in [c["author"] for c in contexts if c]
    # The bot's message is stored and queued for nothing; the person's is.
    assert observed == {n for n in configurations if n.startswith("jessica")}


@pytest.mark.asyncio
async def test_a_peer_that_already_exists_keeps_its_configuration(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, [_record(1)])
    _ingest(
        client,
        workspace,
        [_record(2, author={"id": "user-1", "name": "Jessica", "bot": True})],
    )

    assert list((await _peer_configurations(db_session, workspace)).values()) == [{}]


@pytest.mark.asyncio
async def test_an_explicit_peer_ignores_the_author_flag(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(
        client,
        workspace,
        [
            _record(
                1,
                peer_id="alice",
                author={"id": "user-1", "name": "Jessica", "bot": True},
            )
        ],
    )

    assert (await _peer_configurations(db_session, workspace))["alice"] == {}

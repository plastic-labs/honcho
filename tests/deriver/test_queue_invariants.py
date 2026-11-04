"""Deriver queue lifecycle invariants.

Each scenario creates messages through the API (so the real enqueue runs),
injects one fault, drains the queue the way the polling loop does, then checks:

- nothing vanishes: every message has a representation queue item
- nothing gets stuck: no item is left unprocessed and no work unit stays claimed
- errors stay visible: queue status doesn't count an errored item as completed

A scenario that a known bug breaks is a strict xfail naming its issue, so the
fix has to remove the marker.
"""

import contextlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS, QueueManager

MESSAGES_PER_SCENARIO = 3
MAX_DRAIN_ROUNDS = 50


@dataclass(frozen=True)
class Scenario:
    # Raised in order by the deriver call, which then succeeds
    deriver_errors: tuple[Exception, ...] = ()
    # Raised by every deriver call
    deriver_error_forever: Exception | None = None
    # The queue insert in enqueue() hits a statement timeout
    enqueue_insert_fails: bool = False
    # A queue row whose key parse_work_unit_key rejects
    unparsable_work_unit_key: bool = False


_OUTAGE = httpx.ConnectError("upstream unreachable")
_STATUS_HIDES_ERRORS = pytest.mark.xfail(
    strict=True, reason="queue status counts errored items as completed (issue 1236)"
)

SCENARIOS = [
    pytest.param(Scenario(), id="healthy"),
    pytest.param(
        Scenario(deriver_errors=(_OUTAGE,) * (MAX_RETRYABLE_ATTEMPTS - 1)),
        id="provider-outage-recovers",
    ),
    pytest.param(
        Scenario(deriver_error_forever=_OUTAGE),
        id="provider-outage-outlasts-retries",
        marks=_STATUS_HIDES_ERRORS,
    ),
    pytest.param(
        Scenario(deriver_error_forever=ValueError("malformed output")),
        id="deriver-fails-terminally",
        marks=_STATUS_HIDES_ERRORS,
    ),
    pytest.param(
        Scenario(enqueue_insert_fails=True),
        id="enqueue-insert-fails",
        marks=pytest.mark.xfail(
            strict=True,
            reason="enqueue() logs and drops a failed queue insert; the saved "
            + "message is never derived",
        ),
    ),
    pytest.param(
        Scenario(unparsable_work_unit_key=True),
        id="unparsable-work-unit-key",
    ),
]


def _inject_faults(scenario: Scenario, stack: contextlib.ExitStack) -> None:
    deriver = AsyncMock(
        side_effect=scenario.deriver_error_forever
        or [*scenario.deriver_errors, *([None] * MAX_DRAIN_ROUNDS)]
    )
    stack.enter_context(
        patch("src.deriver.queue_manager.process_representation_batch", deriver)
    )
    if scenario.enqueue_insert_fails:
        timeout = OperationalError("INSERT INTO queue", {}, Exception("timeout"))
        stack.enter_context(
            patch("src.deriver.enqueue.handle_session", AsyncMock(side_effect=timeout))
        )


async def _drain(queue_manager: QueueManager) -> None:
    """Claim and process work units until none are claimable, like polling_loop."""
    for _ in range(MAX_DRAIN_ROUNDS):
        claimed = await queue_manager.get_and_claim_work_units()
        if not claimed:
            return
        for work_unit_key, aqs_id in claimed.items():
            worker_id = queue_manager.create_worker_id()
            queue_manager.track_worker_work_unit(worker_id, work_unit_key, aqs_id)
            # polling_loop runs each work unit as a task nobody awaits
            with contextlib.suppress(Exception):
                await queue_manager.process_work_unit(work_unit_key, worker_id)
    pytest.fail(f"queue still had claimable work after {MAX_DRAIN_ROUNDS} rounds")


def _create_messages(client: TestClient, workspace: str, session: str) -> None:
    peer = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace}/sessions/{session}/messages",
        json={
            "messages": [
                {"peer_id": peer, "content": f"message {index}"}
                for index in range(MESSAGES_PER_SCENARIO)
            ]
        },
    )
    assert response.status_code in (200, 201), response.text


async def _add_unparsable_work_unit(db_session: AsyncSession, workspace: str) -> None:
    db_session.add(
        models.QueueItem(
            task_type="dream",
            work_unit_key=f"dream:omni:{workspace}:user:extra:peer",
            payload={"task_type": "dream"},
            workspace_name=workspace,
        )
    )
    await db_session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_queue_invariants_hold_after_drain(
    scenario: Scenario,
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = sample_data[0].name
    session = str(generate_nanoid())
    monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", True)
    monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0)

    with contextlib.ExitStack() as faults:
        _inject_faults(scenario, faults)
        _create_messages(client, workspace, session)
        if scenario.unparsable_work_unit_key:
            await _add_unparsable_work_unit(db_session, workspace)
        queue_manager = QueueManager()
        await _drain(queue_manager)

    await db_session.rollback()

    # Nothing vanishes
    messages_without_items = (
        await db_session.execute(
            select(func.count())
            .select_from(models.Message)
            .where(models.Message.workspace_name == workspace)
            .where(
                ~select(models.QueueItem.id)
                .where(models.QueueItem.message_id == models.Message.id)
                .where(models.QueueItem.task_type == "representation")
                .exists()
            )
        )
    ).scalar_one()
    assert messages_without_items == 0, "messages saved with no representation item"

    # Nothing gets stuck
    unprocessed = (
        await db_session.execute(
            select(func.count())
            .select_from(models.QueueItem)
            .where(models.QueueItem.workspace_name == workspace)
            .where(~models.QueueItem.processed)
        )
    ).scalar_one()
    claims = (
        await db_session.execute(
            select(func.count()).select_from(models.ActiveQueueSession)
        )
    ).scalar_one()
    assert unprocessed == 0, "queue items left unprocessed after drain"
    assert claims == 0, "work units left claimed after drain"
    assert queue_manager.get_total_owned_work_units() == 0

    # Errors stay visible
    cleanly_processed = (
        await db_session.execute(
            select(func.count())
            .select_from(models.QueueItem)
            .where(models.QueueItem.workspace_name == workspace)
            .where(models.QueueItem.processed)
            .where(models.QueueItem.error.is_(None))
        )
    ).scalar_one()
    status = client.get(f"/v3/workspaces/{workspace}/queue/status").json()
    assert status["completed_work_units"] <= cleanly_processed, (
        "queue status reports errored items as completed"
    )

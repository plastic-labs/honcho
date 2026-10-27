"""Progress writes across actual images, alongside the full worker matrix.

These bounded fixtures call real worker methods and the progress callback at
controlled boundaries. They do not sample a running model task's transient stage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.compat_matrix.api import HonchoAPI
from tests.compat_matrix.ledger import (
    Ledger,
    QueueProgressExpect,
    SessionExpect,
    WorkspaceExpect,
)
from tests.compat_matrix.stack import Stack

logger = logging.getLogger(__name__)
WORKER_SCRIPT = Path(__file__).with_name("progress_worker.py").read_text()
PROGRESS_FIELDS = ("started_at", "last_activity_at", "stage", "current_queue_item_id")


def action(stack: Stack, image: str, mode: str, **request: Any) -> Any:
    output = stack.run(
        f"deriver-{image}",
        "python",
        "-c",
        WORKER_SCRIPT,
        json.dumps({**request, "mode": mode}),
    )
    prefix = "CMX_PROGRESS "
    line = next(row for row in reversed(output.splitlines()) if row.startswith(prefix))
    return json.loads(line.removeprefix(prefix))


def inspect(stack: Stack, fixture: dict[str, Any]) -> dict[str, Any]:
    rows = stack.query(
        "SELECT started_at, last_activity_at, stage, current_queue_item_id,"
        + " last_updated FROM active_queue_sessions WHERE id = %s",
        (fixture["claim"],),
    )
    assert len(rows) == 1, fixture
    return dict(zip((*PROGRESS_FIELDS, "last_updated"), rows[0], strict=True))


def public_row(fixture: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": fixture["claim"],
        "workspace_id": fixture["workspace"],
        "session_id": fixture["session"],
        "task_type": fixture["task"],
        "observer": "alice",
        "observers": [],
        "observed": "bob",
        **{key: fields[key] for key in ("started_at", "last_activity_at", "stage")},
    }


def snapshot(
    stack: Stack,
    ledger: Ledger,
    apis: dict[str, HonchoAPI],
    workspace: str,
    phase: str,
    counts: tuple[int, int, int, int],
    fixtures: list[dict[str, Any]],
) -> None:
    expectation = QueueProgressExpect(
        phase,
        workspace,
        dict(
            zip(
                (
                    "total_work_units",
                    "completed_work_units",
                    "in_progress_work_units",
                    "pending_work_units",
                ),
                counts,
                strict=True,
            )
        ),
        [public_row(f, inspect(stack, f)) for f in fixtures],
    )
    # Active claims are deliberately ephemeral; each controlled write supersedes
    # the previous expectation. The final absent/finished expectation persists.
    ledger.retract(
        lambda e: isinstance(e, QueueProgressExpect) and e.workspace == workspace
    )
    ledger.add(expectation)
    current = Ledger()
    current.add(expectation)
    problems = current.verify(list(apis.values()))
    if problems:
        raise AssertionError("\n".join(problems))
    logger.info("progress %s: counts=%s, claims=%d", phase, counts, len(fixtures))


def seed_legacy_progress(
    stack: Stack, ledger: Ledger, api: HonchoAPI
) -> dict[str, Any]:
    workspace = "cmx-progress-legacy"
    api.workspace(workspace, {})
    api.session(workspace, "legacy", [])
    ledger.add(
        WorkspaceExpect("progress before migration", workspace, {}),
        SessionExpect("progress before migration", workspace, "legacy", {}),
    )
    return action(
        stack,
        "old",
        "seed",
        workspace=workspace,
        session="legacy",
        task="summary",
        count=1,
    )


def check_legacy_progress(
    stack: Stack,
    ledger: Ledger,
    apis: dict[str, HonchoAPI],
    fixture: dict[str, Any],
) -> None:
    fields = inspect(stack, fixture)
    # Fields unknown to the old image remain null; fields a later old release
    # already recorded must survive migration unchanged.
    before = fixture["progress_before"]
    for key in PROGRESS_FIELDS:
        expected = before.get(key)
        if expected is not None and key in {"started_at", "last_activity_at"}:
            expected = datetime.fromisoformat(expected)
        assert fields[key] == expected
    read = action(stack, "old", "read", **fixture)
    assert read["columns"] == fixture["columns"]
    assert {"id", "work_unit_key", "last_updated"} <= set(read["columns"])
    assert read["item"] == fixture["items"][0]
    snapshot(
        stack,
        ledger,
        apis,
        fixture["workspace"],
        "pre-migration claim",
        (1, 0, 1, 0),
        [fixture],
    )
    action(stack, "old", "finish", **fixture)
    snapshot(
        stack, ledger, apis, fixture["workspace"], "legacy finalizer", (1, 1, 0, 0), []
    )


def run_progress_combo(
    stack: Stack, ledger: Ledger, apis: dict[str, HonchoAPI], tag: str
) -> None:
    """One controlled old/new image lifecycle per backfill round, workers stopped."""
    try:
        _run_progress_combo(stack, ledger, apis, tag)
    except Exception:
        # Do not let a failed controlled fixture become real worker input in
        # the following round. The original assertion still fails the matrix.
        workspace = f"cmx-progress-{tag}"
        with stack.connect() as conn:
            conn.execute(
                "DELETE FROM active_queue_sessions WHERE work_unit_key IN"
                + " (SELECT work_unit_key FROM queue WHERE workspace_name = %s)",
                (workspace,),
            )
            conn.execute("DELETE FROM queue WHERE workspace_name = %s", (workspace,))
        ledger.retract(
            lambda e: isinstance(e, QueueProgressExpect) and e.workspace == workspace
        )
        raise


def _run_progress_combo(
    stack: Stack, ledger: Ledger, apis: dict[str, HonchoAPI], tag: str
) -> None:
    workspace = f"cmx-progress-{tag}"
    apis["old"].workspace(workspace, {})
    ledger.add(WorkspaceExpect(tag, workspace, {}))
    for image, session in (
        ("old", "old-claim"),
        ("new", "new-claim"),
        ("old", "pending"),
    ):
        apis[image].session(workspace, session, [])
        ledger.add(SessionExpect(tag, workspace, session, {}))

    old = action(
        stack,
        "old",
        "seed",
        workspace=workspace,
        session="old-claim",
        task="summary",
        count=2,
    )
    new = action(
        stack,
        "new",
        "seed",
        workspace=workspace,
        session="new-claim",
        task="dream",
        count=1,
    )
    pending = action(
        stack,
        "old",
        "seed",
        workspace=workspace,
        session="pending",
        task="summary",
        count=1,
        claim=False,
    )
    for fixture in (old, new):
        fields = inspect(stack, fixture)
        assert fields["stage"] == "claimed" and fields["current_queue_item_id"] is None
        assert (
            fields["started_at"] == fields["last_activity_at"] == fields["last_updated"]
        )
        assert fields["started_at"] is not None
    snapshot(
        stack,
        ledger,
        apis,
        workspace,
        "old/new claims and pending",
        (4, 0, 3, 1),
        [old, new],
    )

    before = inspect(stack, old)
    action(stack, "old", "mark", **{**old, "items": old["items"][:1]})
    after = inspect(stack, old)
    assert after["last_updated"] > before["last_updated"]
    assert all(after[key] == before[key] for key in PROGRESS_FIELDS)
    snapshot(
        stack, ledger, apis, workspace, "old lease renewal", (4, 1, 2, 1), [old, new]
    )

    before = inspect(stack, new)
    action(stack, "new", "report", **new, stage="deduction")
    after = inspect(stack, new)
    assert (
        after["stage"] == "deduction"
        and after["current_queue_item_id"] == new["items"][0]
    )
    assert after["last_activity_at"] > before["last_activity_at"]
    assert after["started_at"] == before["started_at"]
    assert after["last_updated"] == before["last_updated"]
    assert action(stack, "old", "read", **new)["item"] == new["items"][0]
    assert action(stack, "old", "claim", **new) == {}
    assert not action(stack, "old", "cleanup", **{**new, "claim": "wrong-claim"})
    snapshot(
        stack,
        ledger,
        apis,
        workspace,
        "new progress and old reader",
        (4, 1, 2, 1),
        [old, new],
    )

    assert action(stack, "old", "stale", **old)
    snapshot(stack, ledger, apis, workspace, "old stale cleanup", (4, 1, 1, 2), [new])
    expired = old["claim"]
    old["claim"] = action(stack, "old", "claim", **old)[old["key"]]
    assert old["claim"] != expired
    replacement = inspect(stack, old)
    assert replacement["stage"] == "claimed"
    assert replacement["current_queue_item_id"] is None
    assert replacement["started_at"] == replacement["last_activity_at"]
    assert not action(stack, "old", "cleanup", **{**old, "claim": expired})
    action(stack, "new", "report", **{**old, "claim": expired}, stage="summary")
    assert inspect(stack, old) == replacement
    snapshot(
        stack,
        ledger,
        apis,
        workspace,
        "reclaim and obsolete ID guards",
        (4, 1, 2, 1),
        [old, new],
    )

    action(stack, "old", "finish", **old)
    snapshot(
        stack, ledger, apis, workspace, "old empty-loop finalizer", (4, 2, 1, 1), [new]
    )
    action(stack, "new", "finish", **new)
    snapshot(
        stack, ledger, apis, workspace, "new empty-loop finalizer", (4, 3, 0, 1), []
    )
    action(stack, "old", "finish", **pending)
    snapshot(
        stack, ledger, apis, workspace, "all fixture work finished", (4, 4, 0, 0), []
    )

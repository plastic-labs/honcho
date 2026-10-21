"""Fast harness tests; deliberately outside tests/ and its database fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call

import httpx
import pytest
from psycopg.errors import ForeignKeyViolation


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICE_PROJECT", "slice_harness")
    path = Path(__file__).parents[1] / "slice_runner.py"
    spec = importlib.util.spec_from_file_location("slice_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        label="unit",
        image_under_test="prototype",
        previous_image="previous",
        scenario="persisted-cache",
        unsafe_negative_control_image="unsafe-v3.4.0",
        release_on="same",
        delete_action="api",
        settle_seconds=0,
        results_dir=str(tmp_path),
    )
    return module.Run(args)


@pytest.mark.parametrize(
    "check_id",
    [
        "context-clean",
        "summary-clean-short",
        "summary-clean-long",
        "card-clean-alice",
        "card-clean-bob",
        "card-clean-carol",
    ],
)
@pytest.mark.parametrize(
    "phase,status,canary,passed",
    [
        ("unsafe-negative-control", 200, True, True),
        ("unsafe-negative-control", 200, False, False),
        ("unsafe-negative-control", 500, True, False),
        ("unsafe-negative-control", 599, False, False),
        ("prototype-persisted-cache", 200, True, False),
        ("prototype-persisted-cache", 200, False, True),
        ("prototype-persisted-cache", 500, False, False),
    ],
)
def test_visibility_expectations(runner, check_id, phase, status, canary, passed):
    runner.visibility_check(
        check_id,
        "surface",
        "api-1",
        phase,
        httpx.Response(status),
        canary,
    )
    check = runner.checks[-1]
    assert check["pass"] is passed
    assert check["canary_present"] is canary
    assert check["status_code"] == status


def cache_fixture(runner, monkeypatch):
    keys = [f"ns:workspace:slice-ws:peer:{p}" for p in ("alice", "bob", "carol")]
    keys.append("ns:workspace:slice-ws:session:slice-session")
    runner.cache_keys = keys
    client = MagicMock()
    client.__enter__.return_value = client
    client.info.return_value = {
        "run_id": "pre-outage",
        "aof_enabled": 1,
        "aof_current_size": 100,
    }
    client.ttl.return_value = -1
    values = {k: runner.canary.encode() * (2 if ":session:" in k else 1) for k in keys}
    client.get.side_effect = values.get
    monkeypatch.setattr(runner, "redis", lambda: client)
    monkeypatch.setattr(
        runner,
        "compose",
        lambda *a, **kw: subprocess.CompletedProcess([], 0, "same-container"),
    )
    runner.cache_snapshot("before-outage", baseline=True)
    client.info.return_value["run_id"] = "post-outage"
    return client, values


@pytest.mark.parametrize(
    "defect", ["missing", "changed", "expiring", "no-aof", "no-restart"]
)
def test_cache_proof_rejects_false_positive(runner, monkeypatch, defect):
    client, values = cache_fixture(runner, monkeypatch)
    key = runner.cache_keys[0]
    if defect == "missing":
        del values[key]
    elif defect == "changed":
        values[key] += b"rewritten"
    elif defect == "expiring":
        client.ttl.return_value = 30
    elif defect == "no-aof":
        client.info.return_value["aof_enabled"] = 0
    else:
        client.info.return_value["run_id"] = "pre-outage"
    with pytest.raises(AssertionError):
        runner.cache_snapshot("after-restart")


def test_exact_bytes_survive_and_are_not_modified_by_proof(runner, monkeypatch):
    client, _ = cache_fixture(runner, monkeypatch)
    runner.cache_snapshot("after-restart")
    assert runner.cache_snapshots[0]["entries"] == runner.cache_snapshots[1]["entries"]
    client.set.assert_not_called()
    client.delete.assert_not_called()
    client.expire.assert_not_called()


@pytest.mark.parametrize(
    "defect", [None, "no-client", "fallback", "no-log", "live-worker"]
)
def test_redis_backend_proof(runner, monkeypatch, defect):
    client = MagicMock()
    client.__enter__.return_value = client
    client.client_list.return_value = (
        [{"addr": "172.20.0.2:45000"}] if defect != "no-client" else []
    )
    monkeypatch.setattr(runner, "redis", lambda: client)
    monkeypatch.setattr(
        runner,
        "inspect_container",
        lambda service: {
            "NetworkSettings": {"Networks": {"test": {"IPAddress": "172.20.0.2"}}},
        },
    )
    logs = "Connected to cache at redis://redis:6379/0"
    if defect == "fallback":
        logs += " Falling back to a process-local in-memory cache"
    elif defect == "no-log":
        logs = ""
    monkeypatch.setattr(
        runner, "compose", lambda *a, **kw: subprocess.CompletedProcess([], 0, logs)
    )
    monkeypatch.setattr(
        runner, "container_exited", lambda service: defect != "live-worker"
    )
    if defect:
        with pytest.raises(AssertionError):
            runner.assert_redis_startup("comparison")
    else:
        runner.assert_redis_startup("comparison")
        assert set(runner.redis_connections["comparison"]) == {"api", "api-2"}


def test_comparison_restores_redis_before_either_image_and_stops_workers(
    runner, monkeypatch
):
    calls = []
    for method in (
        "stop_apps",
        "start_redis",
        "cache_snapshot",
        "swap",
        "assert_redis_startup",
        "run_checks",
    ):
        monkeypatch.setattr(
            runner, method, lambda *a, _name=method, **kw: calls.append((_name, a, kw))
        )
    runner.persisted_comparison()
    assert calls[:3] == [
        ("stop_apps", (), {}),
        ("start_redis", (), {}),
        ("cache_snapshot", ("redis-returned-before-image-start",), {}),
    ]
    swaps = [c for c in calls if c[0] == "swap"]
    assert swaps == [
        ("swap", ("unsafe-v3.4.0",), {"readers_only": True}),
        ("swap", ("prototype",), {"readers_only": True}),
    ]
    assert [c[0] for c in calls].count("stop_apps") == 3
    assert calls[-1] == ("stop_apps", (), {})


@pytest.mark.parametrize(
    "outcome", ["bare-success", "recovered", "detach-fails", "retry-fails"]
)
def test_sql_delete_verdict_through_execute(runner, monkeypatch, outcome):
    runner.args.scenario = "original"
    runner.args.delete_action = "sql"
    runner.message_id = 42
    # Keep the actual delete() and execute() paths; stub only orchestration and
    # I/O so recovery and its final verdict are tested without a database/Docker.
    for method in (
        "up",
        "seed",
        "wait_paused",
        "warm_reads",
        "stop_redis",
        "run_checks",
        "release",
        "swap",
        "start_redis",
        "record_phase",
    ):
        monkeypatch.setattr(runner, method, MagicMock())
    bare_delete = "DELETE FROM messages WHERE id = %s"
    detach = "UPDATE queue SET message_id = NULL WHERE message_id = %s"
    effects = [ForeignKeyViolation("queue message FK"), [], []]
    queries = [bare_delete, detach, bare_delete]
    if outcome == "bare-success":
        effects = [[]]
        queries = [bare_delete]
    elif outcome == "detach-fails":
        effects[1] = ForeignKeyViolation("detach failed")
        queries = queries[:2]
    elif outcome == "retry-fails":
        effects[2] = ForeignKeyViolation("retry failed")
    sql = MagicMock(side_effect=effects)
    monkeypatch.setattr(runner, "sql", sql)

    result = runner.execute()

    assert sql.call_args_list == [call(query, (42,)) for query in queries]
    assert ("queue message FK" in result["notes"]) is (outcome != "bare-success")
    delete_steps = [step for step in result["steps"] if step["name"] == "delete-sql"]
    if outcome in ("bare-success", "recovered"):
        assert result["verdict"] == "pass"
        assert result["counts"]["failed_steps"] == 0
        assert len(delete_steps) == 1 and delete_steps[0]["ok"] is True
        assert ("detached" in delete_steps[0]["detail"]) is (outcome == "recovered")
        runner.release.assert_called_once_with()
        assert runner.run_checks.call_args_list == [
            call("after-delete"),
            call("after-release"),
            call("after-swap"),
        ]
    else:
        assert result["verdict"] == "blocked"
        assert result["counts"]["failed_steps"] == 1
        assert not delete_steps
        error = next(step for step in result["steps"] if step["name"] == "runner-error")
        assert error["ok"] is False
        assert outcome.replace("-fails", " failed") in error["detail"]
        runner.release.assert_not_called()
        runner.run_checks.assert_not_called()
        runner.record_phase.assert_called_once_with("runner-error")


def test_failed_prerequisite_cannot_report_pass(runner, monkeypatch):
    for method in (
        "up",
        "seed",
        "wait_paused",
        "warm_reads",
        "preserve_cache",
        "stop_redis",
        "delete",
        "run_checks",
        "release",
        "persisted_comparison",
    ):
        monkeypatch.setattr(runner, method, lambda *a: None)
    monkeypatch.setattr(runner, "up", lambda: runner.step("topology-up", False))
    assert runner.execute()["verdict"] == "fail"


def test_original_order_is_preserved(runner, monkeypatch):
    runner.args.scenario = "original"
    calls = []
    for method in (
        "up",
        "seed",
        "wait_paused",
        "warm_reads",
        "stop_redis",
        "delete",
        "run_checks",
        "release",
        "swap",
        "start_redis",
    ):
        monkeypatch.setattr(
            runner, method, lambda *a, _name=method: calls.append((_name, a))
        )
    runner.execute()
    assert calls == [
        ("up", ()),
        ("seed", ()),
        ("wait_paused", ()),
        ("warm_reads", ()),
        ("stop_redis", ()),
        ("delete", ()),
        ("run_checks", ("after-delete",)),
        ("release", ()),
        ("run_checks", ("after-release",)),
        ("swap", ()),
        ("start_redis", ()),
        ("run_checks", ("after-swap",)),
    ]

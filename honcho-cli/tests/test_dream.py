"""`honcho dream list` / `honcho dream inspect` (Representation History)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from honcho import NotFoundError
from honcho.api_types import DreamRunDetailResponse, DreamRunResponse
from typer.testing import CliRunner

from honcho_cli.main import app

STARTED = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
FINISHED = datetime(2026, 11, 3, 12, 1, 30, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    f.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
    return f


@pytest.fixture
def runner():
    return CliRunner()


def _run(**overrides) -> DreamRunResponse:
    fields = {
        "id": "run1",
        "observer_id": "maya",
        "observed_id": "maya",
        "dream_type": "omni",
        "trigger_reason": "threshold",
        "status": "completed",
        "created_count": 1,
        "removed_count": 1,
        "peer_card_updated": True,
        "started_at": STARTED,
        "finished_at": FINISHED,
    }
    fields.update(overrides)
    return DreamRunResponse(**fields)


def _detail() -> DreamRunDetailResponse:
    return DreamRunDetailResponse(
        **_run().model_dump(),
        created=[
            {
                "conclusion": {
                    "id": "c3",
                    "content": "Maya relies on caffeine to get through night shifts",
                    "level": "deductive",
                    "status": "current",
                },
                "sources": [
                    {
                        "id": "c1",
                        "content": "Maya drinks coffee daily",
                        "level": "explicit",
                        "status": "removed",
                        "removed_in_run": "run1",
                    },
                    {
                        "id": "c2",
                        "content": "Maya works night shifts",
                        "level": "explicit",
                        "status": "current",
                    },
                    {"id": "c0", "status": "unavailable"},
                ],
                "specialist": "deduction",
            }
        ],
        removed=[
            {
                "id": "c1",
                "content": "Maya drinks coffee daily",
                "level": "explicit",
                "replaced_by": "c3",
                "specialist": "deduction",
            }
        ],
        peer_card={
            "before": ["IDENTITY: Name: Maya", "ATTRIBUTE: Drinks coffee"],
            "after": ["IDENTITY: Name: Maya", "ATTRIBUTE: Works night shifts"],
        },
    )


def _client_with(**methods) -> MagicMock:
    client = MagicMock()
    for name, value in methods.items():
        setattr(client, name, value)
    return client


def test_list_filters_by_peer_and_observer(cfg, runner):
    page = MagicMock(items=[_run()], page=1, pages=1)
    dreams = MagicMock(return_value=page)
    config = MagicMock(peer_id="maya", workspace_id="ws1")
    with patch(
        "honcho_cli.commands.dream.get_client",
        return_value=(_client_with(dreams=dreams), config),
    ):
        result = runner.invoke(
            app,
            ["dream", "list", "-w", "ws1", "-p", "maya", "--observer", "assistant", "--limit", "5"],
        )

    assert result.exit_code == 0, result.stderr
    dreams.assert_called_once_with(observed="maya", observer="assistant", page=1, size=5)
    [row] = json.loads(result.stdout)
    assert (row["id"], row["status"], row["created_count"]) == ("run1", "completed", 1)


def test_list_without_a_peer_sends_no_peer_filter(cfg, runner):
    dreams = MagicMock(return_value=MagicMock(items=[], page=1, pages=0))
    config = MagicMock(peer_id="", workspace_id="ws1")  # the config default
    with patch(
        "honcho_cli.commands.dream.get_client",
        return_value=(_client_with(dreams=dreams), config),
    ):
        result = runner.invoke(app, ["dream", "list", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    dreams.assert_called_once_with(observed=None, observer=None, page=1, size=20)
    assert json.loads(result.stdout) == []


def test_list_rejects_non_positive_paging(cfg, runner):
    result = runner.invoke(app, ["dream", "list", "-w", "ws1", "--limit", "0"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"


def test_list_renders_full_run_ids_for_a_terminal(cfg, runner):
    run_id = "kRkwKbgz4bqSRJjRXmp2_"
    page = MagicMock(items=[_run(id=run_id, removed_count=0)], page=1, pages=1)
    config = MagicMock(peer_id="maya", workspace_id="ws1")
    with (
        patch(
            "honcho_cli.commands.dream.get_client",
            return_value=(_client_with(dreams=MagicMock(return_value=page)), config),
        ),
        patch("honcho_cli.commands.dream.use_json", return_value=False),
    ):
        narrow = runner.invoke(app, ["dream", "list", "-w", "ws1", "-p", "maya"], env={"COLUMNS": "60"})
        wide = runner.invoke(app, ["dream", "list", "-w", "ws1", "-p", "maya"], env={"COLUMNS": "160"})

    assert narrow.exit_code == 0, narrow.stderr
    assert run_id in narrow.stdout  # the other columns wrap; the id never does
    assert "maya → maya" in wide.stdout
    assert "+1 -0  card" in wide.stdout


def test_inspect_json_is_the_full_detail(cfg, runner):
    get_dream = MagicMock(return_value=_detail())
    with patch(
        "honcho_cli.commands.dream.get_client",
        return_value=(_client_with(get_dream=get_dream), MagicMock()),
    ):
        result = runner.invoke(app, ["dream", "inspect", "run1", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    get_dream.assert_called_once_with("run1")
    body = json.loads(result.stdout)
    assert body["created"][0]["sources"][0]["status"] == "removed"
    assert body["removed"][0]["replaced_by"] == "c3"
    assert body["peer_card"]["after"][1] == "ATTRIBUTE: Works night shifts"


def test_inspect_renders_changes_for_a_terminal(cfg, runner):
    get_dream = MagicMock(return_value=_detail())
    with (
        patch(
            "honcho_cli.commands.dream.get_client",
            return_value=(_client_with(get_dream=get_dream), MagicMock()),
        ),
        patch("honcho_cli.commands.dream.use_json", return_value=False),
    ):
        result = runner.invoke(app, ["dream", "inspect", "run1", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert "maya → maya" in out
    assert "1m 30s" in out
    assert "Maya relies on caffeine" in out
    assert "removed in this dream" in out
    assert "unavailable" in out
    assert "replaced by: Maya relies on caffeine" in out
    assert "- ATTRIBUTE: Drinks coffee" in out
    assert "+ ATTRIBUTE: Works night shifts" in out


def test_inspect_says_when_retention_pruned_the_run(cfg, runner):
    detail = _detail().model_copy(
        update={"pruned_at": FINISHED, "removed": [], "peer_card": None}
    )
    get_dream = MagicMock(return_value=detail)
    with (
        patch(
            "honcho_cli.commands.dream.get_client",
            return_value=(_client_with(get_dream=get_dream), MagicMock()),
        ),
        patch("honcho_cli.commands.dream.use_json", return_value=False),
    ):
        result = runner.invoke(app, ["dream", "inspect", "run1", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    assert "past retention" in result.stdout
    assert "Maya relies on caffeine" in result.stdout


def test_inspect_unknown_dream_exits_with_not_found(cfg, runner):
    get_dream = MagicMock(side_effect=NotFoundError("not found"))
    with patch(
        "honcho_cli.commands.dream.get_client",
        return_value=(_client_with(get_dream=get_dream), MagicMock()),
    ):
        result = runner.invoke(app, ["dream", "inspect", "nope", "-w", "ws1"])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "DREAM_NOT_FOUND"

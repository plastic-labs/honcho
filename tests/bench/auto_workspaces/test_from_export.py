"""A connector export becomes a dataset with identity and bot truth."""

import json
from pathlib import Path
from typing import Any

from tests.bench.auto_workspaces.from_export import convert
from tests.bench.auto_workspaces.model import Dataset


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _record(source_id: str, author: str, *participants: str) -> dict[str, Any]:
    return {
        "source": "discord",
        "source_instance": "77",
        "source_id": source_id,
        "created_at": "2026-11-04T20:47:00+00:00",
        "container": {"id": "200", "name": "dev", "parent_id": None},
        "author": {"id": author, "name": f"n{author}"},
        "participants": [{"id": p, "name": f"n{p}"} for p in participants],
        "content": "hello",
    }


def test_an_export_becomes_a_dataset(tmp_path: Path) -> None:
    (tmp_path / "export.json").write_text(
        json.dumps({"source": "discord", "source_instance": "77"})
    )
    _write(
        tmp_path / "accounts.jsonl",
        [
            {"id": "1", "username": "jess", "display_name": "Jess", "bot": False},
            {"id": "2", "username": "sentry", "display_name": "Sentry", "bot": True},
            {"id": "3", "username": "hook", "display_name": "Hook", "bot": False},
            {"id": "4", "username": "guest", "display_name": "Guest", "bot": False},
        ],
    )
    _write(
        tmp_path / "records.jsonl",
        [_record("9001", "1", "4"), _record("9002", "2"), _record("9003", "3")],
    )
    people = {"people": {"Jess": {"discord": ["jess"]}}, "bots": {"discord": ["hook"]}}

    dataset = convert(tmp_path, people)

    key = "discord/77/id:{}".format
    assert dataset.truth.person_of[key("1")] == "Jess"
    assert dataset.truth.person_of[key("4")] == f"self:{key('4')}"
    # Flagged by the source, or listed as an integration that posts as a person.
    assert dataset.truth.bots == {key("2"), key("3")}
    assert dataset.profiles[key("2")].is_bot
    assert not dataset.profiles[key("3")].is_bot
    assert [r.key for r in dataset.records] == [
        "discord/77/9001",
        "discord/77/9002",
        "discord/77/9003",
    ]
    # A record goes to its author's unit, so a split holds out whole people.
    assert dataset.records[0].unit == "discord:Jess"
    assert dataset.truth.groups_of == {}

    dataset.write(tmp_path / "out")
    assert Dataset.read(tmp_path / "out").truth.person_of == dataset.truth.person_of


def test_only_keeps_a_category_or_channel_with_its_threads(tmp_path: Path) -> None:
    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "source": "discord",
                "source_instance": "77",
                "categories": {"10": "growth", "11": "dev"},
            }
        )
    )
    _write(
        tmp_path / "containers.jsonl",
        [
            {"id": "200", "name": "general", "parent_id": "10", "type": 0},
            {"id": "201", "name": "a-thread", "parent_id": "200", "type": 11},
            {"id": "300", "name": "devops", "parent_id": "11", "type": 0},
        ],
    )
    _write(
        tmp_path / "accounts.jsonl",
        [
            {"id": "1", "username": "a", "display_name": "A", "bot": False},
            {"id": "2", "username": "b", "display_name": "B", "bot": False},
        ],
    )

    def at(record: dict[str, Any], container: str) -> dict[str, Any]:
        return {**record, "container": {"id": container, "name": "x"}}

    _write(
        tmp_path / "records.jsonl",
        [
            at(_record("1", "1"), "200"),
            at(_record("2", "1"), "201"),
            at(_record("3", "2"), "300"),
        ],
    )

    for only in ({"growth"}, {"200"}):
        dataset = convert(tmp_path, {}, only=only)
        assert [r.record["source_id"] for r in dataset.records] == ["1", "2"]
        # Only the accounts the kept records name.
        assert set(dataset.truth.person_of) == {"discord/77/id:1"}
        assert set(dataset.profiles) == {"discord/77/id:1"}


def test_a_thread_is_kept_by_its_channel_even_when_the_channel_isnt_listed(
    tmp_path: Path,
) -> None:
    (tmp_path / "export.json").write_text(
        json.dumps({"source": "discord", "source_instance": "77", "categories": {}})
    )
    _write(
        tmp_path / "containers.jsonl",
        [{"id": "201", "name": "a-thread", "parent_id": "200", "type": 11}],
    )
    _write(
        tmp_path / "accounts.jsonl",
        [{"id": "1", "username": "a", "display_name": "A", "bot": False}],
    )
    _write(
        tmp_path / "records.jsonl",
        [{**_record("1", "1"), "container": {"id": "201", "name": "a-thread"}}],
    )

    dataset = convert(tmp_path, {}, only={"200"})

    assert [r.record["source_id"] for r in dataset.records] == ["1"]

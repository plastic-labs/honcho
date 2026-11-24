"""The published ingest record schema: current, usable, and in step with the SDK.

`docs/schemas/ingest-record.v0.json` is what exporter authors validate against;
`honcho ingest import --check` validates with the SDK's `IngestRecord`. The two
must accept the same fields.
"""

import json
from typing import Any

import pytest
from pydantic import BaseModel

from scripts.generate_ingest_schema import SCHEMA_PATH, build_schema, render
from sdks.python.src.honcho.api_types import (
    IngestContainer,
    IngestParticipant,
    IngestRecord,
)

RECORD: dict[str, Any] = {
    "source": "github",
    "source_instance": "github-com",
    "source_id": "IC_kwDO123",
    "revised_at": "2026-11-02T09:00:00Z",
    "created_at": "2026-11-01T09:00:00Z",
    "container": {"id": "R_1", "name": "acme/web", "parent_id": "O_1"},
    "author": {"id": "1001", "name": "alex"},
    "participants": [{"id": "9", "name": "coderabbitai", "bot": True}],
    "content": "Comment on PR #12: looks good",
    "metadata": {"thread": "pr:12", "reply_to": "IC_kwDO122"},
    "configuration": {"reasoning": {"enabled": False}},
}


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def test_the_committed_schema_is_current() -> None:
    assert SCHEMA_PATH.read_text() == render(build_schema())


def test_a_connector_record_validates_and_a_bad_one_does_not() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(_schema())

    assert list(validator.iter_errors(RECORD)) == []
    without_source = {k: v for k, v in RECORD.items() if k != "source"}
    assert list(validator.iter_errors(without_source))
    # A misspelled field fails rather than being dropped, here and in a container.
    assert list(validator.iter_errors({**RECORD, "contnet": "x"}))
    assert list(
        validator.iter_errors({**RECORD, "container": {"id": "R_1", "nmae": "web"}})
    )
    # Explicit session IDs follow the resource-name pattern, as ingest checks.
    assert list(validator.iter_errors({**RECORD, "session_id": "bad/session"}))
    assert list(validator.iter_errors({**RECORD, "session_id": "team_chat-1"})) == []


@pytest.mark.parametrize(
    ("model", "schema_path"),
    [
        (IngestRecord, ()),
        (IngestContainer, ("$defs", "IngestContainer")),
        (IngestParticipant, ("$defs", "IngestParticipant")),
    ],
)
def test_the_schema_and_the_sdk_have_the_same_fields(
    model: type[BaseModel], schema_path: tuple[str, ...]
) -> None:
    node: dict[str, Any] = _schema()
    for key in schema_path:
        node = node[key]
    assert set(node["properties"]) == set(model.model_fields)
    assert node["additionalProperties"] is False


def _with(**changes: Any) -> dict[str, Any]:
    return {**RECORD, **changes}


_PERSON = {"id": "1", "name": "alex"}


def _nested(levels: int) -> dict[str, Any]:
    """Metadata `levels` dictionaries deep."""
    node: dict[str, Any] = {"leaf": 1}
    for _ in range(levels - 1):
        node = {"next": node}
    return node


_PARITY_CASES: list[tuple[str, dict[str, Any]]] = [
    ("valid", RECORD),
    (
        "minimal",
        {k: RECORD[k] for k in ("source", "source_instance", "created_at", "content")},
    ),
    ("source with a slash", _with(source="bad/source")),
    ("source too long", _with(source="s" * 65)),
    ("instance with a space", _with(source_instance="git hub")),
    ("instance too long", _with(source_instance="i" * 129)),
    ("source_id with NUL", _with(source_id="a\x00b")),
    ("source_id with a lone surrogate", _with(source_id="a\ud800")),
    ("source_id too long", _with(source_id="x" * 513)),
    ("empty container id", _with(container={"id": ""})),
    ("container parent with NUL", _with(container={"id": "c", "parent_id": "p\x00"})),
    ("author name only NUL", _with(author={"id": "1", "name": "\x00"})),
    ("author name with NUL", _with(author={"id": "1", "name": "al\x00ex"})),
    ("100 participants", _with(participants=[_PERSON] * 100)),
    ("101 participants", _with(participants=[_PERSON] * 101)),
    ("session_id with a slash", _with(session_id="bad/session")),
    ("session_id", _with(session_id="team_chat-1")),
    ("peer_id with NUL", _with(peer_id="p\x00")),
    ("no created_at", {k: v for k, v in RECORD.items() if k != "created_at"}),
    ("content with NUL", _with(content="a\x00b")),
    ("metadata with 100 keys", _with(metadata={f"k{i}": i for i in range(100)})),
    ("metadata with 101 keys", _with(metadata={f"k{i}": i for i in range(101)})),
    ("metadata 5 deep", _with(metadata=_nested(5))),
    ("metadata 6 deep", _with(metadata=_nested(6))),
    ("metadata with NUL", _with(metadata={"thread": "pr:\x0012"})),
]


def _accepts(model: type[BaseModel], record: dict[str, Any]) -> bool:
    try:
        model.model_validate(record)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    ("case", "record"), _PARITY_CASES, ids=[c for c, _ in _PARITY_CASES]
)
def test_the_sdk_accepts_what_the_server_accepts(
    case: str, record: dict[str, Any]
) -> None:
    # `honcho ingest import --check` validates with the SDK model: a record it
    # passes must be one the server takes, and the other way round.
    from src.schemas.api import IngestRecordCreate

    assert _accepts(IngestRecord, record) == _accepts(IngestRecordCreate, record), case


def test_the_deliberate_differences() -> None:
    from src.schemas.api import IngestRecordCreate

    # Unknown fields: the server ignores them, the SDK (and so the CLI) rejects
    # them, at the top level and inside the configuration alike.
    for record in (
        _with(contnet="x"),
        _with(configuration={"reasoning": {"enabld": False}}),
    ):
        assert _accepts(IngestRecordCreate, record)
        assert not _accepts(IngestRecord, record)
    # Two limits are server settings: content length (MAX_MESSAGE_SIZE, which
    # `--check` applies at its default) and the token budget for custom
    # instructions, which only the server checks.
    long = _with(content="x" * 25_001)
    assert not _accepts(IngestRecordCreate, long)
    assert _accepts(IngestRecord, long)
    instructed = _with(
        configuration={"reasoning": {"custom_instructions": "word " * 20_000}}
    )
    assert not _accepts(IngestRecordCreate, instructed)
    assert _accepts(IngestRecord, instructed)

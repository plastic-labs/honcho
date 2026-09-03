"""Tests for the evidence accumulator in src/utils/evidence.py.

These exercise collation in isolation: rows are built in memory and never
written, so nothing here depends on what the agent or its tools do. The wiring
that feeds the accumulator is covered in tests/utils/test_agent_tools.py.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src import models
from src.schemas.api import EVIDENCE_MESSAGE_PREVIEW_CHARS
from src.utils.evidence import EvidenceAccumulator

NOW = datetime(2026, 1, 1, tzinfo=UTC)
LEVELS = ("explicit", "deductive", "inductive", "contradiction")


def make_document(
    doc_id: str,
    *,
    level: str = "explicit",
    content: str = "User likes coffee",
    internal_metadata: dict[str, Any] | None = None,
    source_ids: list[str] | None = None,
    session_name: str | None = "session-1",
    created_at: datetime = NOW,
) -> models.Document:
    """Build an unpersisted Document.

    Note `internal_metadata=` and not `metadata=`: `metadata` is SQLAlchemy's
    own class attribute and assigning it shadows that instead of setting the
    column.
    """
    return models.Document(
        id=doc_id,
        level=level,
        content=content,
        internal_metadata=internal_metadata or {},
        source_ids=source_ids,
        session_name=session_name,
        created_at=created_at,
        observer="observer",
        observed="observed",
        workspace_name="workspace",
    )


def make_message(
    public_id: str,
    *,
    content: str = "I love coffee",
    peer_name: str = "alice",
    session_name: str = "session-1",
    created_at: datetime = NOW,
) -> models.Message:
    return models.Message(
        public_id=public_id,
        content=content,
        peer_name=peer_name,
        session_name=session_name,
        created_at=created_at,
        workspace_name="workspace",
        seq_in_session=1,
    )


class TestConclusionCollection:
    def test_records_id_level_and_content(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_documents([make_document("doc-1", content="User likes tea")])

        (conclusion,) = accumulator.build().conclusions
        assert conclusion.id == "doc-1"
        assert conclusion.level == "explicit"
        assert conclusion.content == "User likes tea"
        assert conclusion.session_id == "session-1"

    def test_deduplicates_by_id_across_tools(self):
        """A conclusion two tools both returned is reported once."""
        accumulator = EvidenceAccumulator()
        accumulator.add_documents([make_document("doc-1")])
        accumulator.add_documents([make_document("doc-1"), make_document("doc-2")])

        assert [c.id for c in accumulator.build().conclusions] == ["doc-1", "doc-2"]

    def test_keeps_distinct_conclusions_that_read_alike(self):
        """Identical text is not identity.

        `Representation`'s own deduplication keys on content and timestamp and
        ignores IDs, so building evidence through it without keying on ID first
        would drop one of these and pick arbitrarily between their IDs.
        """
        accumulator = EvidenceAccumulator()
        accumulator.add_documents(
            [
                make_document("doc-1", content="User likes coffee"),
                make_document("doc-2", content="User likes coffee"),
            ]
        )

        assert [c.id for c in accumulator.build().conclusions] == ["doc-1", "doc-2"]

    @pytest.mark.parametrize(
        ("level", "document_kwargs"),
        [
            ("inductive", {"source_ids": ["doc-1", "doc-2"]}),
            ("deductive", {"internal_metadata": {"premise_ids": ["doc-1", "doc-2"]}}),
            ("inductive", {"internal_metadata": {"source_ids": ["doc-1", "doc-2"]}}),
        ],
        ids=["column", "metadata-premise-ids", "metadata-source-ids"],
    )
    def test_resolves_source_ids_from_either_location(
        self, level: str, document_kwargs: dict[str, Any]
    ):
        """Rows predating the `source_ids` column keep their premises in metadata."""
        accumulator = EvidenceAccumulator()
        accumulator.add_documents(
            [make_document("doc-3", level=level, **document_kwargs)]
        )

        (conclusion,) = accumulator.build().conclusions
        assert conclusion.level == level
        assert conclusion.source_ids == ["doc-1", "doc-2"]

    def test_explicit_conclusions_have_no_source_ids(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_documents([make_document("doc-1")])

        assert accumulator.build().conclusions[0].source_ids == []

    def test_carries_a_null_session_for_unscoped_conclusions(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_documents([make_document("doc-1", session_name=None)])

        assert accumulator.build().conclusions[0].session_id is None

    def test_reports_every_level(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_documents(
            [
                make_document("doc-1", level="explicit"),
                make_document("doc-2", level="deductive"),
                make_document("doc-3", level="inductive"),
                make_document("doc-4", level="contradiction"),
            ]
        )

        assert {c.level for c in accumulator.build().conclusions} == set(LEVELS)

    def test_orders_conclusions_by_derivation_time(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_documents(
            [
                make_document("doc-late", created_at=NOW + timedelta(hours=1)),
                make_document("doc-early", created_at=NOW),
            ]
        )

        assert [c.id for c in accumulator.build().conclusions] == [
            "doc-early",
            "doc-late",
        ]

    def test_dates_a_conclusion_from_its_source_messages(self):
        """The logical timestamp beats the row's insert time when it is recorded."""
        derived_from = NOW - timedelta(days=30)
        accumulator = EvidenceAccumulator()
        accumulator.add_documents(
            [
                make_document(
                    "doc-1",
                    created_at=NOW,
                    internal_metadata={"message_created_at": derived_from.isoformat()},
                )
            ]
        )

        # Seconds resolution: representations drop microseconds.
        assert accumulator.build().conclusions[0].created_at == derived_from


class TestMessageCollection:
    def test_records_identity_and_provenance(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_messages([make_message("msg-1", peer_name="bob")])

        (message,) = accumulator.build().messages
        assert message.id == "msg-1"
        assert message.peer_id == "bob"
        assert message.session_id == "session-1"

    def test_deduplicates_by_id(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_messages([make_message("msg-1"), make_message("msg-1")])
        accumulator.add_messages([make_message("msg-1")])

        assert [m.id for m in accumulator.build().messages] == ["msg-1"]

    @pytest.mark.parametrize(
        ("content", "expected_length"),
        [
            ("short", len("short")),
            ("x" * EVIDENCE_MESSAGE_PREVIEW_CHARS, EVIDENCE_MESSAGE_PREVIEW_CHARS),
            ("x" * 5000, EVIDENCE_MESSAGE_PREVIEW_CHARS),
        ],
        ids=["short", "exactly-at-the-cap", "long"],
    )
    def test_caps_the_preview_without_truncating_short_messages(
        self, content: str, expected_length: int
    ):
        accumulator = EvidenceAccumulator()
        accumulator.add_messages([make_message("msg-1", content=content)])

        preview = accumulator.build().messages[0].content_preview
        assert preview == content[:expected_length]
        assert len(preview) == expected_length

    def test_orders_messages_chronologically(self):
        accumulator = EvidenceAccumulator()
        accumulator.add_messages(
            [
                make_message("msg-late", created_at=NOW + timedelta(minutes=1)),
                make_message("msg-early", created_at=NOW),
            ]
        )

        assert [m.id for m in accumulator.build().messages] == [
            "msg-early",
            "msg-late",
        ]


class TestToolCallCollection:
    def test_reads_the_accumulated_loop_history(self):
        accumulator = EvidenceAccumulator()
        accumulator.record_tool_calls(
            [
                {
                    "tool_name": "search_memory",
                    "tool_input": {"query": "coffee"},
                    "tool_result": "Found 3 observations",
                    "tool_result_metadata": {"results_count": 3},
                }
            ]
        )

        (call,) = accumulator.build().tool_calls
        assert call.tool_name == "search_memory"
        assert call.tool_input == {"query": "coffee"}

    def test_reads_the_raw_provider_shape(self):
        """A provider response names the same fields `name`/`input`."""
        accumulator = EvidenceAccumulator()
        accumulator.record_tool_calls(
            [{"id": "call-1", "name": "grep_messages", "input": {"text": "coffee"}}]
        )

        (call,) = accumulator.build().tool_calls
        assert call.tool_name == "grep_messages"
        assert call.tool_input == {"text": "coffee"}

    def test_omits_tool_results(self):
        """Results are large and already reflected in conclusions and messages."""
        accumulator = EvidenceAccumulator()
        accumulator.record_tool_calls(
            [
                {
                    "tool_name": "search_memory",
                    "tool_input": {"query": "coffee"},
                    "tool_result": "SENTINEL-RESULT-TEXT",
                }
            ]
        )

        assert "SENTINEL-RESULT-TEXT" not in accumulator.build().model_dump_json()

    def test_preserves_call_order(self):
        accumulator = EvidenceAccumulator()
        accumulator.record_tool_calls(
            [
                {"tool_name": "search_memory", "tool_input": {}},
                {"tool_name": "search_messages", "tool_input": {}},
                {"tool_name": "search_memory", "tool_input": {}},
            ]
        )

        assert [c.tool_name for c in accumulator.build().tool_calls] == [
            "search_memory",
            "search_messages",
            "search_memory",
        ]

    def test_overwrites_rather_than_appends(self):
        """The tool loop rewrites its log wholesale on every exit path."""
        accumulator = EvidenceAccumulator()
        accumulator.record_tool_calls(
            [{"tool_name": "search_memory", "tool_input": {}}]
        )
        accumulator.record_tool_calls(
            [
                {"tool_name": "search_memory", "tool_input": {}},
                {"tool_name": "grep_messages", "tool_input": {}},
            ]
        )

        assert len(accumulator.build().tool_calls) == 2

    @pytest.mark.parametrize(
        "entry",
        [
            {"tool_name": "search_memory"},
            {"tool_name": "search_memory", "tool_input": None},
            {"tool_name": "search_memory", "tool_input": "not-a-dict"},
        ],
    )
    def test_tolerates_a_missing_or_malformed_input(self, entry: dict[str, Any]):
        accumulator = EvidenceAccumulator()
        accumulator.record_tool_calls([entry])

        assert accumulator.build().tool_calls[0].tool_input == {}


class TestTimestamps:
    def test_every_timestamp_names_its_timezone(self):
        """A caller should never have to guess what zone a timestamp is in.

        Conclusion timestamps arrive via `Representation`, which strips tzinfo
        so observations render compactly into prompts; message timestamps come
        straight off the column and keep theirs. Evidence has to be consistent.
        """
        accumulator = EvidenceAccumulator()
        accumulator.add_documents(
            [make_document(f"doc-{level}", level=level) for level in LEVELS]
        )
        accumulator.add_messages([make_message("msg-1")])
        evidence = accumulator.build()

        assert len(evidence.conclusions) == len(LEVELS)
        for conclusion in evidence.conclusions:
            assert conclusion.created_at.tzinfo is not None, conclusion.id
        for message in evidence.messages:
            assert message.created_at.tzinfo is not None, message.id


class TestEmptyEvidence:
    def test_builds_an_empty_object_rather_than_none(self):
        """An agent that read nothing still reports evidence, just empty.

        `evidence` being absent means the caller did not ask for it; an empty
        `evidence` means it was asked for and nothing was read. Callers can
        tell those apart, so a test asserting only `evidence is not None`
        proves nothing.
        """
        evidence = EvidenceAccumulator().build()

        assert evidence.conclusions == []
        assert evidence.messages == []
        assert evidence.tool_calls == []
        assert evidence.reasoning_trace_id is None

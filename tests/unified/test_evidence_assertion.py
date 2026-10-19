"""Unit tests for the `evidence_contains` evaluator, which needs no server."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from honcho.api_types import (
    Evidence,
    EvidenceMessageRef,
    EvidenceObservation,
    EvidenceToolCall,
)

from tests.unified import runner
from tests.unified.runner import ConclusionAttribution, evaluate_evidence
from tests.unified.schema import EvidenceContainsAssertion

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _conclusion(id_: str, content: str, session_id: str | None = None):
    return EvidenceObservation(
        id=id_, level="explicit", content=content, created_at=_T, session_id=session_id
    )


def _message(id_: str, peer_id: str, session_id: str):
    return EvidenceMessageRef(
        id=id_, session_id=session_id, peer_id=peer_id, created_at=_T
    )


_EVIDENCE = Evidence(
    conclusions=[
        _conclusion("c1", "Alice is a professional violinist", "s1"),
        _conclusion("c2", "Bob is a certified rescue diver", "s1"),
        _conclusion("c3", "Bob likes the Great Blue Hole"),
    ],
    messages=[_message("m1", "carol", "s1"), _message("m2", "bob", "s2")],
    tool_calls=[EvidenceToolCall(tool_name="list_peers")],
)
_ATTRIBUTION = {
    "c1": ConclusionAttribution(observer="alice", observed="alice"),
    "c2": ConclusionAttribution(observer="bob", observed="bob"),
}
_CONTENTS = {"m1": "I finished my third novel", "m2": "the vault code"}


def _check(**fields: object) -> None:
    evaluate_evidence(
        EvidenceContainsAssertion.model_validate(fields),
        _EVIDENCE,
        _ATTRIBUTION,
        _CONTENTS,
    )


def test_conclusions_match_is_case_insensitive_substring() -> None:
    _check(conclusions_match="VIOLIN")
    _check(conclusions_match="div")


def test_conclusions_from_peers_counts_distinct_observed_peers() -> None:
    _check(conclusions_from_peers=["alice", "bob", "frank"], min_count=2)
    with pytest.raises(runner.TestExecutionError, match="found \\['alice', 'bob'\\]"):
        _check(conclusions_from_peers=["alice", "bob", "frank"])


def test_unattributed_conclusions_do_not_count_toward_peers() -> None:
    evidence = Evidence(
        conclusions=[_conclusion("c3", "Bob likes the Great Blue Hole")]
    )
    with pytest.raises(runner.TestExecutionError, match="found \\[\\]"):
        evaluate_evidence(
            EvidenceContainsAssertion(conclusions_from_peers=["bob"]), evidence, {}, {}
        )


def test_messages_match_reads_fetched_content() -> None:
    _check(messages_match="novel")
    with pytest.raises(runner.TestExecutionError, match="no evidence message contains"):
        _check(messages_match="montmartre")


def test_not_from_sessions_checks_conclusions_and_messages() -> None:
    _check(not_from_sessions=["s9"])
    with pytest.raises(runner.TestExecutionError, match="excluded sessions \\['s2'\\]"):
        _check(not_from_sessions=["s2"])
    with pytest.raises(runner.TestExecutionError, match="excluded sessions \\['s1'\\]"):
        _check(conclusions_match="violin", not_from_sessions=["s1"])


def test_failure_message_lists_tools_peers_and_clipped_content() -> None:
    with pytest.raises(runner.TestExecutionError) as exc_info:
        _check(conclusions_match="tea")
    text = str(exc_info.value)
    assert "no evidence conclusion or peer card contains 'tea'" in text
    assert "tool_calls=['list_peers']" in text
    assert "conclusion alice (observer alice) explicit session=s1: Alice is" in text
    assert "conclusion unattributed explicit: Bob likes" in text
    assert "message peer=carol session=s1: I finished my third novel" in text


def test_all_conditions_are_reported_together() -> None:
    with pytest.raises(runner.TestExecutionError) as exc_info:
        _check(conclusions_match="tea", messages_match="cipher")
    text = str(exc_info.value)
    assert "'tea'" in text and "'cipher'" in text


def test_peer_cards_count_toward_conclusions_match() -> None:
    cards = {("bob", "bob"): ["Bob dives twice a month"]}
    evaluate_evidence(
        EvidenceContainsAssertion(conclusions_match="dives"),
        Evidence(tool_calls=[EvidenceToolCall(tool_name="get_peer_card")]),
        {},
        {},
        cards,
    )


def test_peer_cards_do_not_count_toward_conclusions_from_peers() -> None:
    cards = {("bob", "bob"): ["Bob dives twice a month"]}
    with pytest.raises(runner.TestExecutionError, match="found \\[\\]"):
        evaluate_evidence(
            EvidenceContainsAssertion(conclusions_from_peers=["bob"]),
            Evidence(tool_calls=[EvidenceToolCall(tool_name="get_peer_card")]),
            {},
            {},
            cards,
        )


def test_failure_message_lists_peer_cards() -> None:
    cards = {("bob", "bob"): ["Bob dives twice a month"]}
    with pytest.raises(runner.TestExecutionError) as exc_info:
        evaluate_evidence(
            EvidenceContainsAssertion(conclusions_match="tea"),
            Evidence(),
            {},
            {},
            cards,
        )
    assert "peer card bob (observer bob): Bob dives twice a month" in str(
        exc_info.value
    )

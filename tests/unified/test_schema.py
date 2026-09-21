"""Assert every unified JSON case still parses against the schema."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

_CASES = sorted(Path(__file__).parent.joinpath("test_cases").glob("*.json"))


@pytest.mark.parametrize("path", _CASES, ids=lambda p: p.name)
def test_unified_case_parses(path: Path) -> None:
    from tests.unified.schema import TestDefinition

    TestDefinition(**json.loads(path.read_text()))


def test_evidence_contains_parses_as_query_assertion() -> None:
    from tests.unified.schema import EvidenceContainsAssertion, QueryAction

    step = QueryAction(
        target="workspace_chat",
        input="q",
        assertions=[
            {  # pyright: ignore[reportArgumentType]
                "assertion_type": "evidence_contains",
                "conclusions_from_peers": ["dan", "emma", "frank"],
                "min_count": 2,
                "not_from_sessions": ["out_of_scope"],
            }
        ],
    )
    assertion = step.assertions[0]
    assert isinstance(assertion, EvidenceContainsAssertion)
    assert assertion.required_peer_count == 2


def test_evidence_contains_min_count_defaults_to_all_peers() -> None:
    from tests.unified.schema import EvidenceContainsAssertion

    assertion = EvidenceContainsAssertion(conclusions_from_peers=["a", "b"])
    assert assertion.required_peer_count == 2


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"min_count": 1},
        {"conclusions_from_peers": []},
        {"conclusions_from_peers": ["a"], "min_count": 2},
        {"conclusions_from_peers": ["a"], "min_count": 0},
    ],
    ids=["empty", "min_count_alone", "no_peers", "min_count_too_high", "zero"],
)
def test_evidence_contains_rejects_incoherent_fields(fields: dict[str, object]) -> None:
    from tests.unified.schema import EvidenceContainsAssertion

    with pytest.raises(ValidationError):
        EvidenceContainsAssertion.model_validate(fields)

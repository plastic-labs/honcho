"""Config that doesn't use the model catalog resolves exactly as before.

See `model_config_snapshot.py` for what each case covers and how the
snapshot was recorded.
"""

import json
from typing import Any

import pytest

from tests.llm.model_config_snapshot import CASES, SNAPSHOT_PATH, run_all


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    return run_all()


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text())


def test_snapshot_covers_every_case(recorded: dict[str, Any]) -> None:
    assert set(recorded) == set(CASES)


@pytest.mark.parametrize("case", sorted(CASES))
def test_case_matches_snapshot(
    case: str, current: dict[str, Any], recorded: dict[str, Any]
) -> None:
    assert current[case] == recorded[case]

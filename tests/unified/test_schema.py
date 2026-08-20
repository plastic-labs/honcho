"""Assert every unified JSON case still parses against the schema."""

import json
from pathlib import Path

import pytest

_CASES = sorted(Path(__file__).parent.joinpath("test_cases").glob("*.json"))


@pytest.mark.parametrize("path", _CASES, ids=lambda p: p.name)
def test_unified_case_parses(path: Path) -> None:
    from tests.unified.schema import TestDefinition

    TestDefinition(**json.loads(path.read_text()))

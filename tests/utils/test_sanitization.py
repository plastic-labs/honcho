from typing import Any

import pytest

from src.utils.sanitization import strip_nul


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("before\x00after", "beforeafter", id="string"),
        pytest.param("no nul here", "no nul here", id="string-unchanged"),
        pytest.param("\x00\x00", "", id="string-all-nul"),
        pytest.param(["a\x00b", "c"], ["ab", "c"], id="list"),
        pytest.param({"k\x00": "v\x00"}, {"k": "v"}, id="dict-key-and-value"),
        pytest.param(
            {"a": [{"b": "c\x00d"}]},
            {"a": [{"b": "cd"}]},
            id="nested",
        ),
        # Optional fields are passed in without a guard, so None has to survive.
        pytest.param(None, None, id="none"),
        pytest.param(7, 7, id="int"),
        pytest.param(True, True, id="bool"),
        pytest.param([], [], id="empty-list"),
    ],
)
def test_strip_nul(value: Any, expected: Any) -> None:
    assert strip_nul(value) == expected


def test_strip_nul_does_not_mutate_its_argument() -> None:
    original = {"a": ["b\x00c"]}

    stripped = strip_nul(original)

    assert stripped == {"a": ["bc"]}
    assert original == {"a": ["b\x00c"]}

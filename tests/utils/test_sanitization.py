from typing import Any

import pytest

from src.utils.sanitization import escape_unstorable, strip_nul


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("a\x00b", "a\\x00b", id="nul"),
        pytest.param("broken \ud83d emoji", "broken \\ud83d emoji", id="lone-high"),
        pytest.param("\udc00", "\\udc00", id="lone-low"),
        # Two surrogate code points side by side are still unencodable in a str.
        pytest.param("\ud83d\ude00", "\\ud83d\\ude00", id="split-pair"),
        pytest.param(
            "ok \U0001f600 \u2028 \x7f", "ok \U0001f600 \u2028 \x7f", id="storable"
        ),
        pytest.param("literal \\x00 text", "literal \\x00 text", id="literal-escape"),
        pytest.param("", "", id="empty"),
    ],
)
def test_escape_unstorable(value: str, expected: str) -> None:
    escaped = escape_unstorable(value)

    assert escaped == expected
    escaped.encode("utf-8")  # a surrogate would raise here
    assert "\x00" not in escaped

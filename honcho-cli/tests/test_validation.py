"""Tests for resource-ID validation.

Agents hallucinate IDs; ``validate_resource_id`` is the defense in depth
between that and the API. These tests pin the accept/reject rules.
"""

from __future__ import annotations

import pytest

from honcho_cli.validation import validate_resource_id


class TestAccepts:
    @pytest.mark.parametrize(
        "value",
        [
            "eri",
            "my-peer-01",
            "workspace_name",
            "UPPER",
            "with.dots",
            "123abc",
            "a",
            "long-id-with-many-parts_v2",
        ],
    )
    def test_safe_id_round_trips(self, value):
        assert validate_resource_id(value, "peer") == value


class TestRejects:
    def test_empty_string(self):
        with pytest.raises(SystemExit):
            validate_resource_id("", "peer")

    @pytest.mark.parametrize(
        "value",
        [
            "bad/slash",
            "bad\\backslash",
            "bad?query",
            "bad#hash",
            "bad%encoded",
            "with\x00null",
            "with\x1fctrl",
            "with\x7fdel",
        ],
    )
    def test_unsafe_chars(self, value):
        with pytest.raises(SystemExit):
            validate_resource_id(value, "peer")

    @pytest.mark.parametrize("value", ["..", "../etc", "foo/..", "a..b"])
    def test_path_traversal(self, value):
        with pytest.raises(SystemExit):
            validate_resource_id(value, "peer")

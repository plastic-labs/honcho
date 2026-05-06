"""Tests for src/utils/json_parser.py."""

import json

import pytest

from src.utils.json_parser import validate_and_repair_json


class TestValidateAndRepairJsonInputTypes:
    """Regression tests for non-string inputs.

    Some providers (Cloudflare Workers AI /compat for llama-4-scout)
    return chat.completions content as an already-parsed dict instead
    of a JSON-encoded string. The repair pipeline must JSON-encode the
    input first instead of crashing on .strip().
    """

    def test_dict_input_produces_same_result_as_equivalent_string(self):
        payload = {"explicit": [{"content": "user_alice exists"}]}
        result_from_dict = validate_and_repair_json(payload)
        result_from_str = validate_and_repair_json(json.dumps(payload))
        assert json.loads(result_from_dict) == json.loads(result_from_str)

    def test_list_input_does_not_crash(self):
        payload = [{"fact": "a"}, {"fact": "b"}]
        result = validate_and_repair_json(payload)
        assert json.loads(result) == payload

    def test_nested_dict_input(self):
        payload = {
            "explicit": [
                {"content": "fact one"},
                {"content": "fact two"},
            ],
            "implicit": [],
        }
        result = validate_and_repair_json(payload)
        assert json.loads(result) == payload

    def test_string_input_still_works(self):
        payload = '{"key": "value"}'
        result = validate_and_repair_json(payload)
        assert json.loads(result) == {"key": "value"}

    def test_string_input_with_whitespace_still_stripped(self):
        payload = '   {"key": "value"}   \n'
        result = validate_and_repair_json(payload)
        assert json.loads(result) == {"key": "value"}

import json

from src.utils.json_parser import extract_json_payload, validate_and_repair_json


def test_extract_json_payload_strips_think_tags_and_prose():
    raw = '<think>hidden reasoning</think>\nHere you go:\n{"explicit":[{"content":"I live in Berlin"}]}'
    extracted = extract_json_payload(raw)
    assert json.loads(extracted) == {"explicit": [{"content": "I live in Berlin"}]}


def test_validate_and_repair_json_extracts_fenced_json():
    raw = '```json\n{"explicit":[{"content":"I use Cubase"}]}\n```\nextra prose'
    repaired = validate_and_repair_json(raw)
    assert json.loads(repaired) == {"explicit": [{"content": "I use Cubase"}]}


def test_validate_and_repair_json_extracts_top_level_array():
    raw = 'analysis first\n["I live in Berlin", {"text":"I use Cubase"}]'
    repaired = validate_and_repair_json(raw)
    assert json.loads(repaired) == ["I live in Berlin", {"text": "I use Cubase"}]


def test_extract_json_payload_prefers_json_fence_over_earlier_non_json_fence():
    raw = '```text\nnot json\n```\n```json\n{"explicit":[{"content":"I live in Berlin"}]}\n```'
    extracted = extract_json_payload(raw)
    assert json.loads(extracted) == {"explicit": [{"content": "I live in Berlin"}]}


def test_extract_json_payload_prefers_payload_over_earlier_schema_object():
    raw = 'Schema: {"type":"object"}\nActual: {"explicit":[{"content":"I use Cubase"}]}'
    extracted = extract_json_payload(raw)
    assert json.loads(extracted) == {"explicit": [{"content": "I use Cubase"}]}


def test_extract_json_payload_preserves_literal_think_text_inside_valid_json():
    raw = '{"content":"literal <think> tag","explicit":[]}'
    extracted = extract_json_payload(raw)
    assert json.loads(extracted) == {"content": "literal <think> tag", "explicit": []}


def test_extract_json_payload_prefers_later_array_payload_over_earlier_schema_dict():
    raw = 'Schema: {"type":"object"}\nActual: [{"content":"I use Cubase"}]'
    extracted = extract_json_payload(raw)
    assert json.loads(extracted) == [{"content": "I use Cubase"}]

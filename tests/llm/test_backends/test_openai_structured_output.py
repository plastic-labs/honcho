"""Unit tests for the OpenAI-backend structured-output helpers."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from src.llm.backends.openai import (
    _inject_schema_hint,  # pyright: ignore[reportPrivateUsage]
    _strip_reasoning_and_fences,  # pyright: ignore[reportPrivateUsage]
)


class _Obs(BaseModel):
    facts: list[str] = []


# --------------------------------------------------------------------------- #
# _strip_reasoning_and_fences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        # empty <think></think> + ```json fence (typical local llama.cpp output)
        ('<think>\n\n</think>\n\n```json\n{"a": 1}\n```', '{"a": 1}'),
        # closed reasoning block, JSON, then trailing prose
        ('<think>r</think>{"ok": true} done.', '{"ok": true}'),
        # bare ``` fence without the json language tag
        ('```\n{"z": 9}\n```', '{"z": 9}'),
        # already-valid object / array — fast path, returned untouched
        ('{"facts": ["x", "y"]}', '{"facts": ["x", "y"]}'),
        ('[{"a": 1}, {"b": 2}]', '[{"a": 1}, {"b": 2}]'),
        # brace/bracket inside a string value must not confuse extraction
        ('{"n": "a } b ] c"}', '{"n": "a } b ] c"}'),
        # C-1 regression: a string value literally containing "<think>" must be
        # preserved (fast path runs before any <think> stripping)
        ('{"note": "a <think> tag in text"}', '{"note": "a <think> tag in text"}'),
        # C1: object followed by prose containing the OTHER bracket type
        ('{"k": "v"} list: [1, 2]', '{"k": "v"}'),
        # M1: object followed by prose containing the SAME closing bracket
        ('{"items": [1,2,3]} the value is 4}', '{"items": [1,2,3]}'),
        # array first, trailing prose containing a stray "}"
        ('[1, 2] and a dict: {"k": "v"}', "[1, 2]"),
        # unclosed <think> (truncated) followed by recoverable JSON
        ('<think>partial reasoning\n{"ok": true}', '{"ok": true}'),
        # pure prose with no JSON is returned unchanged
        ("I cannot help with that.", "I cannot help with that."),
        # None / empty
        (None, ""),
        ("", ""),
    ],
)
def test_strip_reasoning_and_fences(raw: str | None, expected: str) -> None:
    assert _strip_reasoning_and_fences(raw) == expected


# --------------------------------------------------------------------------- #
# _inject_schema_hint
# --------------------------------------------------------------------------- #
def test_inject_appends_to_existing_system_message():
    messages = [
        {"role": "system", "content": "Extract facts."},
        {"role": "user", "content": "hi"},
    ]
    out = _inject_schema_hint(messages, _Obs)
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("Extract facts.")
    assert "JSON schema" in out[0]["content"]
    assert "facts" in out[0]["content"]  # schema body present
    assert out[1] == {"role": "user", "content": "hi"}


def test_inject_inserts_system_when_absent():
    messages = [{"role": "user", "content": "hi"}]
    out = _inject_schema_hint(messages, _Obs)
    assert out[0]["role"] == "system"
    assert "JSON schema" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_inject_targets_system_even_when_not_first():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "system", "content": "sys"},
    ]
    out = _inject_schema_hint(messages, _Obs)
    assert out[0] == {"role": "user", "content": "u1"}
    assert out[1]["role"] == "system"
    assert out[1]["content"].startswith("sys")
    assert "facts" in out[1]["content"]


def test_inject_into_multimodal_system_content_appends_text_part():
    messages = [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
    out = _inject_schema_hint(messages, _Obs)
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "sys"}
    assert content[-1]["type"] == "text"
    assert "facts" in content[-1]["text"]


def test_inject_does_not_mutate_caller_messages():
    messages = [{"role": "system", "content": "orig"}]
    snapshot = {"role": "system", "content": "orig"}
    _inject_schema_hint(messages, _Obs)
    assert messages[0] == snapshot  # caller's list/dicts untouched


def test_inject_serialize_failure_logs_and_passes_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BadSchema:
        @staticmethod
        def model_json_schema() -> dict[str, object]:
            raise RuntimeError("boom")

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.WARNING):
        out = _inject_schema_hint(messages, BadSchema)  # pyright: ignore[reportArgumentType]
    assert out == messages  # no hint added, call proceeds
    assert "without a schema hint" in caplog.text

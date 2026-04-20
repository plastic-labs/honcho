from typing import Any

from src.llm.conversation import (
    _is_tool_result_message,  # pyright: ignore[reportPrivateUsage]
    _is_tool_use_message,  # pyright: ignore[reportPrivateUsage]
    truncate_messages_to_fit,
)


def test_truncate_messages_to_fit_keeps_last_unit_when_over_limit() -> None:
    messages = [
        {"role": "user", "content": "x " * 2000},
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=1)

    assert truncated == messages


def test_truncate_messages_to_fit_preserves_tool_result_pair() -> None:
    messages = [
        {"role": "user", "content": "old context " * 1000},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=5)

    assert truncated == messages[1:]


def test_is_tool_use_message_detects_gemini_function_call_in_parts() -> None:
    msg: dict[str, Any] = {
        "role": "model",
        "parts": [
            {"function_call": {"name": "search", "args": {"q": "honcho"}}},
        ],
    }
    assert _is_tool_use_message(msg) is True


def test_is_tool_result_message_detects_gemini_function_response_in_parts() -> None:
    msg: dict[str, Any] = {
        "role": "user",
        "parts": [
            {"function_response": {"name": "search", "response": {"result": "ok"}}},
        ],
    }
    assert _is_tool_result_message(msg) is True


def test_is_tool_use_message_detects_anthropic_tool_use_block() -> None:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "calling lookup"},
            {"type": "tool_use", "id": "t_1", "name": "lookup", "input": {}},
        ],
    }
    assert _is_tool_use_message(msg) is True


def test_truncate_messages_to_fit_preserves_gemini_tool_pair() -> None:
    """A Gemini-shaped function_call / function_response pair must stay
    grouped when older units get dropped. Regression: before adding the
    parts-based detection, neither message would be recognized as a tool
    unit, and truncation could split or drop them individually."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "parts": [{"text": "old context " * 1000}]},
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "lookup", "args": {}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "lookup",
                        "response": {"result": "found"},
                    }
                }
            ],
        },
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=20)

    # The oldest (bulk-text) message should be dropped; the function_call +
    # function_response pair stays intact together.
    assert truncated == messages[1:]

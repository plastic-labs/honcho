from src.llm.conversation import truncate_messages_to_fit


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

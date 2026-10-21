import json
from typing import Any

import pytest

from src.llm.conversation import (
    _is_tool_result_message,  # pyright: ignore[reportPrivateUsage]
    _is_tool_use_message,  # pyright: ignore[reportPrivateUsage]
    count_message_tokens,
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


_HUGE_RESULT = "row, " * 50_000


_TOOL_PAIRS: list[tuple[dict[str, Any], dict[str, Any]]] = [
    (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": _HUGE_RESULT},
    ),
    (
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t_1", "name": "grep", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t_1",
                    "content": _HUGE_RESULT,
                }
            ],
        },
    ),
    (
        {
            "role": "model",
            "parts": [{"function_call": {"name": "grep", "args": {}}}],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "grep",
                        "response": {"result": _HUGE_RESULT},
                    }
                }
            ],
        },
    ),
]


@pytest.mark.parametrize(
    ("tool_use", "tool_result"), _TOOL_PAIRS, ids=["openai", "anthropic", "gemini"]
)
def test_truncate_messages_to_fit_clips_oversized_newest_tool_result(
    tool_use: dict[str, Any], tool_result: dict[str, Any]
) -> None:
    """One tool result bigger than the whole budget used to be sent whole, after
    the query that started the conversation was dropped to make room for it."""
    query = {"role": "user", "content": "What does the user like? " * 20}
    messages = [
        {"role": "system", "content": "You are a helper."},
        query,
        tool_use,
        tool_result,
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=2_000)

    assert count_message_tokens(truncated) <= 2_000
    assert truncated[:3] == messages[:3]
    assert "[TRUNCATED to fit the input budget" in json.dumps(truncated[3])
    # Only the payload shrank: the result still answers the same tool call.
    assert _is_tool_result_message(truncated[3])
    assert {k for k in truncated[3] if k not in ("content", "parts")} == {
        k for k in tool_result if k not in ("content", "parts")
    }
    # The caller's list is untouched.
    assert messages[3] is tool_result
    assert _HUGE_RESULT in json.dumps(tool_result)


_SYSTEM = {"role": "system", "content": "You are a helper."}
_QUERY = {"role": "user", "content": "What does the user like? " * 20}


def _openai_round(call_id: str, results: list[str]) -> list[dict[str, Any]]:
    """One assistant turn calling len(results) tools, plus their results."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"{call_id}_{i}",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{}"},
                }
                for i in range(len(results))
            ],
        },
        *(
            {"role": "tool", "tool_call_id": f"{call_id}_{i}", "content": result}
            for i, result in enumerate(results)
        ),
    ]


def test_truncate_messages_to_fit_drops_middle_rounds_before_the_query() -> None:
    """Earlier tool rounds give way, oldest first, before the opening query does."""
    oldest = _openai_round("oldest", ["oldest evidence " * 800])
    older = _openai_round("older", ["older evidence " * 800])
    newest = _openai_round("new", [_HUGE_RESULT])
    messages = [_SYSTEM, _QUERY, *oldest, *older, *newest]

    truncated = truncate_messages_to_fit(messages, max_tokens=4_000)

    assert count_message_tokens(truncated) <= 4_000
    assert truncated == [_SYSTEM, _QUERY, *older, newest[0], truncated[-1]]
    assert "[TRUNCATED to fit the input budget" in truncated[-1]["content"]


def test_truncate_messages_to_fit_keeps_evidence_behind_a_closing_prompt() -> None:
    """The tool loop appends a closing synthesis prompt after the last round; the
    oversized round before it is clipped rather than dropped wholesale."""
    newest = _openai_round("new", [_HUGE_RESULT])
    closing = {"role": "user", "content": "Answer now with what you have."}
    messages = [_SYSTEM, _QUERY, *newest, closing]

    truncated = truncate_messages_to_fit(messages, max_tokens=2_000)

    assert count_message_tokens(truncated) <= 2_000
    assert truncated[:3] == [_SYSTEM, _QUERY, newest[0]]
    assert truncated[3]["content"].startswith("row, row, ")
    assert truncated[-1] == closing


def test_truncate_messages_to_fit_clips_many_parallel_results() -> None:
    """Every payload can be cut, not just a fixed number of the largest."""
    newest = _openai_round("new", ["row, " * 8_000] * 16)
    messages = [_SYSTEM, _QUERY, *newest]

    truncated = truncate_messages_to_fit(messages, max_tokens=5_000)

    assert count_message_tokens(truncated) <= 5_000
    assert truncated[:2] == [_SYSTEM, _QUERY]
    assert len(truncated) == len(messages)


def test_truncate_messages_to_fit_sizes_nested_payloads_as_counted() -> None:
    """Nested payloads are counted as JSON, where non-ASCII text is escaped and
    costs several times its raw tokens; the cut must not overshoot because of it."""
    payload = "用户喜欢在周末徒步旅行和摄影。" * 6_000
    messages: list[dict[str, Any]] = [
        _SYSTEM,
        _QUERY,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t_1", "name": "grep", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t_1", "content": payload}
            ],
        },
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=4_000)

    tokens = count_message_tokens(truncated)
    # One result is capped at half the budget; overshooting the cut because of
    # escaping would land far below that.
    assert 1_500 < tokens <= 2_000


def test_truncate_messages_to_fit_runaway_result_does_not_displace_earlier_rounds() -> (
    None
):
    """One runaway result is cut down on its own; earlier rounds that fit
    alongside a capped copy of it are kept."""
    alpha = _openai_round("alpha", ["alpha " * 500])
    beta = _openai_round("beta", ["beta " * 500])
    newest = _openai_round("new", [_HUGE_RESULT])
    messages = [_SYSTEM, _QUERY, *alpha, *beta, *newest]

    truncated = truncate_messages_to_fit(messages, max_tokens=4_000)

    assert count_message_tokens(truncated) <= 4_000
    assert truncated[: 2 + len(alpha) + len(beta)] == [_SYSTEM, _QUERY, *alpha, *beta]
    assert "[TRUNCATED to fit the input budget" in truncated[-1]["content"]


def test_truncate_keeps_a_budget_line_with_its_tool_round() -> None:
    """A plain user turn after the tool results (the tool loop's budget line)
    stays in the round's unit, so it is kept or clipped with the results it
    describes rather than standing alone as the newest unit."""
    tool_use, tool_result = _TOOL_PAIRS[0]
    budget_line = {"role": "user", "content": "Tool calls used: 1 of 10."}
    messages = [_SYSTEM, _QUERY, tool_use, tool_result, budget_line]

    truncated = truncate_messages_to_fit(messages, max_tokens=2_000)

    assert count_message_tokens(truncated) <= 2_000
    assert truncated[:3] == messages[:3]
    assert _is_tool_result_message(truncated[3])
    assert "[TRUNCATED to fit the input budget" in json.dumps(truncated[3])
    assert truncated[4] == budget_line

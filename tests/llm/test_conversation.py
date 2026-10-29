from src.llm.conversation import (
    count_message_tokens,
    strip_thinking_blocks,
    truncate_messages_to_fit,
)
from src.llm.messages import (
    AIMessage,
    ChatMessage,
    ModelOrigin,
    ReasoningPart,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_SYSTEM = SystemMessage(content="You are a helper.")
_QUERY = UserMessage(content="What does the user like? " * 20)
_HUGE_RESULT = "row, " * 50_000
_CLAUDE = ModelOrigin(transport="anthropic", model="claude-sonnet-5-5")


def _round(call_id: str, results: list[str]) -> list[ChatMessage]:
    """One assistant turn calling len(results) tools, plus their results."""
    return [
        AIMessage(
            tool_calls=[
                ToolCall(id=f"{call_id}_{i}", name="grep", input={})
                for i in range(len(results))
            ]
        ),
        *(
            ToolMessage(tool_call_id=f"{call_id}_{i}", name="grep", content=result)
            for i, result in enumerate(results)
        ),
    ]


def _clipped(message: ChatMessage) -> bool:
    return isinstance(message, ToolMessage) and (
        "[TRUNCATED to fit the input budget" in message.content
    )


def test_truncate_messages_to_fit_keeps_last_unit_when_over_limit() -> None:
    messages: list[ChatMessage] = [UserMessage(content="x " * 2000)]

    truncated = truncate_messages_to_fit(messages, max_tokens=1)

    assert truncated == messages


def test_truncate_messages_to_fit_preserves_tool_result_pair() -> None:
    messages: list[ChatMessage] = [
        UserMessage(content="old context " * 1000),
        *_round("call", ["result"]),
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=20)

    assert truncated == messages[1:]


def test_truncate_messages_to_fit_clips_oversized_newest_tool_result() -> None:
    """One tool result bigger than the whole budget used to be sent whole, after
    the query that started the conversation was dropped to make room for it."""
    tool_round = _round("call", [_HUGE_RESULT])
    messages: list[ChatMessage] = [_SYSTEM, _QUERY, *tool_round]

    truncated = truncate_messages_to_fit(messages, max_tokens=2_000)

    assert count_message_tokens(truncated) <= 2_000
    assert truncated[:3] == messages[:3]
    # Only the payload shrank: the result still answers the same tool call.
    result = truncated[3]
    assert _clipped(result)
    assert isinstance(result, ToolMessage)
    assert (result.tool_call_id, result.name) == ("call_0", "grep")
    # The caller's list is untouched.
    assert messages[3] is tool_round[1]


def test_truncate_messages_to_fit_drops_middle_rounds_before_the_query() -> None:
    """Earlier tool rounds give way, oldest first, before the opening query does."""
    oldest = _round("oldest", ["oldest evidence " * 800])
    older = _round("older", ["older evidence " * 800])
    newest = _round("new", [_HUGE_RESULT])
    messages: list[ChatMessage] = [_SYSTEM, _QUERY, *oldest, *older, *newest]

    truncated = truncate_messages_to_fit(messages, max_tokens=4_000)

    assert count_message_tokens(truncated) <= 4_000
    assert truncated == [_SYSTEM, _QUERY, *older, newest[0], truncated[-1]]
    assert _clipped(truncated[-1])


def test_truncate_messages_to_fit_keeps_evidence_behind_a_closing_prompt() -> None:
    """The tool loop appends a closing synthesis prompt after the last round; the
    oversized round before it is clipped rather than dropped wholesale."""
    newest = _round("new", [_HUGE_RESULT])
    closing = UserMessage(content="Answer now with what you have.")
    messages: list[ChatMessage] = [_SYSTEM, _QUERY, *newest, closing]

    truncated = truncate_messages_to_fit(messages, max_tokens=2_000)

    assert count_message_tokens(truncated) <= 2_000
    assert truncated[:3] == [_SYSTEM, _QUERY, newest[0]]
    result = truncated[3]
    assert isinstance(result, ToolMessage)
    assert result.content.startswith("row, row, ")
    assert truncated[-1] == closing


def test_truncate_messages_to_fit_clips_many_parallel_results() -> None:
    """Every payload can be cut, not just a fixed number of the largest."""
    newest = _round("new", ["row, " * 8_000] * 16)
    messages: list[ChatMessage] = [_SYSTEM, _QUERY, *newest]

    truncated = truncate_messages_to_fit(messages, max_tokens=5_000)

    assert count_message_tokens(truncated) <= 5_000
    assert truncated[:2] == [_SYSTEM, _QUERY]
    assert len(truncated) == len(messages)


def test_truncate_messages_to_fit_cuts_non_ascii_payloads_to_the_cap() -> None:
    """A payload whose characters cost several tokens each is cut to its cap
    without overshooting below it."""
    payload = "用户喜欢在周末徒步旅行和摄影。" * 6_000
    messages: list[ChatMessage] = [_SYSTEM, _QUERY, *_round("t", [payload])]

    truncated = truncate_messages_to_fit(messages, max_tokens=4_000)

    tokens = count_message_tokens(truncated)
    # One result is capped at half the budget.
    assert 1_500 < tokens <= 2_000


def test_truncate_messages_to_fit_runaway_result_does_not_displace_earlier_rounds() -> (
    None
):
    """One runaway result is cut down on its own; earlier rounds that fit
    alongside a capped copy of it are kept."""
    alpha = _round("alpha", ["alpha " * 500])
    beta = _round("beta", ["beta " * 500])
    newest = _round("new", [_HUGE_RESULT])
    messages: list[ChatMessage] = [_SYSTEM, _QUERY, *alpha, *beta, *newest]

    truncated = truncate_messages_to_fit(messages, max_tokens=4_000)

    assert count_message_tokens(truncated) <= 4_000
    assert truncated[: 2 + len(alpha) + len(beta)] == [_SYSTEM, _QUERY, *alpha, *beta]
    assert _clipped(truncated[-1])


def test_truncate_keeps_a_budget_line_with_its_tool_round() -> None:
    """A plain user turn after the tool results (the tool loop's budget line)
    stays in the round's unit, so it is kept or clipped with the results it
    describes rather than standing alone as the newest unit."""
    budget_line = UserMessage(content="Tool calls used: 1 of 10.")
    messages: list[ChatMessage] = [
        _SYSTEM,
        _QUERY,
        *_round("call", [_HUGE_RESULT]),
        budget_line,
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=2_000)

    assert count_message_tokens(truncated) <= 2_000
    assert truncated[:3] == messages[:3]
    assert _clipped(truncated[3])
    assert truncated[4] == budget_line


def test_orphaned_tool_turns_are_dropped_by_truncation() -> None:
    """A tool call with no result, or a result with no call, cannot be sent."""
    messages: list[ChatMessage] = [
        UserMessage(content="old context " * 1000),
        ToolMessage(tool_call_id="stray", name="grep", content="x"),
        AIMessage(tool_calls=[ToolCall(id="lost", name="grep", input={})]),
        UserMessage(content="now"),
    ]

    truncated = truncate_messages_to_fit(messages, max_tokens=20)

    assert truncated == [UserMessage(content="now")]


def test_strip_thinking_blocks_keeps_other_content_and_drops_emptied_turns() -> None:
    user = UserMessage(content="hi")
    tool_turn = AIMessage(
        tool_calls=[ToolCall(id="t1", name="search", input={})],
        reasoning=[
            ReasoningPart(
                kind="anthropic_thinking",
                payload={"type": "thinking", "thinking": "plan", "signature": "s"},
                origin=_CLAUDE,
            ),
            ReasoningPart(
                kind="anthropic_redacted_thinking",
                payload={"type": "redacted_thinking", "data": "opaque"},
                origin=_CLAUDE,
            ),
        ],
    )
    thinking_only = AIMessage(
        reasoning=[
            ReasoningPart(
                kind="anthropic_thinking",
                payload={"type": "thinking", "thinking": "x", "signature": "s"},
                origin=_CLAUDE,
            )
        ]
    )

    stripped, removed = strip_thinking_blocks([user, tool_turn, thinking_only])

    assert removed == 3
    assert stripped[0] is user
    assert stripped[1] == AIMessage(
        tool_calls=[ToolCall(id="t1", name="search", input={})]
    )
    assert len(stripped) == 2
    # The input turn is not modified.
    assert len(tool_turn.reasoning) == 2

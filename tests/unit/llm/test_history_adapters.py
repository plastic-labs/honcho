from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.history_adapters import (
    AnthropicHistoryAdapter,
    GeminiHistoryAdapter,
    OpenAIHistoryAdapter,
)


def test_anthropic_history_adapter_preserves_thinking_blocks() -> None:
    adapter = AnthropicHistoryAdapter()
    result = CompletionResult(
        content="Done",
        thinking_blocks=[
            {
                "type": "thinking",
                "thinking": "private reasoning",
                "signature": "sig_123",
            }
        ],
        tool_calls=[
            ToolCallResult(id="tool_1", name="search", input={"query": "honcho"})
        ],
    )

    message = adapter.format_assistant_tool_message(result)

    assert message["role"] == "assistant"
    assert message["content"][0]["type"] == "thinking"
    assert message["content"][1] == {"type": "text", "text": "Done"}
    assert message["content"][2]["type"] == "tool_use"


def test_gemini_history_adapter_preserves_thought_signature() -> None:
    adapter = GeminiHistoryAdapter()
    result = CompletionResult(
        content="Calling a tool",
        tool_calls=[
            ToolCallResult(
                id="tool_1",
                name="search",
                input={"query": "honcho"},
                thought_signature="sig_abc",
            )
        ],
    )

    message = adapter.format_assistant_tool_message(result)

    assert message["role"] == "model"
    assert message["parts"][1]["thought_signature"] == "sig_abc"


def test_openai_history_adapter_preserves_reasoning_details() -> None:
    adapter = OpenAIHistoryAdapter()
    result = CompletionResult(
        content="Calling a tool",
        reasoning_details=[{"type": "reasoning", "content": "step 1"}],
        tool_calls=[
            ToolCallResult(id="tool_1", name="search", input={"query": "honcho"})
        ],
    )

    message = adapter.format_assistant_tool_message(result)

    assert message["role"] == "assistant"
    assert message["reasoning_details"] == [{"type": "reasoning", "content": "step 1"}]
    assert message["tool_calls"][0]["function"]["name"] == "search"

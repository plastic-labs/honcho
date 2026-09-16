"""A terminal tool failure removes only that tool from the current loop."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.config import ModelConfig, ModelTransport
from src.llm import tool_loop
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse, ProviderClient
from src.utils.agent_tools import (
    MAX_PEER_CARD_FACTS,
    TOOLS,
    ToolContext,
    _handle_update_peer_card,  # pyright: ignore[reportPrivateUsage]
    create_tool_executor,
)

CARD_TOOL = "update_peer_card"
OTHER_TOOL = "search"
FINAL_CONTENT = "Card update not applied; other work finished."
OTHER_RESULT = "Search completed."


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("include_other_tool", [True, False])
@pytest.mark.parametrize("same_response", [True, False])
async def test_two_card_failures_remove_tool_without_stopping_other_work(
    provider: ModelTransport, include_other_tool: bool, same_response: bool
) -> None:
    plan = AttemptPlan(
        provider=provider,
        model="test-model",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(model="test-model", transport=provider),
        attempt=1,
        retry_attempts=1,
        is_fallback=False,
    )
    over_cap = [f"IDENTITY: Aliases: alias-{i}" for i in range(MAX_PEER_CARD_FACTS + 1)]
    tools = [TOOLS[CARD_TOOL]]
    if include_other_tool:
        tools.append(
            {
                "name": OTHER_TOOL,
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        )
    original_tools = deepcopy(tools)
    requests: list[dict[str, Any]] = []
    executed: list[str] = []

    async def search(_ctx: ToolContext, _input: dict[str, Any]) -> str:
        return OTHER_RESULT

    handlers = {CARD_TOOL: _handle_update_peer_card, OTHER_TOOL: search}
    executor = await create_tool_executor(
        "test-workspace", "observer", "observed", handler_resolver=handlers.get
    )

    async def recording_executor(name: str, payload: dict[str, Any]) -> str:
        executed.append(name)
        return await executor(name, payload)

    def card_call(call_id: str) -> dict[str, Any]:
        return {"id": call_id, "name": CARD_TOOL, "input": {"content": over_cap}}

    terminal_calls = [card_call("second"), card_call("third-in-same-response")]
    if include_other_tool:
        terminal_calls.append({"id": "other", "name": OTHER_TOOL, "input": {}})
    scripted_responses: list[HonchoLLMCallResponse[Any]] = []
    if same_response:
        terminal_calls.insert(0, card_call("first"))
    else:
        scripted_responses.append(
            HonchoLLMCallResponse(
                content="",
                output_tokens=1,
                finish_reasons=["tool_calls"],
                tool_calls_made=[card_call("first")],
            )
        )
    scripted_responses.extend(
        [
            HonchoLLMCallResponse(
                content="",
                output_tokens=1,
                finish_reasons=["tool_calls"],
                tool_calls_made=terminal_calls,
            ),
            HonchoLLMCallResponse(
                content=FINAL_CONTENT,
                output_tokens=1,
                finish_reasons=["stop"],
                tool_calls_made=[],
            ),
        ]
    )
    responses = iter(scripted_responses)

    async def fake_call(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        requests.append(deepcopy(kwargs))
        return next(responses)

    with patch.object(tool_loop, "honcho_llm_call_inner", new=fake_call):
        result = await execute_tool_loop(
            prompt="Update the card, then do other work.",
            max_tokens=256,
            messages=None,
            tools=tools,
            tool_choice={"type": "tool", "name": CARD_TOOL},
            tool_executor=recording_executor,
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=False,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=lambda: plan,
            before_retry_callback=lambda _state: None,
        )

    assert isinstance(result, HonchoLLMCallResponse)
    assert result.content == FINAL_CONTENT
    expected_tools = [OTHER_TOOL] if include_other_tool else []
    remaining_tools = cast(list[dict[str, Any]], requests[-1]["tools"] or [])
    assert [tool["name"] for tool in remaining_tools] == expected_tools
    expected_choice = "auto" if include_other_tool else None
    assert requests[-1]["tool_choice"] == expected_choice
    assert tools == original_tools
    assert executed == [CARD_TOOL, CARD_TOOL, *expected_tools]
    expected_call_count = 4 if include_other_tool else 3
    assert len(result.tool_calls_made) == expected_call_count
    if include_other_tool:
        assert result.tool_calls_made[-1]["tool_result"] == OTHER_RESULT

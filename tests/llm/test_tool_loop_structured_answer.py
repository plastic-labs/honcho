# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""`execute_tool_loop` retries a final answer that misses `response_model`."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from tenacity import wait_none

from src.llm import tool_loop
from src.llm.errors import llm_retry_condition
from src.llm.runtime import AttemptPlan, current_attempt
from src.llm.structured_output import StructuredOutputError
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import HonchoLLMCallResponse, StreamingResponseWithMetadata

_TOOLS: list[dict[str, Any]] = [
    {"name": "recall", "description": "recall", "input_schema": {}}
]
_PRIMARY = "primary-model"
_FALLBACK = "fallback-model"
_REFUSAL = "I'm sorry, I can't help with that."


class _Answer(BaseModel):
    answer: str


def _plan_for_current_attempt(retry_attempts: int) -> AttemptPlan:
    """Mirror `honcho_llm_call`: the last attempt swaps to the fallback."""
    attempt = current_attempt.get()
    is_fallback = attempt == retry_attempts
    return AttemptPlan(
        provider="openai",
        model=_FALLBACK if is_fallback else _PRIMARY,
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=None,
        attempt=attempt,
        retry_attempts=retry_attempts,
        is_fallback=is_fallback,
    )


def _response(
    content: Any, tool_names: tuple[str, ...] = ()
) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        finish_reasons=["tool_use" if tool_names else "stop"],
        tool_calls_made=[
            {"id": f"t{i}", "name": name, "input": {}}
            for i, name in enumerate(tool_names)
        ],
    )


class _ScriptedModel:
    """Answers per call from a script keyed by the attempt plan's model."""

    def __init__(self, script: list[dict[str, HonchoLLMCallResponse[Any]]]) -> None:
        self.script: list[dict[str, HonchoLLMCallResponse[Any]]] = list(script)
        self.models: list[str] = []
        self.messages: list[list[dict[str, Any]]] = []

    async def __call__(self, *_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[Any]:
        model = kwargs["plan"].model
        self.models.append(model)
        self.messages.append(list(kwargs["messages"]))
        turn = self.script[0]
        if len(self.script) > 1:
            self.script.pop(0)
        return turn[model] if model in turn else turn["*"]


async def _echo_executor(name: str, _input: dict[str, Any]) -> str:
    return f"{name} result"


async def _run_loop(
    model: _ScriptedModel,
    *,
    retry_attempts: int = 3,
    max_tool_iterations: int = 5,
    stream_final: bool = False,
) -> HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata:
    """Run the tool loop with `model` standing in for the provider, retrying with no
    backoff.
    """

    def before_retry(retry_state: Any) -> None:
        current_attempt.set(retry_state.attempt_number + 1)

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", new=model),
        patch.object(tool_loop, "wait_exponential", lambda **_kw: wait_none()),
    ):
        result = await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "q"}],
            tools=_TOOLS,
            tool_choice=None,
            tool_executor=_echo_executor,
            max_tool_iterations=max_tool_iterations,
            response_model=_Answer,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=True,
            retry_attempts=retry_attempts,
            max_input_tokens=None,
            get_attempt_plan=lambda: _plan_for_current_attempt(retry_attempts),
            before_retry_callback=before_retry,
            retry_condition=llm_retry_condition(
                can_fall_back=True, retry_attempts=retry_attempts, model="test/model"
            ),
            stream_final=stream_final,
            telemetry=None,
        )
    return result


async def _run(
    model: _ScriptedModel, *, retry_attempts: int = 3, max_tool_iterations: int = 5
) -> HonchoLLMCallResponse[Any]:
    result = await _run_loop(
        model, retry_attempts=retry_attempts, max_tool_iterations=max_tool_iterations
    )
    assert isinstance(result, HonchoLLMCallResponse)
    return result


@pytest.mark.asyncio
async def test_refusal_is_retried_then_raises() -> None:
    model = _ScriptedModel([{"*": _response(_REFUSAL)}])

    with pytest.raises(StructuredOutputError):
        await _run(model, retry_attempts=3)

    assert model.models == [_PRIMARY, _PRIMARY, _FALLBACK]


@pytest.mark.asyncio
async def test_refusal_on_primary_succeeds_on_fallback() -> None:
    model = _ScriptedModel(
        [{_PRIMARY: _response(_REFUSAL), _FALLBACK: _response(_Answer(answer="ok"))}]
    )

    result = await _run(model, retry_attempts=3)

    assert result.content == _Answer(answer="ok")
    assert model.models == [_PRIMARY, _PRIMARY, _FALLBACK]


@pytest.mark.asyncio
async def test_empty_answer_gets_the_nudge_before_it_counts_as_a_miss() -> None:
    model = _ScriptedModel(
        [{"*": _response("")}, {"*": _response(_Answer(answer="ok"))}]
    )

    result = await _run(model, retry_attempts=3)

    assert result.content == _Answer(answer="ok")
    # One call per iteration: the empty answer went to the nudge, not a retry.
    assert model.models == [_PRIMARY, _PRIMARY]
    assert "Your last response was empty" in model.messages[1][-1]["content"]


@pytest.mark.asyncio
async def test_empty_answer_after_the_nudge_is_retried() -> None:
    model = _ScriptedModel(
        [
            {"*": _response("")},
            {_PRIMARY: _response(""), _FALLBACK: _response(_Answer(answer="ok"))},
        ]
    )

    result = await _run(model, retry_attempts=3)

    assert result.content == _Answer(answer="ok")
    assert model.models == [_PRIMARY, _PRIMARY, _PRIMARY, _FALLBACK]


@pytest.mark.asyncio
async def test_synthesis_refusal_is_retried_on_the_fallback() -> None:
    model = _ScriptedModel(
        [
            {"*": _response("", tool_names=("recall",))},
            {_PRIMARY: _response(_REFUSAL), _FALLBACK: _response(_Answer(answer="ok"))},
        ]
    )

    result = await _run(model, retry_attempts=2, max_tool_iterations=1)

    assert result.content == _Answer(answer="ok")
    assert model.models == [_PRIMARY, _PRIMARY, _FALLBACK]


@pytest.mark.asyncio
async def test_stream_final_answer_is_not_checked() -> None:
    model = _ScriptedModel([{"*": _response(_REFUSAL)}])

    result = await _run_loop(model, retry_attempts=3, stream_final=True)

    assert isinstance(result, StreamingResponseWithMetadata)
    assert model.models == [_PRIMARY]

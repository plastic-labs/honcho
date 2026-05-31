"""Regression tests for B1: deterministic thinking-input errors must surface
as HTTP 422 and must not be retried.

The Anthropic backend rejects invalid thinking inputs (a negative budget or an
unrecognized effort) by raising ``ValidationException`` (status_code 422) from
inside ``_build_thinking_params``. That raise happens inside the closures the
tenacity ``retry`` wrappers decorate. Without a retry predicate, a deterministic
``HonchoException`` would be retried ``retry_attempts`` times, re-wrapped as a
``tenacity.RetryError`` (which is not a ``HonchoException``), and surface as a
generic 500 — losing the intended 422 and wasting a fallback hop.

The fix adds ``retry=retry_if_not_exception_type(HonchoException)`` to every
retry wrapper in ``src/llm/api.py`` and ``src/llm/tool_loop.py`` so these
deterministic errors fail fast and propagate with their own status code.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import ModelConfig
from src.exceptions import ValidationException
from src.llm import registry, tool_loop
from src.llm.api import honcho_llm_call
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from src.llm.types import ProviderClient

_THINKING_ERROR = "thinking_budget_tokens must be >= 0"


@pytest.mark.asyncio
async def test_negative_thinking_budget_surfaces_as_422_not_retried() -> None:
    """End-to-end through ``honcho_llm_call`` with the real Anthropic backend.

    A negative ``thinking_budget_tokens`` is config-reachable: ``ModelConfig``
    only rejects ``0 < budget < 1024`` for Anthropic, so a negative budget
    passes config load and reaches the backend at call time. The backend
    rejects it, and the retry wrapper must let that ``ValidationException``
    propagate as a 422 rather than retrying it into a ``RetryError`` (a 500).
    """
    model_config = ModelConfig(
        model="claude-opus-4-8",
        transport="anthropic",
        thinking_budget_tokens=-1,
    )

    client = Mock()
    client.messages.create = AsyncMock()

    with (
        patch.dict(registry.CLIENTS, {"anthropic": client}),
        pytest.raises(ValidationException) as exc_info,
    ):
        await honcho_llm_call(
            model_config=model_config,
            prompt="hi",
            max_tokens=100,
            enable_retry=True,
            retry_attempts=3,
        )

    assert exc_info.value.status_code == 422
    # Validation fails before any network call, and the wrapper must not retry:
    # the old behavior raised a RetryError after exhausting all attempts.
    assert client.messages.create.await_count == 0


@pytest.mark.parametrize(
    "messages",
    [None, [{"role": "user", "content": "hi"}]],
)
@pytest.mark.asyncio
async def test_toolless_validation_exception_not_retried(
    messages: list[dict[str, Any]] | None,
) -> None:
    """The tool-less retry wrappers (``api.py``) call the inner backend exactly
    once for a deterministic ``ValidationException`` — not ``retry_attempts``
    times. ``messages=None`` exercises the prompt-only wrapper; a supplied
    ``messages`` list exercises the message-list wrapper.
    """
    model_config = ModelConfig(model="claude-opus-4-8", transport="anthropic")

    inner = AsyncMock(side_effect=ValidationException(_THINKING_ERROR))

    with (
        patch.dict(registry.CLIENTS, {"anthropic": Mock()}),
        patch("src.llm.api.honcho_llm_call_inner", inner),
        pytest.raises(ValidationException) as exc_info,
    ):
        await honcho_llm_call(
            model_config=model_config,
            prompt="hi",
            max_tokens=100,
            messages=messages,
            enable_retry=True,
            retry_attempts=3,
        )

    assert exc_info.value.status_code == 422
    assert inner.await_count == 1


def _make_plan() -> AttemptPlan:
    return AttemptPlan(
        provider="anthropic",
        model="claude-opus-4-8",
        client=cast(ProviderClient, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=ModelConfig(model="claude-opus-4-8", transport="anthropic"),
        attempt=1,
        retry_attempts=3,
        is_fallback=False,
    )


def _unused_tool_executor(_name: str, _tool_input: dict[str, Any]) -> str:
    # The inner call raises before any tool runs; this is never invoked.
    return ""


def _noop_before_retry(_retry_state: Any) -> None:
    return None


@pytest.mark.asyncio
async def test_tool_loop_validation_exception_not_retried() -> None:
    """The per-iteration retry wrapper in ``execute_tool_loop`` (the Dialectic
    and Dreamer path) must also fail fast on a deterministic
    ``ValidationException`` — exactly one inner call, propagated as 422.
    """
    inner = AsyncMock(side_effect=ValidationException(_THINKING_ERROR))
    tools: list[dict[str, Any]] = [
        {"name": "noop", "description": "no-op", "input_schema": {"type": "object"}}
    ]

    with (
        patch.object(tool_loop, "honcho_llm_call_inner", inner),
        pytest.raises(ValidationException) as exc_info,
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            tool_executor=_unused_tool_executor,
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=True,
            retry_attempts=3,
            max_input_tokens=None,
            get_attempt_plan=_make_plan,
            before_retry_callback=_noop_before_retry,
            stream_final=False,
            telemetry=None,
        )

    assert exc_info.value.status_code == 422
    assert inner.await_count == 1

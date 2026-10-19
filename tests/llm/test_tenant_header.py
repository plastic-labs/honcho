from __future__ import annotations

import hashlib
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from src.config import ModelConfig, settings
from src.db import tenant_context
from src.llm.backend import CompletionResult as BackendCompletionResult
from src.llm.runtime import AttemptPlan


def _plan() -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="deriver",
        client=cast(Any, object()),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=cast(Any, object()),
        attempt=1,
        retry_attempts=3,
        is_fallback=False,
    )


async def _headers(selected_config: ModelConfig | None = None) -> dict[str, Any] | None:
    from src.llm import executor, registry

    completion = AsyncMock(
        return_value=BackendCompletionResult(
            content="ok", input_tokens=1, output_tokens=1, finish_reason="stop"
        )
    )
    with (
        patch.object(registry, "CLIENTS", {"openai": object()}),
        patch.object(executor, "backend_for_provider", return_value=object()),
        patch.object(executor, "execute_completion", new=completion),
        patch("src.telemetry.events.emit"),
    ):
        await executor.honcho_llm_call_inner(
            "openai",
            "deriver",
            "hello",
            max_tokens=16,
            plan=_plan(),
            selected_config=selected_config,
        )
    assert completion.await_args is not None
    return completion.await_args.kwargs["extra_params"].get("extra_headers")


def _hashed(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


@pytest.mark.asyncio
async def test_multi_tenant_sends_hashed_tenant():
    token = tenant_context.set("tenant-a")
    try:
        with patch.object(settings, "MULTI_TENANT", True):
            headers = await _headers()
    finally:
        tenant_context.reset(token)
    assert headers == {"X-Honcho-Tenant": _hashed("tenant-a")}


@pytest.mark.asyncio
async def test_single_tenant_sends_no_header():
    token = tenant_context.set("tenant-a")
    try:
        with patch.object(settings, "MULTI_TENANT", False):
            headers = await _headers()
    finally:
        tenant_context.reset(token)
    assert headers is None


@pytest.mark.asyncio
async def test_multi_tenant_without_bound_tenant_sends_no_header():
    with patch.object(settings, "MULTI_TENANT", True):
        headers = await _headers()
    assert headers is None


@pytest.mark.asyncio
async def test_operator_headers_are_kept():
    config = ModelConfig(
        transport="openai",
        model="deriver",
        provider_params={"extra_headers": {"X-Operator": "1"}},
    )
    token = tenant_context.set("tenant-a")
    try:
        with patch.object(settings, "MULTI_TENANT", True):
            headers = await _headers(config)
    finally:
        tenant_context.reset(token)
    assert headers == {"X-Honcho-Tenant": _hashed("tenant-a"), "X-Operator": "1"}

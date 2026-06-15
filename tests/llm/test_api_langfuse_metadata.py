from typing import Any

import pytest

from src.config import ModelConfig
from src.llm.api import honcho_llm_call
from src.llm.runtime import AttemptPlan
from src.llm.types import HonchoLLMCallResponse


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(model="gpt-4o-mini", transport="openai")


@pytest.fixture
def attempt_plan(model_config: ModelConfig) -> AttemptPlan:
    return AttemptPlan(
        provider="openai",
        model="gpt-4o-mini",
        client=object(),  # type: ignore[arg-type]
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=model_config,
    )


@pytest.mark.asyncio
async def test_honcho_llm_call_forwards_langfuse_metadata_without_tools(
    monkeypatch: pytest.MonkeyPatch,
    model_config: ModelConfig,
    attempt_plan: AttemptPlan,
) -> None:
    update_calls: list[dict[str, Any]] = []

    monkeypatch.setattr("src.llm.api.plan_attempt", lambda **_: attempt_plan)
    monkeypatch.setattr(
        "src.llm.api.update_current_langfuse_observation",
        lambda provider, model, **kwargs: update_calls.append(
            {"provider": provider, "model": model, **kwargs}
        ),
    )

    async def fake_inner(*_: Any, **__: Any) -> HonchoLLMCallResponse[str]:
        return HonchoLLMCallResponse(
            content="ok",
            output_tokens=1,
            finish_reasons=["stop"],
        )

    monkeypatch.setattr("src.llm.api.honcho_llm_call_inner", fake_inner)

    result = await honcho_llm_call(
        model_config=model_config,
        prompt="hello",
        max_tokens=10,
        track_name="Dialectic Agent",
        enable_retry=False,
        langfuse_metadata={"honcho_operation": "dialectic_chat"},
        langfuse_trace_user_id="user-123",
        langfuse_trace_session_id="chat-abc",
        langfuse_trace_tags=["honcho", "memory", "dialectic_chat"],
    )

    assert result.content == "ok"
    assert update_calls == [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "name": "Dialectic Agent",
            "metadata": {"honcho_operation": "dialectic_chat"},
            "trace_user_id": "user-123",
            "trace_session_id": "chat-abc",
            "trace_tags": ["honcho", "memory", "dialectic_chat"],
        }
    ]


@pytest.mark.asyncio
async def test_honcho_llm_call_forwards_langfuse_metadata_to_tool_loop_attempt_plan(
    monkeypatch: pytest.MonkeyPatch,
    model_config: ModelConfig,
    attempt_plan: AttemptPlan,
) -> None:
    update_calls: list[dict[str, Any]] = []

    monkeypatch.setattr("src.llm.api.plan_attempt", lambda **_: attempt_plan)
    monkeypatch.setattr(
        "src.llm.api.update_current_langfuse_observation",
        lambda provider, model, **kwargs: update_calls.append(
            {"provider": provider, "model": model, **kwargs}
        ),
    )

    async def fake_tool_loop(*_: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        get_attempt_plan = kwargs["get_attempt_plan"]
        plan = get_attempt_plan()
        assert plan.provider == "openai"
        return HonchoLLMCallResponse(
            content="tool-ok",
            output_tokens=1,
            finish_reasons=["stop"],
        )

    monkeypatch.setattr("src.llm.api.execute_tool_loop", fake_tool_loop)

    async def fake_tool_executor(_: str, __: dict[str, Any]) -> str:
        return "done"

    result = await honcho_llm_call(
        model_config=model_config,
        prompt="hello",
        max_tokens=10,
        track_name="Minimal Deriver",
        tools=[{"type": "function", "function": {"name": "noop"}}],
        tool_executor=fake_tool_executor,
        enable_retry=False,
        langfuse_metadata={"honcho_operation": "minimal_deriver"},
        langfuse_trace_user_id="user-123",
        langfuse_trace_session_id="chat-abc",
        langfuse_trace_tags=["honcho", "memory", "minimal_deriver"],
    )

    assert result.content == "tool-ok"
    assert update_calls == [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "name": "Minimal Deriver",
            "metadata": {"honcho_operation": "minimal_deriver"},
            "trace_user_id": "user-123",
            "trace_session_id": "chat-abc",
            "trace_tags": ["honcho", "memory", "minimal_deriver"],
        }
    ]


@pytest.mark.asyncio
async def test_honcho_llm_call_existing_callers_keep_empty_langfuse_metadata(
    monkeypatch: pytest.MonkeyPatch,
    model_config: ModelConfig,
    attempt_plan: AttemptPlan,
) -> None:
    update_calls: list[dict[str, Any]] = []

    monkeypatch.setattr("src.llm.api.plan_attempt", lambda **_: attempt_plan)
    monkeypatch.setattr(
        "src.llm.api.update_current_langfuse_observation",
        lambda provider, model, **kwargs: update_calls.append(
            {"provider": provider, "model": model, **kwargs}
        ),
    )

    async def fake_inner(*_: Any, **__: Any) -> HonchoLLMCallResponse[str]:
        return HonchoLLMCallResponse(
            content="ok",
            output_tokens=1,
            finish_reasons=["stop"],
        )

    monkeypatch.setattr("src.llm.api.honcho_llm_call_inner", fake_inner)

    await honcho_llm_call(
        model_config=model_config,
        prompt="hello",
        max_tokens=10,
        enable_retry=False,
    )

    assert update_calls == [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "name": None,
            "metadata": None,
            "trace_user_id": None,
            "trace_session_id": None,
            "trace_tags": None,
        }
    ]

from pydantic import BaseModel

from src.config import ModelConfig
from src.llm.caching import PromptCachePolicy
from src.llm.request_builder import execute_completion
from tests.unit.llm.conftest import FakeBackend


class SampleResponse(BaseModel):
    answer: str


async def test_gemini_explicit_budget_adjusts_transport_max_tokens(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(
        model="gemini/gemini-2.5-flash",
        thinking_budget_tokens=256,
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    assert call["max_output_tokens"] == 100
    assert call["max_tokens"] == 356
    assert call["thinking_budget_tokens"] == 256


async def test_thinking_params_are_passed_through_without_capability_dropping(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(
        model="anthropic/claude-haiku-4-5",
        thinking_effort="high",
        thinking_budget_tokens=1024,
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    assert call["thinking_effort"] == "high"
    assert call["thinking_budget_tokens"] == 1024


async def test_cache_policy_is_passed_through_extra_params(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(model="openai/gpt-4.1-mini")
    cache_policy = PromptCachePolicy(mode="prefix", ttl_seconds=300)

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=SampleResponse,
        cache_policy=cache_policy,
    )

    call = fake_backend.calls[0]
    assert call["response_format"] is SampleResponse
    assert call["extra_params"]["cache_policy"] == cache_policy


async def test_provider_params_are_merged_into_extra_params(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(
        model="openai/gpt-4.1-mini",
        top_p=0.9,
        provider_params={"custom_flag": True},
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    assert call["extra_params"]["top_p"] == 0.9
    assert call["extra_params"]["custom_flag"] is True

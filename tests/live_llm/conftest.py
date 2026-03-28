from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from src.config import ModelConfig, settings
from src.llm import get_backend
from src.llm.caching import gemini_cache_store

from .model_matrix import LiveModelSpec, selected_model_summary_lines


class StructuredLiveResponse(BaseModel):
    provider: str
    family: str
    answer: str


def pytest_report_header(config: pytest.Config) -> list[str] | None:
    if not config.getoption("--live-llm"):
        return None
    return ["live llm model matrix:"] + [
        f"  {line}" for line in selected_model_summary_lines()
    ]


@pytest.fixture(autouse=True)
def clear_live_gemini_cache_store() -> Iterator[None]:
    # The live Gemini cache store is process-local and should not leak state between tests.
    gemini_cache_store._handles.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    gemini_cache_store._handles.clear()  # pyright: ignore[reportPrivateUsage]


def require_provider_key(model_spec: LiveModelSpec) -> None:
    key_present = {
        "anthropic": bool(settings.LLM.ANTHROPIC_API_KEY),
        "openai": bool(settings.LLM.OPENAI_API_KEY),
        "gemini": bool(settings.LLM.GEMINI_API_KEY),
    }[model_spec.provider]
    if not key_present:
        pytest.skip(f"Missing API key for live provider {model_spec.provider}")


def make_model_config(model_spec: LiveModelSpec, **overrides: Any) -> ModelConfig:
    return ModelConfig(model=model_spec.model, **overrides)


def make_backend(
    model_spec: LiveModelSpec, **config_overrides: Any
) -> tuple[Any, ModelConfig]:
    config = make_model_config(model_spec, **config_overrides)
    return get_backend(config), config


def make_large_system_prompt(*, label: str) -> str:
    repeated_prefix = " ".join([f"{label}-token-{index % 37}" for index in range(2400)])
    return (
        f"{label} system prompt. Reuse this prefix exactly for prompt-caching validation. "
        f"{repeated_prefix}"
    )


def favorite_prime_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_favorite_prime",
            "description": "Return the favorite prime number for the current test run.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Why the caller wants the prime number.",
                    }
                },
                "required": ["topic"],
            },
        }
    ]


def execute_local_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    assert tool_name == "get_favorite_prime"
    assert isinstance(tool_input, dict)
    return "13"


def wrap_async_method(
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
    attribute: str,
) -> list[dict[str, Any]]:
    original = getattr(target, attribute)
    calls: list[dict[str, Any]] = []

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, attribute, wrapped)
    return calls


def extract_openai_reasoning_tokens(raw_response: Any) -> int | None:
    usage = getattr(raw_response, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return None
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    return int(reasoning_tokens) if reasoning_tokens is not None else None

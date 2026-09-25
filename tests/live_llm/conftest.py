from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from src.config import EmbeddingModelConfig, ModelConfig, settings
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]
from src.llm import get_backend
from src.llm.caching import gemini_cache_store

from .embedding_matrix import LiveEmbeddingSpec, selected_embedding_summary_lines
from .model_matrix import LiveModelSpec, selected_model_summary_lines


class StructuredLiveResponse(BaseModel):
    provider: str
    family: str
    answer: str


def pytest_report_header(config: pytest.Config) -> list[str] | None:
    if not config.getoption("--live-llm"):
        return None
    return (
        ["live llm model matrix:"]
        + [f"  {line}" for line in selected_model_summary_lines()]
        + ["live embedding model matrix:"]
        + [f"  {line}" for line in selected_embedding_summary_lines()]
    )


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


def require_embedding_key(spec: LiveEmbeddingSpec) -> str:
    if spec.api_key_env:
        key = os.getenv(spec.api_key_env)
        if not key:
            pytest.skip(f"Missing {spec.api_key_env} for live embedding {spec.id}")
        return key
    key = {
        "openai": settings.LLM.OPENAI_API_KEY,
        "gemini": settings.LLM.GEMINI_API_KEY,
    }[spec.transport]
    if not key:
        pytest.skip(f"Missing API key for live embedding transport {spec.transport}")
    return key


_EMBEDDING_CONFIG_OVERRIDE_KEYS = frozenset({"timeout", "max_batch_size"})


def make_embedding_client(
    spec: LiveEmbeddingSpec, **overrides: Any
) -> _EmbeddingClient:
    """Build a live embedding client for one matrix entry.

    Bypasses the `EmbeddingClient` singleton so each spec gets its own client
    without mutating global settings. `timeout` and `max_batch_size` land on
    `EmbeddingModelConfig`; remaining kwargs go to `_EmbeddingClient`.
    """
    config_overrides = {
        key: overrides.pop(key)
        for key in _EMBEDDING_CONFIG_OVERRIDE_KEYS
        if key in overrides
    }
    kwargs: dict[str, Any] = {
        "vector_dimensions": spec.dimensions,
        "max_input_tokens": 2048,
        "max_tokens_per_request": 300_000,
        "send_dimensions": spec.send_dimensions,
        # Pinned rather than resolved from settings: the matrix exists to exercise
        # the float path that `auto` only picks for third-party providers.
        "encoding_format": "float",
    }
    kwargs.update(overrides)
    return _EmbeddingClient(
        EmbeddingModelConfig(
            transport=spec.transport,
            model=spec.model,
            api_key=require_embedding_key(spec),
            base_url=spec.base_url,
            **config_overrides,
        ),
        **kwargs,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def make_model_config(model_spec: LiveModelSpec, **overrides: Any) -> ModelConfig:
    return ModelConfig(
        model=model_spec.model,
        transport=model_spec.provider,
        **overrides,
    )


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

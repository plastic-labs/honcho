"""Live coverage for features that name a `MODELS` catalog entry.

Each test loads a slot that names catalog entries with `model_ref` and sends
it through `honcho_llm_call`, the same entry point the deriver, summarizer
and dialectic use, so the request is built from the catalog entry's settings
and capability profile and sent to the real provider:

- A primary that can't be reached hands the final attempt to a fallback that
  names an entry on another provider. The fallback attempt carries the
  entry's thinking settings.
- An OpenAI reasoning entry sends its thinking effort, and the token field
  its profile picks, on the primary.
- An entry that declares `max_tokens_param` and `forced_tool_choice =
  "rejected"` makes the OpenAI backend send `max_completion_tokens` and relax
  a forced tool to `auto`, and the provider accepts that request.

Models default to cheap ones and can be changed with
`LIVE_LLM_CATALOG_OPENAI_REASONING_MODEL`,
`LIVE_LLM_CATALOG_OPENAI_NON_REASONING_MODEL` and
`LIVE_LLM_CATALOG_ANTHROPIC_MODEL`.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import BaseModel

from src import config
from src.config import (
    ConfiguredModelSettings,
    ModelCatalogEntry,
    using_model_catalog,
)
from src.llm import honcho_llm_call
from src.llm.backends.anthropic import AnthropicBackend
from src.llm.backends.openai import OpenAIBackend

from .conftest import execute_local_tool, favorite_prime_tools, wrap_async_method

pytestmark = [pytest.mark.live_llm]

_OPENAI_REASONING_MODEL = os.getenv(
    "LIVE_LLM_CATALOG_OPENAI_REASONING_MODEL", "gpt-5.4-mini"
)
_OPENAI_NON_REASONING_MODEL = os.getenv(
    "LIVE_LLM_CATALOG_OPENAI_NON_REASONING_MODEL", "gpt-4.1"
)
_ANTHROPIC_MODEL = os.getenv("LIVE_LLM_CATALOG_ANTHROPIC_MODEL", "claude-haiku-4-5")


class CapitalAnswer(BaseModel):
    country: str
    capital: str


_CAPITAL_PROMPT = (
    "What is the capital of France? Answer with JSON where country is the "
    "country and capital is its capital city."
)


def _require_keys(*transports: str) -> None:
    keys = {
        "openai": config.settings.LLM.OPENAI_API_KEY,
        "anthropic": config.settings.LLM.ANTHROPIC_API_KEY,
    }
    missing = [transport for transport in transports if not keys[transport]]
    if missing:
        pytest.skip(f"Missing API key for {', '.join(missing)}")


def _install_catalog(
    monkeypatch: pytest.MonkeyPatch, entries: dict[str, dict[str, Any]]
) -> dict[str, ModelCatalogEntry]:
    """Load `entries` as the process's `MODELS`, as startup would."""
    catalog = {
        name: ModelCatalogEntry.model_validate(entry) for name, entry in entries.items()
    }
    monkeypatch.setattr(config.settings, "MODELS", catalog)
    return catalog


def _slot(
    catalog: dict[str, ModelCatalogEntry], **fields: Any
) -> ConfiguredModelSettings:
    with using_model_catalog(catalog):
        return ConfiguredModelSettings.model_validate(fields)


def _record_openai_params(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    original = OpenAIBackend._build_params  # pyright: ignore[reportPrivateUsage]
    built: list[dict[str, Any]] = []

    def recording(self: OpenAIBackend, **kwargs: Any) -> dict[str, Any]:
        params = original(self, **kwargs)
        built.append(params)
        return params

    monkeypatch.setattr(OpenAIBackend, "_build_params", recording)
    return built


@pytest.mark.asyncio
async def test_live_catalog_fallback_on_another_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_keys("anthropic")
    catalog = _install_catalog(
        monkeypatch,
        {
            # Nothing listens on port 9, so every primary attempt fails fast.
            "unreachable": {
                "transport": "openai",
                "model": _OPENAI_REASONING_MODEL,
                "overrides": {
                    "base_url": "http://127.0.0.1:9/v1",
                    "api_key": "sk-unreachable",
                },
            },
            "backup": {
                "transport": "anthropic",
                "model": _ANTHROPIC_MODEL,
                "thinking_budget_tokens": 1024,
            },
        },
    )
    slot = _slot(catalog, model_ref="unreachable", fallback={"model_ref": "backup"})
    anthropic_calls = wrap_async_method(monkeypatch, AnthropicBackend, "complete")

    response = await honcho_llm_call(
        model_config=slot,
        prompt=_CAPITAL_PROMPT,
        max_tokens=4096,
        response_model=CapitalAnswer,
        enable_retry=True,
        retry_attempts=2,
    )

    assert response.content.capital.lower() == "paris"
    assert len(anthropic_calls) == 1
    sent = anthropic_calls[0]["kwargs"]
    assert sent["model"] == _ANTHROPIC_MODEL
    assert sent["thinking_budget_tokens"] == 1024


@pytest.mark.asyncio
async def test_live_catalog_reasoning_entry_sends_its_effort_and_token_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_keys("openai")
    catalog = _install_catalog(
        monkeypatch,
        {
            "reasoner": {
                "transport": "openai",
                "model": _OPENAI_REASONING_MODEL,
                "thinking_effort": "low",
            }
        },
    )
    built = _record_openai_params(monkeypatch)

    response = await honcho_llm_call(
        model_config=_slot(catalog, model_ref="reasoner"),
        prompt=_CAPITAL_PROMPT,
        max_tokens=2048,
        response_model=CapitalAnswer,
        enable_retry=False,
    )

    assert response.content.capital.lower() == "paris"
    assert built, "the OpenAI backend built no request"
    params = built[-1]
    assert params["model"] == _OPENAI_REASONING_MODEL
    assert params["reasoning_effort"] == "low"
    assert "max_completion_tokens" in params
    assert "max_tokens" not in params


@pytest.mark.asyncio
async def test_live_catalog_declared_profile_shapes_the_openai_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_keys("openai")
    catalog = _install_catalog(
        monkeypatch,
        {
            "declared": {
                "transport": "openai",
                "model": _OPENAI_NON_REASONING_MODEL,
                "capabilities": {
                    "max_tokens_param": "max_completion_tokens",
                    "forced_tool_choice": "rejected",
                },
            }
        },
    )
    built = _record_openai_params(monkeypatch)

    async def run_tool(name: str, tool_input: dict[str, Any]) -> str:
        return execute_local_tool(name, tool_input)

    response = await honcho_llm_call(
        model_config=_slot(catalog, model_ref="declared"),
        prompt=(
            "Call the get_favorite_prime tool exactly once, then answer in one "
            "sentence that includes the number it returned."
        ),
        max_tokens=1024,
        tools=favorite_prime_tools(),
        tool_choice={"name": "get_favorite_prime"},
        tool_executor=run_tool,
        max_tool_iterations=3,
        enable_retry=False,
    )

    assert "13" in str(response.content)
    assert built, "the OpenAI backend built no request"
    first = built[0]
    assert first["model"] == _OPENAI_NON_REASONING_MODEL
    assert first["tool_choice"] == "auto"
    assert "max_completion_tokens" in first
    assert "max_tokens" not in first

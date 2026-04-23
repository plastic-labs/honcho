from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openai", "gemini"]
FeatureName = Literal["thinking", "structured_output", "caching", "reasoning"]


@dataclass(frozen=True)
class LiveModelFamily:
    provider: ProviderName
    family: str
    env_var: str
    default_models: tuple[str, ...] = ()
    supports_thinking: bool = False
    supports_structured_output: bool = False
    supports_caching: bool = False
    supports_reasoning: bool = False
    supports_tool_replay: bool = False
    docs_url: str | None = None


@dataclass(frozen=True)
class LiveModelSpec:
    provider: ProviderName
    family: str
    model: str
    env_var: str
    supports_thinking: bool
    supports_structured_output: bool
    supports_caching: bool
    supports_reasoning: bool
    supports_tool_replay: bool
    docs_url: str | None = None

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.family}:{self.model}"


MODEL_FAMILIES: tuple[LiveModelFamily, ...] = (
    LiveModelFamily(
        provider="anthropic",
        family="claude_4_5_plus",
        env_var="LIVE_LLM_ANTHROPIC_45_PLUS_MODELS",
        supports_thinking=True,
        supports_structured_output=True,
        supports_caching=True,
        supports_tool_replay=True,
        docs_url="https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    ),
    LiveModelFamily(
        provider="openai",
        family="gpt_4_class",
        env_var="LIVE_LLM_OPENAI_GPT4_MODELS",
        default_models=("gpt-4.1",),
        supports_structured_output=True,
        supports_caching=True,
        docs_url="https://platform.openai.com/docs/models/gpt-4.1",
    ),
    LiveModelFamily(
        provider="openai",
        family="gpt_5_class",
        env_var="LIVE_LLM_OPENAI_GPT5_MODELS",
        default_models=("gpt-5", "gpt-5.4", "gpt-5.4-mini"),
        supports_structured_output=True,
        supports_caching=True,
        supports_reasoning=True,
        docs_url="https://platform.openai.com/docs/models/gpt-5",
    ),
    # OpenAI-compatible transport → OpenRouter-served non-reasoning models.
    # Best canary for operators routing exotic providers through OpenRouter:
    # if honcho works here, it works for most OR-served models. Currently
    # anchored on Inception Labs' Mercury-2 diffusion model (non-chat
    # architecture, must stay on max_tokens, no reasoning_effort).
    LiveModelFamily(
        provider="openai",
        family="openrouter_non_reasoning",
        env_var="LIVE_LLM_OPENAI_OPENROUTER_NON_REASONING_MODELS",
        default_models=("inception/mercury-2",),
        supports_structured_output=False,
        supports_caching=False,
        docs_url="https://openrouter.ai/models",
    ),
    LiveModelFamily(
        provider="gemini",
        family="gemini_2_5_class",
        env_var="LIVE_LLM_GEMINI_25_MODELS",
        default_models=("gemini-2.5-flash",),
        supports_thinking=True,
        supports_structured_output=True,
        supports_caching=True,
        supports_tool_replay=True,
        docs_url="https://ai.google.dev/gemini-api/docs/models/gemini",
    ),
    LiveModelFamily(
        provider="gemini",
        family="gemini_3_0_class",
        env_var="LIVE_LLM_GEMINI_30_MODELS",
        supports_thinking=True,
        supports_structured_output=True,
        supports_caching=True,
        supports_tool_replay=True,
        docs_url="https://ai.google.dev/gemini-api/docs/models/gemini",
    ),
    LiveModelFamily(
        provider="gemini",
        family="gemini_3_1_class",
        env_var="LIVE_LLM_GEMINI_31_MODELS",
        supports_thinking=True,
        supports_structured_output=False,
        supports_caching=False,
        supports_tool_replay=True,
        docs_url="https://ai.google.dev/gemini-api/docs/models/gemini",
    ),
)


def _parse_env_models(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    models = [model.strip() for model in value.split(",")]
    return tuple(model for model in models if model)


def iter_live_model_specs() -> tuple[LiveModelSpec, ...]:
    specs: list[LiveModelSpec] = []
    for family in MODEL_FAMILIES:
        configured_models = _parse_env_models(os.getenv(family.env_var))
        models = configured_models or family.default_models
        for model in models:
            specs.append(
                LiveModelSpec(
                    provider=family.provider,
                    family=family.family,
                    model=model,
                    env_var=family.env_var,
                    supports_thinking=family.supports_thinking,
                    supports_structured_output=family.supports_structured_output,
                    supports_caching=family.supports_caching,
                    supports_reasoning=family.supports_reasoning,
                    supports_tool_replay=family.supports_tool_replay,
                    docs_url=family.docs_url,
                )
            )
    return tuple(specs)


def get_live_model_specs(
    *,
    provider: ProviderName | None = None,
    feature: FeatureName | None = None,
) -> tuple[LiveModelSpec, ...]:
    specs = iter_live_model_specs()
    filtered: list[LiveModelSpec] = []

    for spec in specs:
        if provider is not None and spec.provider != provider:
            continue
        if feature == "thinking" and not spec.supports_thinking:
            continue
        if feature == "structured_output" and not spec.supports_structured_output:
            continue
        if feature == "caching" and not spec.supports_caching:
            continue
        if feature == "reasoning" and not spec.supports_reasoning:
            continue
        filtered.append(spec)

    return tuple(filtered)


def selected_model_summary_lines() -> list[str]:
    lines: list[str] = []
    for family in MODEL_FAMILIES:
        configured_models = _parse_env_models(os.getenv(family.env_var))
        models = configured_models or family.default_models
        joined_models = ", ".join(models) if models else "(none configured)"
        lines.append(
            f"{family.env_var} [{family.provider}/{family.family}]: {joined_models}"
        )
    return lines

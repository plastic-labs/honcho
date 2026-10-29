"""Per-model profiles: how to talk to a model, and what it accepts.

A `ModelProfile` pairs the codec that renders history for a model's transport
with that model's `ModelCapabilities`. Call paths resolve a profile once per
attempt instead of branching on the transport or matching model names inline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.config import ModelTransport

from .capabilities import ModelCapabilities, capabilities_for
from .codecs import ProviderCodec
from .codecs.anthropic import AnthropicCodec
from .codecs.gemini import GeminiCodec
from .codecs.openai import OpenAICodec


@dataclass(frozen=True, slots=True)
class ModelProfile:
    codec: ProviderCodec
    capabilities: ModelCapabilities


class ModelProfileRegistry:
    """Resolves the profile for a (transport, model) pair."""

    def __init__(self, codecs: Mapping[ModelTransport, ProviderCodec]) -> None:
        self._codecs: dict[ModelTransport, ProviderCodec] = dict(codecs)

    def resolve(self, transport: ModelTransport, model: str) -> ModelProfile:
        return ModelProfile(
            codec=self._codecs[transport],
            capabilities=capabilities_for(transport, model),
        )


MODEL_PROFILES = ModelProfileRegistry(
    {
        "anthropic": AnthropicCodec(),
        "openai": OpenAICodec(),
        "gemini": GeminiCodec(),
    }
)


def profile_for(transport: ModelTransport, model: str) -> ModelProfile:
    """The profile for `model` on `transport`, from the default registry."""
    return MODEL_PROFILES.resolve(transport, model)


__all__ = ["MODEL_PROFILES", "ModelProfile", "ModelProfileRegistry", "profile_for"]

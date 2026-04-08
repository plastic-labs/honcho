from __future__ import annotations

from src.config import ModelConfig, settings
from src.exceptions import ValidationException


def resolve_credentials(config: ModelConfig) -> dict[str, str | None]:
    """Resolve credentials for the effective model transport."""

    default_api_key = _default_transport_api_key(config.transport)
    return {
        "api_key": config.api_key or default_api_key,
        "api_base": config.base_url,
    }


def _default_transport_api_key(transport: str) -> str | None:
    if transport == "anthropic":
        return settings.LLM.ANTHROPIC_API_KEY
    if transport == "openai":
        return settings.LLM.OPENAI_API_KEY
    if transport == "gemini":
        return settings.LLM.GEMINI_API_KEY
    raise ValidationException(f"Unknown transport: {transport}")

from __future__ import annotations

from src.config import ModelConfig, settings


def resolve_credentials(config: ModelConfig) -> dict[str, str | None]:
    """Resolve credentials for the effective model transport."""

    default_api_key, default_api_base = _default_transport_credentials(config.transport)
    return {
        "api_key": config.api_key or default_api_key,
        "api_base": config.base_url or default_api_base,
    }


def _default_transport_credentials(transport: str) -> tuple[str | None, str | None]:
    if transport == "anthropic":
        return settings.LLM.ANTHROPIC_API_KEY, settings.LLM.ANTHROPIC_BASE_URL
    if transport == "openai":
        return settings.LLM.OPENAI_API_KEY, settings.LLM.OPENAI_BASE_URL
    if transport == "gemini":
        return settings.LLM.GEMINI_API_KEY, settings.LLM.GEMINI_BASE_URL
    if transport == "groq":
        return settings.LLM.GROQ_API_KEY, settings.LLM.GROQ_BASE_URL
    raise ValueError(f"Unknown transport: {transport}")

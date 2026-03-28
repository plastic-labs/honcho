from __future__ import annotations

import logging

from src.config import ModelConfig, settings

logger = logging.getLogger(__name__)


_PREFIX_MAP: dict[str, tuple[str, str | None]] = {
    "anthropic/": ("ANTHROPIC_API_KEY", None),
    "openai/": ("OPENAI_API_KEY", None),
    "groq/": ("GROQ_API_KEY", None),
    "gemini/": ("GEMINI_API_KEY", None),
    "openrouter/": ("OPENAI_COMPATIBLE_API_KEY", None),
    "hosted_vllm/": ("VLLM_API_KEY", "VLLM_BASE_URL"),
}


def resolve_credentials(config: ModelConfig) -> dict[str, str | None]:
    """Resolve credentials for the effective model transport."""

    if config.transport == "openai_compatible":
        return {
            "api_key": config.api_key,
            "api_base": config.base_url,
        }

    resolved = _resolve_from_global_settings(config.model)
    if config.api_key is not None:
        resolved["api_key"] = config.api_key
    return resolved


def _resolve_from_global_settings(model: str) -> dict[str, str | None]:
    for prefix, (key_attr, base_url_attr) in _PREFIX_MAP.items():
        if model.startswith(prefix):
            result: dict[str, str | None] = {}
            key_value = getattr(settings.LLM, key_attr, None)
            if key_value:
                result["api_key"] = key_value
            if base_url_attr:
                base_url_value = getattr(settings.LLM, base_url_attr, None)
                if base_url_value:
                    result["api_base"] = base_url_value
            return result

    logger.warning("No credential mapping for model prefix: %s", model)
    return {}

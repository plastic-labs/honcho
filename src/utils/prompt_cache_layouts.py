from typing import Literal

from src.utils.types import SupportedProviders

PromptCacheLayoutMode = Literal[
    "auto",
    "split",
    "split_reverse",
    "merged_system",
    "history_in_user",
    "base_in_user",
    "all_user",
]

_MERGED_SYSTEM_PROVIDERS: frozenset[SupportedProviders] = frozenset({"google"})


def provider_default_prompt_cache_layout(
    provider: SupportedProviders,
) -> PromptCacheLayoutMode:
    """Return the default prompt-cache layout for a provider."""
    if provider in _MERGED_SYSTEM_PROVIDERS:
        return "merged_system"
    return "split"


def resolve_prompt_cache_layout_mode(
    provider: SupportedProviders,
    layout_mode: PromptCacheLayoutMode,
) -> PromptCacheLayoutMode:
    """Resolve ``auto`` to the provider's default layout."""
    if layout_mode == "auto":
        return provider_default_prompt_cache_layout(provider)
    return layout_mode


def merge_system_prompt_with_rolling_context(
    base_prompt: str,
    rolling_context: str,
    *,
    wrapper_tag: str = "rolling_history",
) -> str:
    """Combine stable and rolling system context into a single system prompt."""
    stable = base_prompt.strip()
    rolling = rolling_context.strip()

    if not rolling:
        return stable

    wrapped_context = f"<{wrapper_tag}>\n{rolling}\n</{wrapper_tag}>"
    if not stable:
        return wrapped_context
    return f"{stable}\n\n{wrapped_context}"


def build_system_messages(
    provider: SupportedProviders,
    base_prompt: str,
    rolling_context: str | None = None,
    *,
    wrapper_tag: str = "rolling_history",
) -> list[dict[str, str]]:
    """Build provider-aware system messages for cacheable prompt prefixes."""
    base_content = base_prompt.strip()
    rolling_content = rolling_context.strip() if rolling_context else ""

    if not rolling_content:
        return [{"role": "system", "content": base_content}]

    if provider_default_prompt_cache_layout(provider) == "merged_system":
        return [
            {
                "role": "system",
                "content": merge_system_prompt_with_rolling_context(
                    base_content,
                    rolling_content,
                    wrapper_tag=wrapper_tag,
                ),
            }
        ]

    return [
        {"role": "system", "content": base_content},
        {"role": "system", "content": rolling_content},
    ]

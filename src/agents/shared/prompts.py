"""
Shared prompt utilities for Honcho agents.

This module provides common prompt formatting functions and templates
that can be used across multiple agents.
"""

from typing import Any


def format_system_prompt(
    role: str,
    task_description: str,
    guidelines: list[str] | None = None,
    constraints: list[str] | None = None,
) -> str:
    """
    Format a system prompt for an agent.

    Args:
        role: The role of the agent (e.g., "hypothesis generator", "falsifier")
        task_description: Description of the agent's main task
        guidelines: Optional list of guidelines for the agent to follow
        constraints: Optional list of constraints the agent must respect

    Returns:
        Formatted system prompt string
    """
    prompt_parts = [
        f"You are a {role}.",
        "",
        task_description,
    ]

    if guidelines:
        prompt_parts.extend([
            "",
            "Guidelines:",
            *[f"- {guideline}" for guideline in guidelines],
        ])

    if constraints:
        prompt_parts.extend([
            "",
            "Constraints:",
            *[f"- {constraint}" for constraint in constraints],
        ])

    return "\n".join(prompt_parts)


def format_context_section(
    title: str,
    items: list[str] | list[dict[str, Any]],
    item_formatter: Any = None,
) -> str:
    """
    Format a context section with a title and items.

    Args:
        title: Title for the context section
        items: List of items to include (strings or dicts)
        item_formatter: Optional function to format each item

    Returns:
        Formatted context section
    """
    if not items:
        return f"{title}:\n(none)"

    formatted_items = []
    for i, item in enumerate(items, 1):
        if item_formatter:
            formatted_items.append(item_formatter(item, i))
        elif isinstance(item, str):
            formatted_items.append(f"{i}. {item}")
        elif isinstance(item, dict):
            # Default dict formatting
            formatted_items.append(f"{i}. {item.get('content', str(item))}")
        else:
            formatted_items.append(f"{i}. {str(item)}")

    return f"{title}:\n" + "\n".join(formatted_items)


def format_provenance_chain(
    entity: str,
    sources: list[str],
    source_type: str = "premise",
) -> str:
    """
    Format provenance information showing the chain of reasoning.

    Args:
        entity: The entity (hypothesis, prediction, etc.)
        sources: List of source IDs or descriptions
        source_type: Type of sources (e.g., "premise", "hypothesis", "prediction")

    Returns:
        Formatted provenance chain
    """
    if not sources:
        return f"{entity} (no sources)"

    source_list = "\n".join([f"  - {source}" for source in sources])
    return f"{entity}\n  Based on {len(sources)} {source_type}(s):\n{source_list}"


def truncate_text(
    text: str,
    max_length: int = 500,
    suffix: str = "...",
) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def format_peer_info(
    observer: str,
    observed: str,
    observer_card: list[str] | None = None,
    observed_card: list[str] | None = None,
) -> str:
    """
    Format peer information for agent context.

    Args:
        observer: ID of the observing peer
        observed: ID of the observed peer
        observer_card: Optional biographical info about observer
        observed_card: Optional biographical info about observed

    Returns:
        Formatted peer information
    """
    lines = [
        f"Observer: {observer}",
        f"Observed: {observed}",
    ]

    if observer_card:
        lines.append(
            "Observer Background:\n  "
            + "\n  ".join(observer_card)
        )

    if observed_card:
        lines.append(
            "Observed Background:\n  "
            + "\n  ".join(observed_card)
        )

    return "\n".join(lines)

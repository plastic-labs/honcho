"""
Conversation history helpers for LLM calls.

This module is intentionally provider-agnostic. It focuses on token estimation and
safe truncation while preserving tool-call structure across providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from src.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a list of messages using tiktoken-style estimation."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += estimate_tokens(json.dumps(content))

        if "parts" in msg:
            try:
                total += estimate_tokens(json.dumps(msg["parts"]))
            except TypeError:
                total += estimate_tokens(str(msg["parts"]))
    return total


def _is_tool_use_message(msg: dict[str, Any]) -> bool:
    """Return True if a message contains tool calls in any supported format."""
    content = msg.get("content")
    if isinstance(content, list):
        for block in cast(list[dict[str, Any]], content):
            if block.get("type") == "tool_use":
                return True
    return bool(msg.get("tool_calls"))


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    """Return True if a message contains tool results in any supported format."""
    content = msg.get("content")
    if isinstance(content, list):
        for block in cast(list[dict[str, Any]], content):
            if block.get("type") == "tool_result":
                return True
    return msg.get("role") == "tool"


def _group_into_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Group messages into logical conversation units.

    A unit is either:
    - A tool_use message + ALL consecutive tool_result messages that follow
    - A single non-tool message

    This ensures tool_use and tool_results stay together.
    """
    units: list[list[dict[str, Any]]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if _is_tool_use_message(msg):
            j = i + 1
            while j < len(messages) and _is_tool_result_message(messages[j]):
                j += 1

            unit = messages[i:j]
            if len(unit) > 1:
                units.append(unit)
                i = j
            else:
                logger.debug("Skipping orphaned tool_use at index %s", i)
                i += 1
        elif _is_tool_result_message(msg):
            logger.debug("Skipping orphaned tool_result at index %s", i)
            i += 1
        else:
            units.append([msg])
            i += 1

    return units


def truncate_messages_to_fit(
    messages: list[dict[str, Any]],
    max_tokens: int,
    preserve_system: bool = True,
) -> list[dict[str, Any]]:
    """
    Truncate messages to fit within a token limit while maintaining valid structure.

    Strategy:
    1. Group messages into units (tool_use + results together, or single messages)
    2. Remove oldest units first to preserve recent context
    3. Units stay intact so tool_use/tool_result pairs are never broken
    """
    current_tokens = count_message_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    logger.info("Truncating: %s tokens exceeds %s limit", current_tokens, max_tokens)

    system_messages: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system" and preserve_system:
            system_messages.append(msg)
        else:
            conversation.append(msg)

    system_tokens = count_message_tokens(system_messages)
    available_tokens = max_tokens - system_tokens

    if available_tokens <= 0:
        logger.warning("System message exceeds max_input_tokens")
        return messages

    units = _group_into_units(conversation)

    if not units:
        logger.warning("No valid conversation units")
        return system_messages

    while len(units) > 1:
        flat_messages = [msg for unit in units for msg in unit]
        if count_message_tokens(flat_messages) <= available_tokens:
            break

        removed_unit = units.pop(0)
        logger.debug(
            "Removed unit with %s messages (~%s tokens)",
            len(removed_unit),
            count_message_tokens(removed_unit),
        )

    result_conversation = [msg for unit in units for msg in unit]

    result = system_messages + result_conversation
    result_tokens = count_message_tokens(result)
    logger.info(
        "Truncation complete: %s -> %s messages, %s -> %s tokens, %s units kept",
        len(messages),
        len(result),
        current_tokens,
        result_tokens,
        len(units),
    )
    return result

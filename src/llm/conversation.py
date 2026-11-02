"""Conversation-shaping helpers: token counting + tool-aware truncation.

Moved out of src/utils/clients.py as part of the migration into src/llm/.
These are pure helpers with no orchestration dependencies.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from typing import Any

from src.utils.json_coerce import as_dict, as_list
from src.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a list of messages using tiktoken."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Anthropic-style content blocks
            total += estimate_tokens(json.dumps(content))
        if "parts" in msg:
            try:
                total += estimate_tokens(json.dumps(msg["parts"]))
            except TypeError:
                # Non-JSON-serializable content (e.g. bytes) — estimate from repr.
                total += estimate_tokens(str(msg["parts"]))
    return total


def _has_content_block(msg: dict[str, Any], block_type: str) -> bool:
    """Whether an Anthropic-style ``content`` list carries a block of this type."""
    for block in as_list(msg.get("content")) or ():
        entry = as_dict(block)
        if entry is not None and entry.get("type") == block_type:
            return True
    return False


def _has_part_key(msg: dict[str, Any], key: str) -> bool:
    """Whether a Gemini-style ``parts`` list carries an entry with this key."""
    for part in as_list(msg.get("parts")) or ():
        entry = as_dict(part)
        if entry is not None and key in entry:
            return True
    return False


def _is_tool_use_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool calls (any format).

    Recognizes:
    - Anthropic: ``content`` is a list containing a ``{"type": "tool_use"}`` block.
    - Gemini: ``parts`` is a list containing a ``{"function_call": …}`` entry.
    - OpenAI: assistant message with a non-empty ``tool_calls`` field.
    """
    return (
        _has_content_block(msg, "tool_use")
        or _has_part_key(msg, "function_call")
        or bool(msg.get("tool_calls"))
    )


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool results (any format).

    Recognizes:
    - Anthropic: ``content`` is a list containing a ``{"type": "tool_result"}`` block.
    - Gemini: ``parts`` is a list containing a ``{"function_response": …}`` entry.
    - OpenAI: message with ``role == "tool"``.
    """
    return (
        _has_content_block(msg, "tool_result")
        or _has_part_key(msg, "function_response")
        or msg.get("role") == "tool"
    )


def _is_plain_user_message(msg: dict[str, Any]) -> bool:
    """A user turn that carries neither tool calls nor tool results."""
    return (
        msg.get("role") == "user"
        and not _is_tool_use_message(msg)
        and not _is_tool_result_message(msg)
    )


def _group_into_units(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group messages into logical conversation units.

    A unit is either:
    - A tool_use message + ALL consecutive tool_result messages that follow,
      plus a plain user message right after them (a budget line or synthesis
      prompt the tool loop appends to the round)
    - A single non-tool message

    Keeps tool_use / tool_result pairs together so truncation never breaks
    them apart, and keeps a note on the round with it: as a unit of its own it
    would be the last one kept, and every tool result would be dropped.
    """
    units: list[list[dict[str, Any]]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if _is_tool_use_message(msg):
            j = i + 1
            while j < len(messages) and _is_tool_result_message(messages[j]):
                j += 1
            if j > i + 1 and j < len(messages) and _is_plain_user_message(messages[j]):
                j += 1
            unit = messages[i:j]
            if len(unit) > 1:
                units.append(unit)
                i = j
            else:
                # Orphaned tool_use with no results — skip it.
                logger.debug(f"Skipping orphaned tool_use at index {i}")
                i += 1
        elif _is_tool_result_message(msg):
            # Orphaned tool_result — skip it.
            logger.debug(f"Skipping orphaned tool_result at index {i}")
            i += 1
        else:
            units.append([msg])
            i += 1

    return units


# Clipping never shortens a tool-result payload below this, so short results
# and error strings survive intact; only bulk payloads give way.
_MIN_CLIPPED_CHARS = 200
_CLIP_MARKER = "\n\n[TRUNCATED to fit the input budget: {removed:,} of {total:,} characters removed]"


@dataclass
class _PayloadSlot:
    """One tool-result payload string and where it lives in its message.

    `encoded` marks payloads nested in a content-block / parts list, which
    `count_message_tokens` measures as JSON -- so their cost is estimated the
    same way, or escaping (e.g. non-ASCII as \\uXXXX) would skew the cut.
    """

    container: dict[str, Any]
    key: str
    encoded: bool
    original: str
    kept_chars: int | None = None

    @property
    def kept(self) -> str:
        """The payload as currently kept, without any truncation marker."""
        return (
            self.original
            if self.kept_chars is None
            else self.original[: self.kept_chars]
        )

    def tokens(self, text: str) -> int:
        return estimate_tokens(json.dumps(text) if self.encoded else text)

    def cut_by(self, excess_tokens: int) -> bool:
        """Shrink the payload by about `excess_tokens`; False once at the floor."""
        kept = self.kept
        if len(kept) <= _MIN_CLIPPED_CHARS:
            return False
        kept_tokens = max(self.tokens(kept), 1)
        keep_ratio = max(kept_tokens - excess_tokens, 0) / kept_tokens
        # 10% headroom: tokens per char is not uniform across the string.
        self.kept_chars = max(int(len(kept) * keep_ratio * 0.9), _MIN_CLIPPED_CHARS)
        self.container[self.key] = self.original[
            : self.kept_chars
        ] + _CLIP_MARKER.format(
            removed=len(self.original) - self.kept_chars, total=len(self.original)
        )
        return True


def _payload_slots(msg: dict[str, Any]) -> list[_PayloadSlot]:
    """The tool-result payload strings in `msg`, one per result, per provider shape.

    Matches the shapes `history_adapters` builds: OpenAI ``{"role": "tool",
    "content": str}``, Anthropic ``tool_result`` blocks with string
    ``content``, and Gemini ``function_response.response.result``. Ids, names
    and any non-text payload are never touched.
    """
    content = msg.get("content")
    if msg.get("role") == "tool" and isinstance(content, str):
        return [_PayloadSlot(msg, "content", encoded=False, original=content)]
    slots: list[_PayloadSlot] = []
    for block in as_list(content) or ():
        entry = as_dict(block)
        if (
            entry is not None
            and entry.get("type") == "tool_result"
            and isinstance(entry.get("content"), str)
        ):
            slots.append(_PayloadSlot(entry, "content", True, entry["content"]))
    for part in as_list(msg.get("parts")) or ():
        response = as_dict((as_dict(part) or {}).get("function_response"))
        result = as_dict((response or {}).get("response"))
        if result is not None and isinstance(result.get("result"), str):
            slots.append(_PayloadSlot(result, "result", True, result["result"]))
    return slots


def _clip_tool_results(
    messages: list[dict[str, Any]], max_tokens: int
) -> list[dict[str, Any]]:
    """Shorten the largest tool-result payloads until `messages` fits.

    Only tool results are clipped: they are retrieval output, whereas other
    content (the system prompt, a single-prompt caller's user message) is
    passed through as-is. Returns `messages` itself when it has no tool
    results, otherwise a clipped copy; the caller's messages are never mutated.
    May still exceed `max_tokens` when non-tool content alone is too large.
    """
    if not any(_is_tool_result_message(msg) for msg in messages):
        return messages
    clipped = copy.deepcopy(messages)
    slots = [slot for msg in clipped for slot in _payload_slots(msg)]

    # Each pass either fits or cuts one payload to the floor, so the loop is
    # bounded by the payload count; the extra passes absorb estimate error.
    for _ in range(2 * len(slots) + 2):
        excess = count_message_tokens(clipped) - max_tokens
        if excess <= 0:
            break
        if not max(slots, key=lambda s: len(s.kept)).cut_by(excess):
            break
    return clipped


# A token is never longer than this many characters in practice, so a payload
# cut to this many characters per budgeted token still holds a full budget.
# Cutting by length first keeps tiktoken off multi-megabyte strings, which
# would otherwise block the event loop for seconds.
_MAX_CHARS_PER_TOKEN = 8


def _cap_each_tool_result(
    messages: list[dict[str, Any]], max_tokens: int
) -> list[dict[str, Any]]:
    """Clip any single tool-result payload larger than `max_tokens`.

    One runaway result (e.g. an error that embeds a whole SQL statement) should
    be cut down on its own, not displace the earlier tool rounds around it.
    Returns `messages` itself when nothing is over the cap.
    """
    char_cap = max_tokens * _MAX_CHARS_PER_TOKEN
    if not any(
        slot.tokens(slot.original[:char_cap]) > max_tokens
        or len(slot.original) > char_cap
        for msg in messages
        for slot in _payload_slots(msg)
    ):
        return messages
    capped = copy.deepcopy(messages)
    for slot in (slot for msg in capped for slot in _payload_slots(msg)):
        if len(slot.original) > char_cap:
            slot.kept_chars = char_cap
        excess = slot.tokens(slot.kept) - max_tokens
        if excess > 0:
            slot.cut_by(excess)
        elif slot.kept_chars is not None:
            slot.container[slot.key] = slot.kept + _CLIP_MARKER.format(
                removed=len(slot.original) - slot.kept_chars, total=len(slot.original)
            )
    return capped


def truncate_messages_to_fit(
    messages: list[dict[str, Any]],
    max_tokens: int,
    preserve_system: bool = True,
) -> list[dict[str, Any]]:
    """Truncate messages to fit within a token limit while maintaining valid structure.

    Strategy:
    1. Group messages into units (tool_use + results together, or single messages)
    2. Cap each tool-result payload at half the budget, so one runaway result
       cannot displace everything else
    3. Drop the oldest units first, but hold back the opening unit (it states
       the task) and the newest unit carrying tool results (the evidence the
       next turn is about to use)
    4. If the held-back units still don't fit, clip their tool-result payloads
    5. Only then drop the opening unit, and clip again
    Units stay intact so tool_use/tool_result pairs are never broken. Content
    other than tool results is never clipped, so an oversized prompt with no
    tool results is returned over budget.
    """
    current_tokens = count_message_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    logger.info(f"Truncating: {current_tokens} tokens exceeds {max_tokens} limit")

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

    # No one tool result may take more than half the budget; see
    # `_cap_each_tool_result`.
    conversation = _cap_each_tool_result(conversation, available_tokens // 2)

    units = _group_into_units(conversation)

    if not units:
        logger.warning("No valid conversation units")
        return system_messages

    def fits(candidate: list[list[dict[str, Any]]]) -> bool:
        return (
            count_message_tokens([m for u in candidate for m in u]) <= available_tokens
        )

    # Held back from dropping: the opening unit, and the newest unit carrying
    # tool results (or simply the newest unit when none carries any).
    evidence = next(
        (
            i
            for i in range(len(units) - 1, -1, -1)
            if any(_is_tool_result_message(m) for m in units[i])
        ),
        len(units) - 1,
    )
    held = {0, evidence, len(units) - 1}

    for i in range(len(units)):
        if fits(units):
            break
        if i in held:
            continue
        removed_unit = units[i]
        units[i] = []
        logger.debug(
            "Dropping conversation unit with "
            + f"{len(removed_unit)} messages "
            + f"(~{count_message_tokens(removed_unit)} tokens)"
        )

    if not fits(units):
        flat = _clip_tool_results([m for u in units for m in u], available_tokens)
        if (
            count_message_tokens(flat) > available_tokens
            and len(units[0])
            and any(units[1:])
        ):
            # The opening unit is the last thing given up.
            logger.debug("Dropping the opening conversation unit")
            units[0] = []
            flat = _clip_tool_results([m for u in units for m in u], available_tokens)
        units = [flat]

    result = system_messages + [m for unit in units for m in unit]
    result_tokens = count_message_tokens(result)
    logger.info(
        f"Truncation complete: {current_tokens} → {result_tokens} tokens "
        + f"({len(messages)} → {len(result)} messages)"
    )
    return result


__all__ = [
    "count_message_tokens",
    "truncate_messages_to_fit",
]

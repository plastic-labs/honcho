"""Conversation-shaping helpers: token counting + tool-aware truncation.

Pure helpers over typed `ChatMessage` history, with no orchestration
dependencies.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.utils.tokens import estimate_tokens

from .messages import AIMessage, ChatMessage, SystemMessage, ToolMessage, UserMessage

logger = logging.getLogger(__name__)


def _message_tokens(message: ChatMessage) -> int:
    if isinstance(message, SystemMessage | UserMessage | ToolMessage):
        return estimate_tokens(message.content)
    total = estimate_tokens(message.text or "")
    if message.tool_calls:
        total += estimate_tokens(
            json.dumps(
                [
                    {"id": call.id, "name": call.name, "input": call.input}
                    for call in message.tool_calls
                ],
                default=str,
            )
        )
    for part in message.reasoning:
        total += estimate_tokens(json.dumps(part.payload, default=str))
    return total


def count_message_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimate the tokens `messages` take, across text, tool calls, results
    and reasoning."""
    return sum(_message_tokens(message) for message in messages)


def _is_tool_use_message(message: ChatMessage) -> bool:
    return isinstance(message, AIMessage) and bool(message.tool_calls)


def _group_into_units(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """Group messages into logical conversation units.

    A unit is either:
    - An assistant turn with tool calls + ALL consecutive tool results that
      follow, plus a user message right after them (a budget line or
      synthesis prompt the tool loop appends to the round)
    - A single other message

    Keeps tool calls and their results together so truncation never breaks
    them apart, and keeps a note on the round with it: as a unit of its own it
    would be the last one kept, and every tool result would be dropped.
    """
    units: list[list[ChatMessage]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if _is_tool_use_message(msg):
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                j += 1
            if j > i + 1 and j < len(messages) and isinstance(messages[j], UserMessage):
                j += 1
            unit = messages[i:j]
            if len(unit) > 1:
                units.append(unit)
                i = j
            else:
                # Orphaned tool calls with no results — skip them.
                logger.debug(f"Skipping orphaned tool_use at index {i}")
                i += 1
        elif isinstance(msg, ToolMessage):
            # Orphaned tool result — skip it.
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
    """One tool result's payload, by its position in a message list.

    Messages are immutable, so a cut stores a copy of the `ToolMessage` back at
    `index`.
    """

    messages: list[ChatMessage]
    index: int
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

    @staticmethod
    def tokens(text: str) -> int:
        return estimate_tokens(text)

    def keep(self, kept_chars: int) -> None:
        """Keep the first `kept_chars` characters, and write them back with a
        marker of what was cut."""
        self.kept_chars = kept_chars
        message = self.messages[self.index]
        self.messages[self.index] = message.model_copy(
            update={
                "content": self.kept
                + _CLIP_MARKER.format(
                    removed=len(self.original) - kept_chars,
                    total=len(self.original),
                )
            }
        )

    def cut_by(self, excess_tokens: int) -> bool:
        """Shrink the payload by about `excess_tokens`; False once at the floor."""
        kept = self.kept
        if len(kept) <= _MIN_CLIPPED_CHARS:
            return False
        kept_tokens = max(self.tokens(kept), 1)
        keep_ratio = max(kept_tokens - excess_tokens, 0) / kept_tokens
        # 10% headroom: tokens per char is not uniform across the string.
        self.keep(max(int(len(kept) * keep_ratio * 0.9), _MIN_CLIPPED_CHARS))
        return True


def _payload_slots(messages: list[ChatMessage]) -> list[_PayloadSlot]:
    """One slot per tool result in `messages`; ids and names are never touched."""
    return [
        _PayloadSlot(messages, index, message.content)
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
    ]


def _clip_tool_results(
    messages: list[ChatMessage], max_tokens: int
) -> list[ChatMessage]:
    """Shorten the largest tool-result payloads until `messages` fits.

    Only tool results are clipped: they are retrieval output, whereas other
    content (the system prompt, a single-prompt caller's user message) is
    passed through as-is. Returns `messages` itself when it has no tool
    results, otherwise a clipped copy; the caller's list is never mutated.
    May still exceed `max_tokens` when non-tool content alone is too large.
    """
    if not any(isinstance(msg, ToolMessage) for msg in messages):
        return messages
    clipped = list(messages)
    slots = _payload_slots(clipped)

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
    messages: list[ChatMessage], max_tokens: int
) -> list[ChatMessage]:
    """Clip any single tool-result payload larger than `max_tokens`.

    One runaway result (e.g. an error that embeds a whole SQL statement) should
    be cut down on its own, not displace the earlier tool rounds around it.
    Returns `messages` itself when nothing is over the cap.
    """
    char_cap = max_tokens * _MAX_CHARS_PER_TOKEN
    if not any(
        slot.tokens(slot.original[:char_cap]) > max_tokens
        or len(slot.original) > char_cap
        for slot in _payload_slots(messages)
    ):
        return messages
    capped = list(messages)
    for slot in _payload_slots(capped):
        if len(slot.original) > char_cap:
            slot.kept_chars = char_cap
        excess = slot.tokens(slot.kept) - max_tokens
        if excess > 0:
            slot.cut_by(excess)
        elif slot.kept_chars is not None:
            slot.keep(slot.kept_chars)
    return capped


def truncate_messages_to_fit(
    messages: list[ChatMessage],
    max_tokens: int,
    preserve_system: bool = True,
) -> list[ChatMessage]:
    """Truncate messages to fit within a token limit while maintaining valid structure.

    Strategy:
    1. Group messages into units (tool calls + results together, or single messages)
    2. Cap each tool-result payload at half the budget, so one runaway result
       cannot displace everything else
    3. Drop the oldest units first, but hold back the opening unit (it states
       the task) and the newest unit carrying tool results (the evidence the
       next turn is about to use)
    4. If the held-back units still don't fit, clip their tool-result payloads
    5. Only then drop the opening unit, and clip again
    Units stay intact so tool calls and their results are never split. Content
    other than tool results is never clipped, so an oversized prompt with no
    tool results is returned over budget.
    """
    current_tokens = count_message_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    logger.info(f"Truncating: {current_tokens} tokens exceeds {max_tokens} limit")

    system_messages: list[ChatMessage] = []
    conversation: list[ChatMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage) and preserve_system:
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

    def fits(candidate: list[list[ChatMessage]]) -> bool:
        return (
            count_message_tokens([m for u in candidate for m in u]) <= available_tokens
        )

    # Held back from dropping: the opening unit, and the newest unit carrying
    # tool results (or simply the newest unit when none carries any).
    evidence = next(
        (
            i
            for i in range(len(units) - 1, -1, -1)
            if any(isinstance(m, ToolMessage) for m in units[i])
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


def strip_thinking_blocks(
    messages: Sequence[ChatMessage],
) -> tuple[list[ChatMessage], int]:
    """Remove Anthropic thinking (and redacted thinking) from a history.

    A thinking block replayed to a model that checks it against the request
    prefix (see `ModelCapabilities.preserved_thinking`) fails once anything
    before it changes. Every turn's text and tool calls stay, so the history
    remains valid without the reasoning those blocks carried.

    Returns:
        The history with the blocks removed (messages without any are reused
        as-is, assistant turns holding nothing else are dropped) and how many
        blocks were removed.
    """
    removed = 0
    result: list[ChatMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            result.append(msg)
            continue
        kept = [part for part in msg.reasoning if not part.kind.startswith("anthropic")]
        if len(kept) == len(msg.reasoning):
            result.append(msg)
            continue
        removed += len(msg.reasoning) - len(kept)
        if kept or msg.text or msg.tool_calls:
            # A turn left with no content would be rejected; drop it.
            result.append(msg.model_copy(update={"reasoning": kept}))
    return result, removed


__all__ = [
    "count_message_tokens",
    "strip_thinking_blocks",
    "truncate_messages_to_fit",
]

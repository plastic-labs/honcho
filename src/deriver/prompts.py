"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

import re
from datetime import datetime
from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens

DERIVER_EXAMPLE_SENTINEL = "__HONCHO_EXAMPLE_"

_MESSAGE_TAG = re.compile(r"<(?=/?message\b)", re.IGNORECASE)


def contains_deriver_example_sentinel(content: str) -> bool:
    """Return whether model output contains the reserved few-shot marker."""
    return DERIVER_EXAMPLE_SENTINEL.casefold() in content.casefold()


def format_deriver_message(
    idx: int, peer: str, target: str, created_at: datetime, content: str
) -> str:
    """Wrap one batch message in a tag carrying its index, author, and target flag.

    Peer ids are restricted to ``[a-zA-Z0-9_-]`` upstream, so only the content
    can carry markup; any ``<message``/``</message`` inside it is neutralized so
    a message cannot forge its own tag boundary.
    """
    is_target = "true" if peer == target else "false"
    time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
    safe_content = _MESSAGE_TAG.sub("&lt;", content)
    return (
        f'<message idx="{idx}" peer="{peer}" target="{is_target}" time="{time_str}">'
        f"{safe_content}</message>"
    )


def _normalized_custom_instructions(custom_instructions: str | None) -> str | None:
    """Return stripped custom instructions, if any."""
    if custom_instructions is None:
        return None

    normalized = custom_instructions.strip()
    return normalized or None


def _custom_instructions_section(custom_instructions: str | None) -> str:
    """Render optional custom instructions for the deriver prompt."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return ""

    return c(
        f"""
        CUSTOM INSTRUCTIONS:
        These instructions apply to the target peer identified below.
        {normalized_custom_instructions}
        """
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: Batch messages, each wrapped by ``format_deriver_message``.

    Returns:
        Formatted prompt string for observation extraction.
    """
    custom_instructions_section = _custom_instructions_section(custom_instructions)
    return c(
        f"""
Analyze messages to extract **explicit atomic facts** about the target peer.

[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible instead of relative references

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Each message is wrapped as `<message idx="N" peer="..." target="true|false" time="...">`. `target="true"` marks messages authored by the target peer; `target="false"` marks everyone else.
- Extract ALL observations from `target="true"` messages. Use `target="false"` messages only as context to interpret them; never derive a fact about the target peer from what another peer said, did, or reported.
- A batch may contain few or no `target="true"` messages, even when it holds many long messages from other peers (agent turns, tool output, system notices). In that case produce few or no conclusions.
- Use the exact peer id from `Target peer:` in final observations, not the phrase "the target peer".
- Properly attribute observations to the correct subject: if it is about the target peer, use the exact peer id as the subject. If the target peer is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.
- Contextualize each observation sufficiently that it is useful on its own.

<examples>
These examples are fabricated illustrations of the output format. Example-only
entities contain the reserved marker `{DERIVER_EXAMPLE_SENTINEL}`. Never emit a
conclusion containing that marker. Every conclusion must be supported by the
<messages> block only.

EXAMPLES (using `example_peer` as the target peer id):
- EXPLICIT: <message idx="0" peer="example_peer" target="true">I configured __HONCHO_EXAMPLE_SERVICE_ALPHA__ to use __HONCHO_EXAMPLE_MODE_BETA__</message> → "example_peer configured __HONCHO_EXAMPLE_SERVICE_ALPHA__ to use __HONCHO_EXAMPLE_MODE_BETA__"
- EXPLICIT: <message idx="1" peer="example_peer" target="true">I enabled __HONCHO_EXAMPLE_FEATURE_GAMMA__ for __HONCHO_EXAMPLE_PROJECT_DELTA__. __HONCHO_EXAMPLE_PROJECT_DELTA__ uses __HONCHO_EXAMPLE_TIER_EPSILON__.</message> → "example_peer enabled __HONCHO_EXAMPLE_FEATURE_GAMMA__ for __HONCHO_EXAMPLE_PROJECT_DELTA__", "__HONCHO_EXAMPLE_PROJECT_DELTA__ uses __HONCHO_EXAMPLE_TIER_EPSILON__"
- NO CONCLUSION: <message idx="2" peer="other_peer" target="false">I inspected __HONCHO_EXAMPLE_SERVICE_ZETA__</message> → nothing; the other peer acted, not example_peer
</examples>

{custom_instructions_section}

Target peer:
{peer_id}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=None,
    )
    return estimate_tokens(prompt)


def estimate_deriver_prompt_tokens(custom_instructions: str | None) -> int:
    """Estimate minimal deriver prompt tokens, including custom instructions if present."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return estimate_minimal_deriver_prompt_tokens()

    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        custom_instructions=normalized_custom_instructions,
    )
    return estimate_tokens(prompt)

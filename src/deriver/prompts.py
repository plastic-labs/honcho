"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from datetime import datetime
from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def format_deriver_message(
    idx: int, peer: str, target: str, created_at: datetime, content: str
) -> str:
    """Wrap one batch message in a tag carrying its index, author, and target flag."""
    is_target = "true" if peer == target else "false"
    time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f'<message idx="{idx}" peer="{peer}" target="{is_target}" time="{time_str}">'
        f"{content}</message>"
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
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Each message is wrapped as `<message idx="N" peer="..." target="true|false" time="...">`. `target="true"` marks messages authored by the target peer; `target="false"` marks everyone else.
- Extract ALL observations from `target="true"` messages. Use `target="false"` messages only as context to interpret them; never derive a fact about the target peer from what another peer said, did, or reported.
- A batch may contain few or no `target="true"` messages, even when it holds many long messages from other peers (agent turns, tool output, system notices). In that case produce few or no conclusions.
- Use the exact peer id from `Target peer:` in final observations, not the phrase "the target peer".
- Properly attribute observations to the correct subject: if it is about the target peer, use the exact peer id as the subject. If the target peer is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

<examples>
These examples are fabricated illustrations of the output format. Never emit a conclusion for which content comes from these examples. Every conclusion must be supported by the <messages> block only.

EXAMPLES (using `alice` as the target peer id):
- EXPLICIT: <message idx="0" peer="alice" target="true">I just turned 25</message> → "alice is 25 years old"
- EXPLICIT: <message idx="1" peer="alice" target="true">I took my dog for a walk in NYC</message> → "alice has a dog", "alice walked her dog in NYC"
- EXPLICIT: <message idx="2" peer="alice" target="true">I've lived in NYC for six years</message> → "alice lives in NYC", "alice has lived in NYC for six years"
- NO CONCLUSION: <message idx="3" peer="assistant" target="false">I read the config file and found the port is 8080</message> → nothing; the assistant acted, not alice
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

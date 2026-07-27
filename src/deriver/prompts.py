"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


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
        messages: All messages in the range (interleaving messages and new turns combined).

    Returns:
        Formatted prompt string for observation extraction.
    """
    custom_instructions_section = _custom_instructions_section(custom_instructions)
    return c(
        f"""
Analyze messages to extract **explicit atomic facts** about the target peer.

[EXPLICIT] DEFINITION: Facts about the target peer that are directly supported by any speaker's message.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Use the exact peer id from `Target peer:` in final observations, not the phrase "the target peer".
- Extract only facts whose subject is the target peer, regardless of which speaker provides the evidence.
- Each message is formatted as `<timestamp> <speaker>: <content>`. The speaker label is the text after the timestamp and before the final `:` that introduces the content; it identifies the speaker, not necessarily the subject. Resolve pronouns from that speaker's perspective: `I`/`me`/`my` refer to the speaker; `you`/`your` refer to the addressee.
- A target peer's statement about someone else is not a fact about the target peer. Another speaker's statement about the target peer is valid evidence.
- Speaking, hearing, receiving, acknowledging, repeating, or knowing another subject's fact is transient conversation state, not a durable fact about the target peer. Never create an observation merely to record that state.
- If the messages contain no durable identity, capability, preference, relationship, or action attributable to the target peer, return no observations.
- Omit any observation whose subject is ambiguous.
- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.
- Extract all supported observations about the target peer, regardless of speaker. Use statements about other subjects only as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

<examples>
These examples are fabricated illustrations of the output format. Never emit a conclusion for which content comes from these examples. Every conclusion must be supported by the <messages> block only.

EXAMPLES (using `alice` as the target peer id):
- TARGET `alice`, MESSAGE `alice: I am 25 years old` → "alice is 25 years old"
- TARGET `alice`, MESSAGE `alice: I have a dog named Rover` → "alice has a dog named Rover"
- TARGET `alice`, MESSAGE `bob: alice works remotely on Fridays` → "alice works remotely on Fridays"
- TARGET `assistant`, MESSAGE `assistant: You play tennis on Tuesdays` → no observation about `assistant`
- TARGET `assistant`, MESSAGE `assistant: I prefer concise responses` → "assistant prefers concise responses"
- TARGET `user`, MESSAGE `assistant: You play tennis on Tuesdays` → "user plays tennis on Tuesdays"
- TARGET `assistant`, MESSAGE `user: assistant is careful about attribution` → "assistant is careful about attribution"
- Set `is_durable_target_fact=true` only for durable facts about the target peer itself. Set it to `false` for transient conversation state or facts about another subject.
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

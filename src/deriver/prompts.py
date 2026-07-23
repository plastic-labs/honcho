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
Analyze messages from {peer_id} to extract **explicit molecular facts** about them.

[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

DECONTEXTUALITY (stranger test): Each observation must be interpretable by a stranger with no access to the conversation.
   - Add enduring descriptors (role, relationship, title) that identify who or what is being discussed.
   - Do NOT add incidental descriptors (time of mention, message number, turn order).
   - "He is nervous" fails the stranger test. "Ann is nervous about the pharmacy job interview" passes.
   - Use absolute dates, not relative ones ("June 26, 2025" not "yesterday").

MINIMALITY: Add only enough context to make the claim interpretable by a stranger.
   - Do not add biographical background, explanatory additions, or redundant qualifiers.
   - "Ann is nervous about the job interview at the pharmacy" is minimal.
   - "Ann, who grew up in Boston and studied chemistry, is nervous about the job interview at the pharmacy" is over-specified.

RULES:
- The target peer is {peer_id}, identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Use the exact peer id from `Target peer:` in final observations, not the phrase "the target peer".
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")
- source_indices: Each message in the <messages> block is prefixed with a 0-based index like [0], [1], [2]. For each observation, set source_indices to the indices of the messages that directly support it. Source messages can be from ANY speaker — the user, an assistant, or another participant. What matters is which messages contain the evidence for the conclusion, not who said them. For example, if the assistant proposes a plan and the user confirms it, both messages are source material. Include the message containing any context needed to interpret the conclusion (e.g., the question being answered by "the first one"). Only include messages that directly support the observation — not the entire conversation.

EXAMPLES (using `{peer_id}` as the target peer id):
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"
- EXPLICIT: "I went to college and then started working at the pharmacy" → "{peer_id} attended college", "{peer_id} works at the pharmacy"
- EXPLICIT: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"
- EXPLICIT (assistant-sourced): Assistant says "Let's set up a Flask project with SQLite" and {peer_id} replies "Sounds good, let's do that" → "{peer_id} is building a project with Flask and SQLite", source_indices: [1, 2] (the assistant's proposal and the user's confirmation both support this)
- EXPLICIT (multi-message): {peer_id} asks "Should I use Postgres or SQLite?" and assistant says "SQLite is simpler for a project like yours" and {peer_id} says "OK, SQLite it is" → "{peer_id} chose SQLite for their project", source_indices: [0, 1, 2] (the question provides context, the recommendation explains the reasoning, and the confirmation establishes the decision)

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

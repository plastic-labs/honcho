"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def _custom_instructions_section(custom_instructions: str | None) -> str:
    """Render the optional custom instructions block for the deriver prompt."""
    if not custom_instructions or not custom_instructions.strip():
        return ""

    return c(
        f"""
        CUSTOM INSTRUCTIONS:
        {custom_instructions.strip()}
        """
    )


def minimal_deriver_system_prompt(peer_id: str) -> str:
    """Generate the cacheable instructions for observation extraction."""
    return c(
        f"""
Analyze messages from {peer_id} to extract **explicit atomic facts** about them.

[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Do not infer unstated background facts or implications
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id} celebrated their birthday last Saturday"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} said they took their dog for a walk in NYC"
- EXPLICIT: "{peer_id} attended college" → "{peer_id} said they attended college"
"""
    )


def minimal_deriver_user_prompt(
    messages: str,
    *,
    custom_instructions: str | None = None,
) -> str:
    """Generate the per-request message payload for observation extraction."""
    instructions_section = _custom_instructions_section(custom_instructions)
    return c(
        f"""
{instructions_section}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    *,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate the combined prompt for fast observation extraction.

    Prefer `minimal_deriver_system_prompt()` plus `minimal_deriver_user_prompt()`
    when making LLM calls so the instructions can be cached independently.
    """
    return c(
        f"""
{minimal_deriver_system_prompt(peer_id)}

{minimal_deriver_user_prompt(messages, custom_instructions=custom_instructions)}
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = "\n\n".join(
            [
                minimal_deriver_system_prompt(peer_id=""),
                minimal_deriver_user_prompt(messages=""),
            ]
        )
        return estimate_tokens(prompt)
    except ValueError:
        return 300


def estimate_deriver_prompt_tokens(custom_instructions: str | None = None) -> int:
    """Estimate deriver prompt tokens, including optional custom instructions."""
    if not custom_instructions or not custom_instructions.strip():
        return estimate_minimal_deriver_prompt_tokens()

    return estimate_tokens(
        "\n\n".join(
            [
                minimal_deriver_system_prompt(peer_id=""),
                minimal_deriver_user_prompt(
                    messages="",
                    custom_instructions=custom_instructions,
                ),
            ]
        )
    )

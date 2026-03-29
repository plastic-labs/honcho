"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_system_prompt() -> str:
    """Generate the cacheable instructions for observation extraction."""
    return c(
        """
Analyze messages to extract **explicit atomic facts** about the target person.

[EXPLICIT] DEFINITION: Facts about the target person that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Use the target person identifier provided in the user message when attributing observations about them.
- Properly attribute observations to the correct subject: if it is about the target person, say so. If the target person is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand the target person.
- Extract ALL observations from the target person's messages, using others as context.
- Prefer meaningful explicit facts over literal restatements of the raw message when the higher-value fact is directly supported.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "The target person is 25 years old", "The target person's birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "The target person has a dog", "The target person was in NYC"
- EXPLICIT: "I attended college in Boston" + general knowledge → "The target person attended college in Boston", "The target person completed high school or equivalent"
"""
    )


def minimal_deriver_user_prompt(peer_id: str, messages: str) -> str:
    """Generate the per-request message payload for observation extraction."""
    return c(
        f"""
Target person identifier: {peer_id}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
) -> str:
    """
    Generate the combined prompt for fast observation extraction.

    Prefer `minimal_deriver_system_prompt()` plus `minimal_deriver_user_prompt()`
    when making LLM calls so the instructions can be cached independently.
    """
    return c(
        f"""
{minimal_deriver_system_prompt()}

{minimal_deriver_user_prompt(peer_id, messages)}
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = "\n\n".join(
            [
                minimal_deriver_system_prompt(),
                minimal_deriver_user_prompt(peer_id="", messages=""),
            ]
        )
        return estimate_tokens(prompt)
    except ValueError:
        return 300

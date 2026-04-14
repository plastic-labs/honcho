"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: All messages in the range (interleaving messages and new turns combined).

    Returns:
        Formatted prompt string for observation extraction.
    """
    return c(
        f"""
Analyze messages from {peer_id} to extract explicit atomic observations about them.

[EXPLICIT] DEFINITION: Observations about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple observations
   - Each observation must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

OUTPUT CONTRACT:
- Return exactly one JSON object with a top-level key `explicit`.
- `explicit` must be an array of objects.
- Each object must use the key `content`.
- Do NOT use keys like `fact`, `observation`, `source`, `source_message`, or any extra keys.
- Example valid output: {{"explicit":[{{"content":"{peer_id} lives in Wisconsin"}}]}}
- If there are multiple observations, add more objects with the same `content` key.
- Do not wrap the JSON in markdown.

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → {{"content":"{peer_id} is 25 years old"}}, {{"content":"{peer_id}'s birthday is June 21st"}}
- EXPLICIT: "I took my dog for a walk in NYC" → {{"content":"{peer_id} has a dog"}}, {{"content":"{peer_id} lives in NYC"}}
- EXPLICIT: "{peer_id} attended college" + general knowledge → {{"content":"{peer_id} completed high school or equivalent"}}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = minimal_deriver_prompt(
            peer_id="",
            messages="",
        )
        return estimate_tokens(prompt)
    except Exception:
        return 300

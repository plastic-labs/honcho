"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

import datetime
from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_prompt(
    peer_id: str,
    message_created_at: datetime.datetime,
    history: str,
    new_turns: list[str],
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        message_created_at: Timestamp of the message.
        history: Recent conversation history.
        new_turns: New conversation turns to analyze.

    Returns:
        Formatted prompt string for observation extraction.
    """
    new_turns_section = "\n".join(new_turns)

    return c(
        f"""
You analyze messages from {peer_id} to extract observations through explicit and deductive reasoning.

Current timestamp: {message_created_at}

OBSERVATION TYPES:
1. EXPLICIT: Facts literally stated by {peer_id} - direct quotes/paraphrases only
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

2. DEDUCTIVE: Conclusions that MUST be true given explicit facts + general knowledge
   - Multiple premises → one conclusion
   - Only derive if logically necessary (not probable/likely)
   - May NOT use probabilistic conclusions as premises

RULES:
- Start each observation with {peer_id}'s name (e.g. "Maria is 25 years old")
- NEVER use generic phrases like "The user..."
- Extract ALL observations from new turns
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"
- DEDUCTIVE: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"

Recent conversation for context:
<history>
{history}
</history>

New turns to analyze:
<new_turns>
{new_turns_section}
</new_turns>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = minimal_deriver_prompt(
            peer_id="",
            message_created_at=datetime.datetime.now(datetime.timezone.utc),
            history="",
            new_turns=[],
        )
        return estimate_tokens(prompt)
    except Exception:
        return 300

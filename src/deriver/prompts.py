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
Extract **molecular facts** about {peer_id} from their messages.

**Core principle**: One fact = one claim. No bundling. No explanation.

## STEP 1: Identify All Candidate Claims
Read each message from {peer_id}. List every distinct claim (even if nested).

**Example**: "I got the job at the pharmacy and I'm really excited because I start next month"
- Claim A: {peer_id} got a job at a pharmacy
- Claim B: {peer_id} is excited (about the job)
- Claim C: {peer_id} starts the job next month

→ Evaluate each separately.

## STEP 2: For Each Claim, Verify It's Atomic
Does it contain:
- One subject? (If multiple subjects → split)
- One verb/action? (If multiple actions → split)
- One object? (If multiple objects → split)

**Example of splitting**: "I interviewed at two companies and both rejected me"
- Split into: "{peer_id} interviewed at company A" + "{peer_id} interviewed at company B" + "{peer_id} was rejected by company A" + "{peer_id} was rejected by company B"

## STEP 3: Decontextualize (Resolve Ambiguities)
Replace:
- Pronouns → nouns
- Vague references → specific names
- Relative times → absolute dates (or SKIP)
- Unnamed people → named people (or SKIP)

## STEP 4: Strip Excess Detail
Remove: explanations, reasons, emotional elaboration, context.

**Before**: "{peer_id} is really stressed and anxious about their pharmacy interview next Tuesday because they're worried about failing the clinical questions"

**After**: "{peer_id} is stressed about their pharmacy interview next Tuesday"

(The anxiety, worry, and reason are elaboration, not separate facts.)

## OUTPUT FORMAT
- **Fact**: [One atomic claim, decontextualized, no elaboration]
- **Source**: [Direct quote from {peer_id}]

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

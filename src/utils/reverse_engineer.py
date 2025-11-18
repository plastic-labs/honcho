"""
Reverse engineer atomic propositions and deductions from message history.

Given a conversation, a question, and the correct answer, this module generates
the minimal set of atomic propositions (explicit + implicit) and deductive
conclusions that would be sufficient to answer the question correctly.

This is primarily used for generating ground truth training data when the
dialectic produces incorrect answers.
"""

import logging

from pydantic import BaseModel, Field

from src.config import settings
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.representation import DeductiveResponse, ExplicitResponse
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)


class DeductionItem(BaseModel):
    """A single deduction with premises and conclusion."""

    premises: list[str] = Field(
        description="Premises supporting this deduction",
        default_factory=list,
    )
    conclusion: str = Field(
        description="The deductive conclusion",
    )


class ReverseEngineerResponse(BaseModel):
    """Combined response containing propositions, deductions, and peer cards."""

    explicit: list[str] = Field(
        description="Explicit facts directly stated in the conversation",
        default_factory=list,
    )
    implicit: list[str] = Field(
        description="Facts clearly implied by the conversation",
        default_factory=list,
    )
    deductions: list[DeductionItem] = Field(
        description="Deductive conclusions with premises and conclusion",
        default_factory=list,
    )
    observer_card: list[str] | None = Field(
        description="Biographical card for the observer (peer asking the question)",
        default=None,
    )
    observed_card: list[str] | None = Field(
        description="Biographical card for the observed peer (peer being asked about)",
        default=None,
    )


def reverse_engineer_prompt(
    messages: list[dict[str, str]],
    question: str,
    answer: str,
    *,
    observer: str | None = None,
    observed: str | None = None,
) -> str:
    """
    Generate prompt for reverse-engineering minimal trace from conversation.

    This prompt is strictly aligned with the deriver's explicit reasoning,
    deductive reasoning, and peer card extraction templates to ensure the
    reverse-engineered trace follows the same extraction rules.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        question: The question that was asked
        answer: The correct/ground truth answer to the question
        observer: Optional name of the peer asking the question
        observed: Optional name of the peer being asked about

    Returns:
        Formatted prompt string for LLM
    """
    # Format the conversation history
    conversation_lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_lines.append(f"{role.upper()}: {content}")

    conversation_text = "\n\n".join(conversation_lines)

    # Build perspective context if provided
    perspective_context = ""
    if observer and observed:
        if observer == observed:
            perspective_context = (
                f"\nThe conversation involves {observer} (observing themselves)."
            )
        else:
            perspective_context = (
                f"\nThe question is asked by {observer} about {observed}."
            )

    prompt = f"""You are a knowledge extraction system that reverse-engineers the MINIMAL set of observations needed to answer a question.

# Conversation History
{conversation_text}
{perspective_context}

# Question
{question}

# Correct Answer
{answer}

# Task
Extract the MINIMAL set of atomic propositions, deductions, and peer biographical information from the conversation that would be SUFFICIENT to answer the question correctly.

You must follow the EXACT extraction rules used by the deriver system:

## PART 1: ATOMIC PROPOSITIONS (Explicit + Implicit)

An atomic proposition is:
1. A statement with a SINGLE TRUTH VALUE (evaluable as true or false independently)
2. Contains NO LOGICAL CONNECTIVES (no AND, OR, IF-THEN, UNLESS, etc.)
3. SUFFICIENTLY CONTEXTUALIZED to be meaningful standing alone

**EXPLICIT EXTRACTION** - Directly stated facts:
- Extract propositions directly asserted in the conversation
- Each claim becomes a separate atomic proposition
- Word-for-word or clear paraphrase only

**IMPLICIT EXTRACTION** - Clearly implied facts:
- Extract propositions that are obviously implied by the conversation
- Only include implications that are CERTAIN, not speculative
- Examples:
  * "I graduated from college" → IMPLIES: "X attended college"
  * "I'm taking my dog to the vet" → IMPLIES: "X has a dog"

ONLY extract propositions that are NECESSARY to answer the question.

## PART 2: DEDUCTIVE REASONING

A deductive inference is valid when:
1. The conclusion NECESSARILY follows from the premises
2. If all premises are true, the conclusion MUST be true
3. The reasoning follows the laws of formal logic

**SUBSTANTIVE THRESHOLD:**
Only generate deductions that add meaningful, non-obvious information that is semantically differentiated from the atomic propositions.

- ❌ TRIVIAL: "Maria spoke" → "Maria is alive" (biological necessity, assumed)
- ❌ DEFINITIONAL RESTATEMENT: "Liam went to the store" → "Liam visited a retail establishment" (just rewording)
- ✓ SUBSTANTIVE: "Maria attended college" → "Maria completed high school or equivalent education" (non-obvious precondition)

**PERMITTED PREMISE TYPES:**
1. Atomic propositions (explicit/implicit extracted above)
2. General knowledge - widely accepted facts
3. Temporal information
4. Logical principles

ONLY generate deductions that are NECESSARY to answer the question.

## PART 3: PEER BIOGRAPHICAL CARDS

Extract minimal biographical information that would be SUFFICIENT or HELPFUL to answer the question. A biographical card contains essential PERMANENT information:
- Name, nicknames
- Age, location
- Occupation
- Core interests/hobbies
- Key likes/dislikes
- Other permanent traits

**Guidelines:**
- ONLY extract from conversation messages (NOT from the answer)
- ONLY include info that helps answer the question
- Value permanent properties over transient ones ("is a software engineer" not "wrote Python today")
- Value concision over detail
- Never infer traits from one-off behaviors
- Format: "Name: Alice", "Occupation: Artist", etc.
- Set to null if no relevant biographical info exists

# Response Format
Return JSON with five keys:
- "explicit": array of explicit atomic proposition strings (ONLY those needed to answer question)
- "implicit": array of implicit atomic proposition strings (ONLY those needed to answer question)
- "deductions": array of objects with "premises" (array) and "conclusion" (string) (ONLY those needed to answer question)
- "observer_card": array of biographical strings for {observer or "the observer"}, or null
- "observed_card": array of biographical strings for {observed or "the observed"}, or null

For self-queries (observer == observed), only populate "observer_card".

Example:
{{
  "explicit": ["The assistant recommended using a Pilsner or Lager for the recipe"],
  "implicit": ["A Pilsner is a type of light-bodied beer", "A Lager is a type of light-bodied beer"],
  "deductions": [],
  "observer_card": null,
  "observed_card": null
}}
"""

    return prompt


async def reverse_engineer_trace(
    messages: list[dict[str, str]],
    question: str,
    answer: str,
    *,
    observer: str | None = None,
    observed: str | None = None,
    provider: SupportedProviders | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[ExplicitResponse, DeductiveResponse, list[str] | None, list[str] | None]:
    """
    Call LLM to reverse-engineer atomic propositions, deductions, and peer cards.

    This makes a single LLM call that extracts:
    1. Explicit and implicit atomic propositions
    2. Deductive conclusions with premises
    3. Observer and observed peer biographical cards

    All outputs represent the MINIMAL information sufficient to answer the question.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        question: The question that was asked
        answer: The correct/ground truth answer
        observer: Optional name of the peer asking the question
        observed: Optional name of the peer being asked about
        provider: LLM provider to use (defaults to DERIVER settings)
        model: Model name to use (defaults to DERIVER settings)
        max_tokens: Max output tokens (defaults to DERIVER settings)

    Returns:
        Tuple of (ExplicitResponse, DeductiveResponse, observer_card, observed_card)
        where peer cards are list[str] or None
    """
    prompt = reverse_engineer_prompt(
        messages,
        question,
        answer,
        observer=observer,
        observed=observed,
    )

    response: HonchoLLMCallResponse[ReverseEngineerResponse] = await honcho_llm_call(
        provider=provider or settings.DERIVER.PROVIDER,
        model=model or settings.DERIVER.MODEL,
        prompt=prompt,
        max_tokens=max_tokens or settings.DERIVER.MAX_OUTPUT_TOKENS,
        track_name="Reverse Engineer Trace",
        response_model=ReverseEngineerResponse,
        json_mode=True,
        enable_retry=True,
        retry_attempts=3,
    )

    # Convert to existing schema formats
    from src.utils.representation import (
        DeductiveObservationBase,
        ExplicitObservationBase,
        ImplicitObservationBase,
    )

    # Parse the response
    result: ReverseEngineerResponse = response.content

    # Build ExplicitResponse
    explicit_response = ExplicitResponse(
        explicit=[ExplicitObservationBase(content=e) for e in result.explicit],
        implicit=[ImplicitObservationBase(content=i) for i in result.implicit],
    )

    # Build DeductiveResponse
    deductive_response = DeductiveResponse(
        deductions=[
            DeductiveObservationBase(
                premises=d.premises,
                conclusion=d.conclusion,
            )
            for d in result.deductions
        ]
    )

    logger.debug(
        "Reverse engineered trace: %d explicit, %d implicit, %d deductive, observer_card=%s, observed_card=%s",
        len(explicit_response.explicit),
        len(explicit_response.implicit),
        len(deductive_response.deductions),
        result.observer_card is not None,
        result.observed_card is not None,
    )

    return (
        explicit_response,
        deductive_response,
        result.observer_card,
        result.observed_card,
    )

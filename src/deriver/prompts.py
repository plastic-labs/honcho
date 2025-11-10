"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from functools import cache

from src.config import settings
from src.utils.representation import Representation
from src.utils.templates import render_template
from src.utils.tokens import estimate_tokens


def critical_analysis_prompt(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
) -> str:
    """
    Generate the critical analysis prompt for the deriver.

    Args:
        peer_id (str): The ID of the user being analyzed.
        peer_card (list[str] | None): The bio card of the user being analyzed.
        message_created_at (datetime.datetime): Timestamp of the message.
        working_representation (Representation): Current user understanding context.
        history (str): Recent conversation history.
        new_turns (list[str]): New conversation turns to analyze.

    Returns:
        Formatted prompt string for critical analysis
    """
    return render_template(
        settings.DERIVER.CRITICAL_ANALYSIS_TEMPLATE,
        {
            "peer_id": peer_id,
            "peer_card": peer_card,
            "message_created_at": message_created_at,
            "working_representation": str(working_representation),
            "has_working_representation": not working_representation.is_empty(),
            "history": history,
            "new_turns": new_turns,
        },
    )


def peer_card_prompt(
    old_peer_card: list[str] | None,
    new_observations: str,
) -> str:
    """
    Generate the peer card prompt for the deriver.
    Currently optimized for GPT-5 mini/nano.

    Args:
        old_peer_card: Existing biographical card lines, if any.
        new_observations: Pre-formatted observations block (multiple lines).

    Returns:
        Formatted prompt string for (re)generating the peer card JSON.
    """
    return render_template(
        settings.DERIVER.PEER_CARD_TEMPLATE,
        {
            "old_peer_card": old_peer_card,
            "new_observations": new_observations,
        },
    )


@cache
def estimate_base_prompt_tokens() -> int:
    """Estimate base prompt tokens for explicit and deductive reasoning prompts.

    This value is cached since it only changes on redeploys when prompt templates change.
    Returns the combined token estimate for both reasoning passes.
    """

    try:
        # Estimate explicit reasoning prompt tokens
        explicit_prompt = explicit_reasoning_prompt(
            peer_id="",
            peer_card=None,
            message_created_at=datetime.datetime.now(datetime.timezone.utc),
            working_representation=Representation(),
            history="",
            new_turns=[],
        )
        explicit_tokens = estimate_tokens(explicit_prompt)

        # Estimate deductive reasoning prompt tokens
        deductive_prompt = deductive_reasoning_prompt(
            peer_id="",
            peer_card=None,
            message_created_at=datetime.datetime.now(datetime.timezone.utc),
            working_representation=Representation(),
            atomic_propositions=[],
            history="",
            new_turns=[],
        )
        deductive_tokens = estimate_tokens(deductive_prompt)

        return explicit_tokens + deductive_tokens
    except Exception:
        # Return a conservative estimate if estimation fails
        return 1000  # Increased from 500 since we have two prompts now


def explicit_reasoning_prompt(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
) -> str:
    """
    Generate the explicit reasoning prompt for the deriver.

    Args:
        peer_id (str): The ID of the user being analyzed.
        peer_card (list[str] | None): The bio card of the user being analyzed.
        message_created_at (datetime.datetime): Timestamp of the message.
        working_representation (Representation): Current user understanding context.
        history (str): Recent conversation history.
        new_turns (list[str]): New conversation turns to analyze.

    Returns:
        Formatted prompt string for explicit reasoning
    """
    return render_template(
        settings.DERIVER.EXPLICIT_REASONING_TEMPLATE,
        {
            "peer_id": peer_id,
            "peer_card": peer_card,
            "message_created_at": message_created_at,
            "working_representation": str(working_representation),
            "has_working_representation": not working_representation.is_empty(),
            "history": history,
            "new_turns": new_turns,
        },
    )


def deductive_reasoning_prompt(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    atomic_propositions: list[str],
    history: str,
    new_turns: list[str],
) -> str:
    """
    Generate the deductive reasoning prompt for the deriver.

    Args:
        peer_id (str): The ID of the user being analyzed.
        peer_card (list[str] | None): The bio card of the user being analyzed.
        message_created_at (datetime.datetime): Timestamp of the message.
        working_representation (Representation): Current user understanding context.
        atomic_propositions (list[str]): New atomic propositions from explicit reasoning
            (includes both explicit and implicit observations as content strings).
        history (str): Recent conversation history.
        new_turns (list[str]): New conversation turns to analyze.

    Returns:
        Formatted prompt string for deductive reasoning
    """
    # Format atomic propositions as numbered list
    atomic_propositions_section = "\n".join(
        [f"{i}. {prop}" for i, prop in enumerate(atomic_propositions, 1)]
    )

    # Format existing deductions from working representation
    # Uses the same format as Representation.__str__() for DEDUCTIVE section
    existing_deductions_section = ""
    if working_representation.deductive:
        deduction_strings = [
            f"{i}. {deduction}"
            for i, deduction in enumerate(working_representation.deductive, 1)
        ]
        existing_deductions_section = "\n".join(deduction_strings)

    return render_template(
        settings.DERIVER.DEDUCTIVE_REASONING_TEMPLATE,
        {
            "peer_id": peer_id,
            "peer_card": peer_card,
            "message_created_at": message_created_at,
            "working_representation": str(working_representation),
            "has_working_representation": not working_representation.is_empty(),
            "atomic_propositions_section": atomic_propositions_section,
            "existing_deductions_section": existing_deductions_section,
            "history": history,
            "new_turns": new_turns,
        },
    )

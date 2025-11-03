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
    """Estimate base prompt tokens by calling critical_analysis_prompt with empty values.

    This value is cached since it only changes on redeploys when the prompt template changes.
    """

    try:
        base_prompt = critical_analysis_prompt(
            peer_id="",
            peer_card=None,
            message_created_at=datetime.datetime.now(datetime.timezone.utc),
            working_representation=Representation(),
            history="",
            new_turns=[],
        )
        return estimate_tokens(base_prompt)
    except Exception:
        # Return a conservative estimate if estimation fails
        return 500

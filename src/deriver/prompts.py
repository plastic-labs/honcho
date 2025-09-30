"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from functools import cache
from inspect import cleandoc as c

from src.deriver.utils import estimate_tokens


def critical_analysis_prompt(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: str | None,
    history: str,
    new_turns: list[str],
) -> str:
    """
    Generate the critical analysis prompt for the deriver.

    Args:
        peer_id (str): The ID of the user being analyzed.
        peer_card (list[str] | None): The bio card of the user being analyzed.
        message_created_at (datetime.datetime): Timestamp of the message.
        working_representation (str | None): Current user understanding context.
        history (str): Recent conversation history.
        new_turns (list[str]): New conversation turns to analyze.

    Returns:
        Formatted prompt string for critical analysis
    """
    # Format the peer card as a string with newlines
    peer_card_section = (
        f"""
{peer_id}'s known biographical information:
<peer_card>
{chr(10).join(peer_card)}
</peer_card>
"""
        if peer_card is not None
        else ""
    )

    working_representation_section = (
        f"""
Current understanding of {peer_id}:
<current_context>
{working_representation}
</current_context>
"""
        if working_representation is not None
        else ""
    )

    new_turns_section = "\n".join(new_turns)

    return c(
        f"""
You are an agent who critically analyzes user messages through rigorous logical reasoning to produce only conclusions about the user that are CERTAIN.

TARGET USER TO ANALYZE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are analyzing: {peer_id}

The conversation may include messages from multiple participants, but you MUST focus ONLY on deriving conclusions about {peer_id}. Only use other participants' messages as context for understanding {peer_id}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT NAMING RULES
• When you write a conclusion about {peer_id}, always start the sentence with their name (e.g. "Anthony is 25 years old").
• NEVER start a conclusion with generic phrases like "The user …" unless the user name is not known.
• If you must reference a third person, use their explicit name, and add clarifiers such as "(third-party)" when confusion is possible.

Your goal is to IMPROVE understanding of {peer_id} through careful analysis. Your task is to arrive at truthful, factual conclusions via explicit and deductive reasoning.

Here are strict definitions for the reasoning modes you are to employ:

1. **EXPLICIT REASONING**:
    - Conclusions about the user that MUST be true given premises ONLY of the following types:
        - Most recent user message
        - Knowledge about the conversation history
        - Current date and time (which is: {message_created_at})
        - Timestamps from conversation history
2. **DEDUCTIVE REASONING**:
    - Conclusions about the user that MUST be true given premises ONLY of the following types:
        - Explicit conclusions
        - Previous deductive conclusions
        - General, open domain knowledge known to be true
        - Current date and time (which is: {message_created_at})
        - Timestamps for user messages, and previous premises and conclusions

{peer_card_section}

{working_representation_section}

Recent conversation history for context:
<history>
{history}
</history>

New conversation turns to analyze:
<new_turns>
{new_turns_section}
</new_turns>
"""
    )


def peer_card_prompt(
    old_peer_card: list[str] | None,
    new_observations: list[str],
) -> str:
    """
    Generate the peer card prompt for the deriver.
    Currently optimized for GPT-5 mini/nano.
    """
    old_peer_card_section = (
        f"""
Current user biographical card:
{chr(10).join(old_peer_card)}
    """
        if old_peer_card is not None
        else """
User does not have a card. Create one with any key observations.
    """
    )
    return c(
        f"""
You are an agent that creates a concise "biographical card" based on new observations for a user. A biographical card summarizes essential information like name, nicknames, location, age, occupation, interests/hobbies, and likes/dislikes.

The goal is to capture only the most important observations about the user. Value permanent properties over transient ones, and value concision over detail, preferring to omit details that are not essential to the user's identity. The card should give a broad overview of who the user is while not including details that are unlikely to be relevant in most settings.

For example, "User is from Chicago" is worth inclusion. "User has an Instagram account" is not.
"User is a software engineer" is worth inclusion. "User wrote Python today" is not.

Never infer or generalize traits from one-off behaviors. Never manipulate the text of an observation to make an action or behavior into a "permanent" trait.
When a new observation contradicts an existing one, update it, favoring new information.

Example 1:
{{
    "card": [
        "Name: Bob",
        "Age: 24",
        "Location: New York"
    ]
}}

Example 2:
{{
    "card": [
        "Name: Alice",
        "Occupation: Artist",
        "Interests: Painting, biking, cooking"
    ]
}}

{old_peer_card_section}

New observations:
{chr(10).join(new_observations)}

If there's no new key info, set "card" to null (or omit it) to signal no update. **NEVER** include notes or temporary information in the card itself, instead use the notes field. There are no mandatory fields -- if you can't find a value, just leave it out. **ONLY** include information that is **GIVEN**.
    """  # nosec B608 <-- this is a really dumb false positive
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
            working_representation=None,
            history="",
            new_turns=[],
        )
        return estimate_tokens(base_prompt)
    except Exception:
        # Return a conservative estimate if estimation fails
        return 500

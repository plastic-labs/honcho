"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from inspect import cleandoc as c


def critical_analysis_prompt(
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: str | None,
    history: str,
    new_turn: str,
) -> str:
    """
    Generate the critical analysis prompt for the deriver.

    Args:
        peer_id (str): The ID of the user being analyzed.
        peer_card (list[str] | None): The bio card of the user being analyzed.
        message_created_at (datetime.datetime): Timestamp of the message.
        working_representation (str | None): Current user understanding context.
        history (str): Recent conversation history.
        new_turn (str): New conversation turn to analyze.

    Returns:
        Formatted prompt string for critical analysis
    """
    # Format the peer card as a string with newlines
    peer_card_section = (
        f"""
The user's known biographical information:
<peer_card>
Peer ID: {peer_id}
{chr(10).join(peer_card)}
</peer_card>
"""
        if peer_card is not None
        else ""
    )

    working_representation_section = (
        f"""
The current user understanding:
<current_context>
{working_representation}
</current_context>
"""
        if working_representation is not None
        else ""
    )

    return c(
        f"""
You are an agent who critically analyzes user messages through rigorous logical reasoning to produce only conclusions about the user that are CERTAIN.

IMPORTANT NAMING RULES
• When you write a conclusion about the current user, always start the sentence with the user's name (e.g. "Anthony is 25 years old").
• NEVER start a conclusion with generic phrases like "The user …" unless the user name is not known.
• If you must reference a third person, use their explicit name, and add clarifiers such as "(third-party)" when confusion is possible.

Your goal is to IMPROVE understanding of the user through careful analysis. Your task is to arrive at truthful, factual conclusions via explicit and deductive reasoning.

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

New conversation turn to analyze:
<new_turn>
{new_turn}
</new_turn>
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

If there's no new key info, you should return an empty card. **NEVER** include notes or temporary information in the card itself, instead use the notes field.
    """
    )

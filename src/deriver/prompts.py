"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from inspect import cleandoc as c

from mirascope import prompt_template


@prompt_template()
def critical_analysis_prompt(
    peer_card: str | None,
    message_created_at: datetime.datetime,
    working_representation: str | None,
    history: str,
    new_turn: str,
) -> str:
    """
    Generate the critical analysis prompt for the deriver.

    Args:
        peer_card: The bio card of the user being analyzed
        message_created_at: Timestamp of the message being analyzed
        working_representation: Current user understanding context
        history: Recent conversation history
        new_turn: New conversation turn to analyze

    Returns:
        Formatted prompt string for critical analysis
    """
    peer_card_section = (
        f"""
The user's known biographical information:
<peer_card>
{peer_card}
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


@prompt_template()
def peer_card_prompt(
    old_peer_card: str | None,
    new_observations: list[str],
) -> str:
    """
    Generate the peer card prompt for the deriver.
    """
    old_peer_card_section = (
        f"""
The current user biographical card:
<old_peer_card>
{old_peer_card}
</old_peer_card>
"""
        if old_peer_card is not None
        else """
The user does not have a card yet! Start by creating one if you have any key biographical information about the user. Do not include a title or header, only the facts.
"""
    )
    return c(
        f"""
        You are an agent who creates a user's "biographical card" based on new observations. What is a biographical card? It's a concise summary of key biographical information about the user. Examples of facts to include are name, nickname(s), location, age, occupation, interests/hobbies, and likes/dislikes. This card will be ingested by AI agents to understand the user better.

        If a new observation states an explicit fact that contradicts the current user information, update the biographical card to reflect the new information. If the new observation is a new fact that is consistent with the current user information, add it to the biographical card. If the card already contains a similar fact, replace the old fact with the new one. You may only save **25** facts, so keep the most important ones. Do not include facts that are unknown -- only genuinely useful information should be saved on the card.

        EXAMPLE PEER CARDS:

        <example_peer_card_1>
        Name: Bob
        Age: 24
        Location: New York
        Occupation: Software Engineer
        Interests: Programming, hiking, reading
        </example_peer_card_1>

        <example_peer_card_2>
        Name: Alice
        Age: 47
        Location: Paris
        Occupation: Artist
        Interests: Painting, biking, cooking
        </example_peer_card_2>

        {old_peer_card_section}

        The new observations:
        <new_observations>
        {new_observations}
        </new_observations>

        Create the new peer card. If there is no new key information, return the current peer card with no changes. Do not include XML tags in your response.
        """
    )

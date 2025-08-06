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
    context: str | None,
    history: str,
    new_turn: str,
) -> str:
    """
    Generate the critical analysis prompt for the deriver.

    Args:
        peer_name: The name of the user being analyzed
        message_created_at: Timestamp of the message being analyzed
        context: Current user understanding context
        history: Recent conversation history
        new_turn: New conversation turn to analyze

    Returns:
        Formatted prompt string for critical analysis
    """
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

The user's known biographical information:
<peer_card>
{peer_card if peer_card else "No pre-existing information about the user."}
</peer_card>

Here's the current user understanding
<current_context>
{context}
</current_context>

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

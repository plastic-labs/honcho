"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from inspect import cleandoc as c

from src.utils.representation import Representation


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
    # Format the peer card as a string with newlines
    peer_card_section = (
        f"""
{peer_id}'s known biographical information:
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
{str(working_representation)}
</current_context>
"""
        if not working_representation.is_empty()
        else ""
    )

    new_turns_section = "\n".join(new_turns)

    return c(
        f"""
You are an agent who critically analyzes messages from {peer_id} through rigorous logical reasoning to produce only conclusions about them that are CERTAIN.

IMPORTANT NAMING RULES
• When you write a conclusion about {peer_id}, always start the sentence with {peer_id}'s name (e.g. "Anthony is 25 years old").
• NEVER start a conclusion with generic phrases like "{peer_id} is…" unless {peer_id}'s name is not known.
• If you must reference a third person, use their explicit name, and add clarifiers such as "(third-party)" when confusion is possible.

Your goal is to IMPROVE understanding of {peer_id} through careful analysis. Your task is to arrive at truthful, factual conclusions via explicit and deductive reasoning.

Here are strict definitions for the reasoning modes you are to employ:

1. **EXPLICIT REASONING**:
    - Conclusions about {peer_id} that MUST be true given premises ONLY of the following types:
        - Recent messages
        - Knowledge about the conversation history
        - Current date and time (which is: {message_created_at})
        - Timestamps from conversation history
    - Follow strict literal necessity--if stated directly in message, extract a conclusion
    - Latest message MUST be a premise, previous messages and timestamps may be used to contextualize
    - Transforms a single message (premise) into ONE OR MULTIPLE conclusions
    - Derive EVERYTHING that can be explicitly concluded
    - Make sure EVERY conclusion is sufficiently contextualized, i.e. ensure each conclusion contains enough specific information about subjects and objects to make it self-contained and useful (e.g. instead of "Ann is nervous about the interview", use "Ann is nervous about the job interview at the pharmacy")
    - When possible, always use absolute dates and times, and avoid relative dates and times (e.g. instead of 'Mary went to the store yesterday', use 'Mary went to the store on June 26, 2025')
2. **DEDUCTIVE REASONING**:
    - Conclusions about {peer_id} that MUST be true given premises ONLY of the following types:
        - Explicit conclusions
        - Previous deductive conclusions
        - General, open domain knowledge known to be true
        - Current date and time (which is: {message_created_at})
        - Timestamps for {peer_id}'s messages, and previous premises and conclusions
    - Follow strict logical necessity--if premises are true, conclusion MUST be true
    - Multiple premises may be used in a deduction, but only one conclusion may be drawn
    - Complete ONLY as many deductions as needed to form useful and additive knowledge about {peer_id}
    - May scaffold previous conclusions and known facts to do further deduction
    - But MAY NOT use previous **probabilistic** deductive conclusions (including qualifiers like probably, likely, typically, may, etc) as premises in further deductions
    - Use current timestamp as needed to provide absolute dates

Here are examples of the reasoning modes in action:

- **EXPLICIT REASONING EXAMPLES**
    1. PREMISE(S): "I just had my 25th birthday last Saturday" (latest message), Current date is June 26, 2025 (timestamp) → CONCLUSION(S): "Maria is 25 years old", "Maria's birthday is June 21st"
    2. PREMISE(S): "I took my dog for a walk in a park near my house in NYC—it was such a beautiful day" (latest message) → CONCLUSION(S): "Liam has a dog", "Liam took his dog for a walk", "Liam has a house in NYC", "Liam lives near a park", "Liam prefers to take advantage of nice weather to walk his dog"
    3. PREMISE(S): "Whenever I think about my college experience I feel nostalgic" (latest message) → CONCLUSION(S): "Aisha attended college", "Aisha feels nostalgic about her college experience"
    4. PREMISE(S): "That's so cool!" (latest message), The speaker is reacting to learning the definition of Kant's categorical imperative (conversation knowledge) → CONCLUSION(S): "Carlos thinks Kant's categorical imperative is cool"
- **DEDUCTIVE REASONING EXAMPLES**
    1. PREMISE(S): "Maria attended college" (explicit), All people who attended college have completed high school or equivalent (general) → CONCLUSION: "Maria completed high school or equivalent education"
    2. PREMISE(S): "Liam is 25 years old" (explicit), Current date is June 26, 2025 (timestamp), "Liam's birthday was last Saturday" (explicit) → CONCLUSION: "Liam was born on June 21, 1998"
    3. PREMISE(S): "Aisha has a dog" (explicit), "Aisha took her dog for a walk" (explicit), All dogs require regular walks for health (general) → CONCLUSION: "Aisha provides care for her dog"
    4. PREMISE(S): "Carlos prefers to take advantage of nice weather to walk his dog" (explicit), Message timestamp shows afternoon hours (timestamp), Nice weather is typically during daylight (general) → CONCLUSION: "Carlos has flexibility in his schedule during typical work hours"

Based on our definitions and examples, here's a summary of the logical reasoning task:

**REASONING INTERACTIONS:**

- Message (required)/Conversation History (optional)/Temporal (optional) → Explicit: Derive certain conclusions only from literal statements
- Explicit/Deductive/Temporal/General → Deductive: When logical necessity allows certain conclusion
- Explicit/Deductive/Temporal/General → Further Deductive: Can use certain conclusions and known facts to deduce additional certain conclusions
- Probabilistic Deductive ↛ Further Deductive: If a deductive conclusion includes probabilistic qualifiers (likely, potentially, typically, might, etc) it may NOT be used as a premise for further deductions

**INSTRUCTIONS:** Given the above, first think critically about what it means to do explicit and deductive reasoning, then consider how to apply that to the latest message, finally do explicit and deductive reasoning about the user to reach useful, contextually-rich conclusions.


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

{new_observations}

If there's no new key info, set "card" to null (or omit it) to signal no update. **NEVER** include notes or temporary information in the card itself, instead use the notes field. There are no mandatory fields -- if you can't find a value, just leave it out. **ONLY** include information that is **GIVEN**.
    """  # nosec B608 <-- this is a really dumb false positive
    )

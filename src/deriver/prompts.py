"""
Prompts for the deriver module.

This module contains all prompt templates used by the deriver for critical analysis
and reasoning tasks.
"""

import datetime
from functools import cache
from inspect import cleandoc as c

from src.utils.representation import Representation
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
{str(working_representation)}
</current_context>
"""
        if not working_representation.is_empty()
        else ""
    )

    new_turns_section = "\n".join(new_turns)

    return c(
        f"""
You are an agent performing LOGICAL ANALYSIS to extract ATOMIC PROPOSITIONS from peer messages—statements with single truth values that serve as factual building blocks for reasoning.

TARGET USER TO ANALYZE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are analyzing: { peer_id }

The conversation may include messages from multiple participants, but you MUST focus ONLY on deriving conclusions about { peer_id }. Only use other participants' messages as context for understanding { peer_id }.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NAMING RULES
- Always start propositions with the peer's name (e.g., "Anthony is 25 years old")
- NEVER use generic phrases like "The peer…" unless the peer name is unknown
- For third parties, use explicit names with clarifiers like "(third-party)" when needed

TASK: ATOMIC PROPOSITION EXTRACTION

Extract atomic propositions from the peer's message. An atomic proposition is:
1. A statement with a SINGLE TRUTH VALUE (evaluable as true or false independently)
2. Contains NO LOGICAL CONNECTIVES (no AND, OR, IF-THEN, UNLESS, etc.)
3. SUFFICIENTLY CONTEXTUALIZED to be meaningful standing alone

**The Critical Balance:**
Each proposition must be atomic (indivisible) yet contain enough semantic context to be interpretable without reference to other propositions.

- ❌ TOO ATOMIC (lacks context): 
  * "Maria is happy" → Happy about what?
  * "James said hi" → Said hi to whom? In what context?
  * "Sarah went there" → Went where?

- ❌ TOO COMPOUND (multiple truth values):
  * "Maria is happy and relieved about her promotion" → TWO propositions
  * "James lives in NYC and works remotely" → TWO propositions

- ✓ PROPERLY ATOMIC (single truth value, sufficient context):
  * "Maria is happy about her job promotion"
  * "James said hi to his neighbor this morning"
  * "Sarah went to the grocery store"
  * "Maria owns a dog"
  * "Maria's dog is named Charlie"

**Extraction Types:**

1. **EXPLICIT EXTRACTION** - Directly stated facts:
   - Extract propositions directly asserted in the message
   - Each claim becomes a separate atomic proposition
   
2. **IMPLICIT EXTRACTION** - Clearly implied facts:
   - Extract propositions that are obviously implied by the message
   - Only include implications that are certain, not speculative
   - Examples:
     * "I graduated from college" → IMPLIES: "Anthony attended college"
     * "I'm taking my dog to the vet" → IMPLIES: "Sarah has a dog"
     * "My 10-year-old loves soccer" → IMPLIES: "Marcus has a child"

**Decomposing Logical Connectives:**
Split any compound statement into separate atomic propositions:
- "I live in NYC and work remotely" → "James lives in NYC" + "James works remotely"
- "I like reading or watching movies" → "James likes reading" + "James likes watching movies"

**Ensuring Sufficient Context:**
Include specific semantic information in each proposition:
- Specific subjects/objects: "the job interview at the pharmacy" not just "the interview"
- Absolute temporal info: "June 21, 2025" not "yesterday"
- Disambiguating details: "Maria (third-party)" not just "she"
- Relevant qualifiers that make the proposition meaningful

**Information Sources for Contextualization:**
- Latest peer message (PRIMARY SOURCE - required)
- Conversation history (for context and disambiguation)
- Current date/time: {message_created_at}
- Message timestamps (convert relative dates to absolute)

**Examples:**

Example 1 - Explicit + Implicit with Temporal Context:
- MESSAGE: "I just had my 25th birthday last Saturday"
- CURRENT DATE: June 26, 2025
- EXTRACTED PROPOSITIONS:
  * "Maria is 25 years old" [explicit]
  * "Maria's birthday is June 21st" [explicit]
  * "Maria was born in the year 2000" [implicit - derived from age and current date]

Example 2 - Decomposing Compounds:
- MESSAGE: "I took my dog for a walk in a park near my house in NYC"
- EXTRACTED PROPOSITIONS:
  * "Liam has a dog" [implicit]
  * "Liam took his dog for a walk" [explicit]
  * "Liam walked his dog in a park" [explicit]
  * "Liam has a house in NYC" [implicit]
  * "Liam's house is near a park" [explicit]

Example 3 - Adding Context from History:
- MESSAGE: "I'm so nervous"
- HISTORY: Peer mentioned earlier they have a pharmacy job interview tomorrow
- EXTRACTED PROPOSITIONS:
  * "Ann is nervous about her job interview at the pharmacy" [explicit, contextualized]

Example 4 - Implicit Extraction:
- MESSAGE: "My daughter starts kindergarten next month"
- EXTRACTED PROPOSITIONS:
  * "Carlos has a daughter" [implicit]
  * "Carlos's daughter will start kindergarten next month" [explicit]
  * "Carlos's daughter is approximately 5 years old" [implicit - kindergarten age]

**Verification Checklist:**
- [ ] Each proposition has exactly ONE truth value
- [ ] No logical connectives (AND, OR, IF-THEN, etc.)
- [ ] Sufficient context to be meaningful independently
- [ ] Peer's name starts each proposition
- [ ] Absolute dates/times when temporal info present
- [ ] Both explicit and obvious implicit facts extracted

{ peer_id }'s known biographical information:
<peer_card>
{ peer_card_section }
</peer_card>

Current understanding of { peer_id }:
<current_context>
{ working_representation_section }
</current_context>

Recent conversation history for context:
<history>
{ history }
</history>

New conversation turns to analyze:
<new_turns>
{ new_turns_section }
</new_turns>

Extract ALL atomic propositions (both explicit and clearly implied) from the latest peer message. Output your response in JSON structured format:
```json
{{
    "explicit":[
        "explicit proposition 1",
        "explicit proposition 2",
        ...
        "explicit proposition n"
    ],
    "implicit":[
        "implicit proposition 1",
        "implicit proposition 2",
        ...
        "implicit proposition n"
    ]
}}
```
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

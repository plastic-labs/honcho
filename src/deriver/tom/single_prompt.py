import os

import sentry_sdk
from anthropic import Anthropic
from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

# Place the code below at the beginning of your application to initialize the tracer

# Initialize the Anthropic client
anthropic = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_retries=5,
)

ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


@ai_track("Tom Inference")
@observe(as_type="generation")
async def get_tom_inference_single_prompt(
    chat_history: str, session_id: str, user_representation: str = "None", **kwargs
) -> str:
    with sentry_sdk.start_transaction(op="tom-inference", name="ToM Inference"):
        system_prompt = """You are a system for analyzing conversations to make evidence-based inferences about user mental states.

REQUIREMENTS:
1. Only make inferences that are directly supported by conversation evidence
2. For each inference, cite the specific message that supports it
3. Use uncertainty qualifiers (may, might, possibly) for speculative inferences
4. Do not make assumptions about demographics unless explicitly stated
5. Focus on current mental state and immediate context
6. Consider your own knowledge gaps and violations of expectations (what would surprise you)
7. Always wrap your prediction in <prediction> tags.

OUTPUT FORMAT:
<prediction>
CURRENT STATE:
- Immediate Context: User's current situation
- Active Goals: What user is trying to achieve
- Present Mood: Observable emotional state

SUPPORTED OBSERVATIONS:
- List only behaviors/preferences with direct evidence
- Format: "OBSERVATION: [detail] (SOURCE: [exact message])"

TENTATIVE INFERENCES:
- List possible but uncertain interpretations
- Format: "POSSIBLE: [interpretation] (BASIS: [supporting message])"

KNOWLEDGE GAPS:
- List important unknown information
- Format: "UNKNOWN: [topic/question]"

EXPECTATION VIOLATIONS:
- Based on the above information, if the next message were to surprise you, what could it contain?
- Format: "POTENTIAL SURPRISE: [possible content] [reason] [confidence level]"
- Include 3-5 possible surprises
</prediction>
"""

        messages = [
            {
                "role": "user",
                "content": f"Please analyze this conversation and provide a prediction following the format above:\n{chat_history}",
            }
        ]

        # Add existing user representation if available
        if user_representation != "None":
            messages.append(
                {
                    "role": "user",
                    "content": f"Consider this existing user representation for context, but focus on current state:\n{user_representation}",
                }
            )

        langfuse_context.update_current_observation(
            input=messages, model=ANTHROPIC_MODEL
        )
        message = anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=messages,
            system=system_prompt,
        )
        print(f"tom_inference in single_prompt.py: {message.content[0].text=}")
        message = message.content[0].text
        return message


@ai_track("User Representation")
@observe(as_type="generation")
async def get_user_representation_single_prompt(
    chat_history: str,
    session_id: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    **kwargs,
) -> str:
    with sentry_sdk.start_transaction(
        op="user-representation-inference", name="User Representation"
    ):
        system_prompt = """You are a system for maintaining factual user representations based on conversation history and theory of mind analysis.

Your job is to update the existing user representation (if provided) with the new information from the conversation history and theory of mind analysis.

Copy over information as-is from the existing user representation. Add new information as needed. Only remove content from this section if new information contradicts it. This is especially important for Persistent Information and Tentative Patterns.

If the existing user representation contains sources, copy them over to the new user representation.

Sometimes, especially at the beginning of a conversation, the user representation might contain information from previous conversations. In this case, preserve all previous information unless new information contradicts it, especially in regards to traits, interests, style, and other information about the user that is likely to be useful across conversations.

Always use the format below. If the existing user representation is in another format, convert it to the format below.

Always wrap your output in <representation> tags.

REQUIREMENTS:
1. Distinguish between temporary states and persistent patterns
2. Only incorporate verified information into core profile
3. Track certainty levels for all information
4. Maintain areas of uncertainty explicitly
5. Update representation incrementally

OUTPUT FORMAT:
<representation>
CURRENT STATE:
- ACTIVE CONTEXT: [detail on situation/activity/location] (SOURCE: [exact message]) # Current situation/activity/location
- TEMPORARY CONDITIONS: [detail on immediate circumstances] (SOURCE: [exact message]) # Immediate circumstances
- PRESENT MOOD/ACTIVITY: [what user is doing right now] (SOURCE: [exact message]) # What user is doing right now

PERSISTENT INFORMATION:
- STYLE: [pattern] (SOURCE: [exact message]) # Communication style: observed patterns in language use
- STATEMENT: [fact] (SOURCE: [exact message]) # Explicitly stated information

TENTATIVE PATTERNS:
- LIKELY PATTERN: [pattern] (SOURCE: [exact message]) # Patterns that are almost certain to be true
- POTENTIAL PATTERN: [pattern] (SOURCE: [exact message]) # Patterns that are possible but less likely
- SPECULATIVE PATTERN: [pattern] (SOURCE: [exact message]) # Patterns that are highly uncertain but remotely possible

KNOWLEDGE GAPS:
- List key missing information
- Note areas needing clarification

EXPECTATION VIOLATIONS:
- Based on the above information, if the next message were to surprise you, what could it contain?
- Format: "POTENTIAL SURPRISE: [possible content] [reason] [confidence level]"
- Include 3-5 possible surprises

UPDATES:
- New Information: Recent observations
- Changes: Modified interpretations
- Removals: Information no longer supported
</representation>
"""

        messages = []

        print(f"in single_prompt.py: chat_history: {chat_history}")
        print(f"in single_prompt.py: user_representation: {user_representation}")
        # Build the context message
        context_str = f"CONVERSATION:\n{chat_history}\n\n"
        if tom_inference != "None":
            context_str += f"PREDICTION OF USER MENTAL STATE - MIGHT BE INCORRECT:\n{tom_inference}\n\n"
        if user_representation != "None":
            context_str += f"EXISTING USER REPRESENTATION - INCOMPLETE, TO BE UPDATED:\n{user_representation}"

        messages.append(
            {
                "role": "user",
                "content": f"Please analyze this information and provide an updated user representation:\n{context_str}",
            }
        )

        langfuse_context.update_current_observation(
            input=messages, model=ANTHROPIC_MODEL
        )
        message = anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=messages,
            system=system_prompt,
        )
        message = message.content[0].text
        return message

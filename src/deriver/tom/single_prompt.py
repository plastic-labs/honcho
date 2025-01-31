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

ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"


@ai_track("Tom Inference")
@observe(as_type="generation")
async def get_tom_inference_single_prompt(chat_history: str,
                                    session_id: str,
                                    user_representation: str = "None",
                                    **kwargs
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
                "content": f"Please analyze this conversation and provide a prediction following the format above:\n{chat_history}"
            }
        ]
        
        # Add existing user representation if available
        if user_representation != "None":
            messages.append({
                "role": "user",
                "content": f"Consider this existing user representation for context, but focus on current state:\n{user_representation}"
            })
        
        langfuse_context.update_current_observation(
            input=messages, model=ANTHROPIC_MODEL
        )
        message = anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=messages,
        )
        return message.content[0].text


@ai_track("User Representation")
@observe(as_type="generation")
async def get_user_representation_single_prompt(
    chat_history: str,
    session_id: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    **kwargs
) -> str:
    with sentry_sdk.start_transaction(op="user-representation-inference", name="User Representation"):
        system_prompt = """You are a system for maintaining factual user representations based on conversation history and theory of mind analysis.

Your job is to update the existing user representation (if provided) with the new information from the conversation history and theory of mind analysis.


REQUIREMENTS:
1. Distinguish between temporary states and persistent patterns
2. Only incorporate verified information into core profile
3. Track certainty levels for all information
4. Maintain areas of uncertainty explicitly
5. Update representation incrementally

OUTPUT FORMAT:
<representation>
CURRENT STATE:
- Active Context: Current situation/activity
- Temporary Conditions: Immediate circumstances
- Present Mood/Activity: What user is doing right now

PERSISTENT INFORMATION:
- Communication Style: Observed patterns in language use
- Verified Facts: Explicitly stated information
- Consistent Patterns: Behaviors seen multiple times
- Note: keep as much persistent information as possible from the existing user representation - only remove if new information contradicts it. Append new information as needed

TENTATIVE PATTERNS:
- Possible Traits: Mark confidence (Low/Medium/High)
- Potential Interests: Need more evidence
- Speculative Elements: Clearly marked as unconfirmed

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


        messages = [
        ]

        # Build the context message
        context_str = f"CONVERSATION:\n{chat_history}\n\n"
        if tom_inference != "None":
            context_str += f"PREDICTION OF USER MENTAL STATE - MIGHT BE INCORRECT:\n{tom_inference}\n\n"
        if user_representation != "None":
            context_str += f"EXISTING USER REPRESENTATION - INCOMPLETE, TO BE UPDATED:\n{user_representation}"

        messages.append({
            "role": "user",
            "content": f"Please analyze this information and provide an updated user representation:\n{context_str}"
        })

        langfuse_context.update_current_observation(
            input=messages, model=ANTHROPIC_MODEL
        )
        message = anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=messages,
        )
        return message.content[0].text

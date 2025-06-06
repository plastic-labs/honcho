import logging
from typing import Optional
from pydantic import BaseModel

from sentry_sdk.ai.monitoring import ai_track

from mirascope import llm
from mirascope.integrations.langfuse import with_langfuse
from inspect import cleandoc as c

# Configure logging
logger = logging.getLogger(__name__)


class PotentialSurprise(BaseModel):
    content: str
    reason: str
    confidence_level: float


class UserRepresentation(BaseModel):
    current_state: str
    tentative_patterns: list[str]
    knowledge_gaps: list[str]
    expectation_violations: list[PotentialSurprise]
    updates: list[str]


@ai_track("User Representation")
@with_langfuse()
@llm.call(
    provider="groq", model="llama-3.3-70b-versatile", response_model=UserRepresentation
)
async def get_user_representation_long_term(
    chat_history: str,
    session_id: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    facts: Optional[list[str]] = None,
):
    return c(
        f"""
        You are a system for maintaining factual user representations based on conversation history and theory of mind analysis.

        Your job is to update the existing user representation (if provided) with the new information from the conversation history and theory of mind analysis.

        REQUIREMENTS:
        1. Distinguish between temporary states and persistent patterns
        2. Only incorporate verified information into core profile
        3. Track certainty levels for all information
        4. Maintain areas of uncertainty explicitly
        5. Update representation incrementally
        6. DO NOT generate persistent information - it will be injected separately.


        OUTPUT FORMAT:
        current_state: str
            - Active Context: Current situation/activity
            - Temporary Conditions: Immediate circumstances
            - Present Mood/Activity: What user is doing right now
        tentative_patterns: list[str]
            - Possible Traits: Mark confidence (Low/Medium/High)
            - Potential Interests: Need more evidence
            - Speculative Elements: Clearly marked as unconfirmed
        knowledge_gaps: list[str]
            - List key missing information
            - Note areas needing clarification
        expectation_violations: list
            content: str
            reason: str
            confidence_level: float
            - Based on the above information, if the next message were to surprise you, what could it contain?
            - Include 3-5 possible surprises
        updates: list[str]
            - New Information: Recent observations
            - Changes: Modified interpretations
            - Removals: Information no longer supported

        CONVERSATION:
        {chat_history}

        PREDICTION OF USER MENTAL STATE - MIGHT BE INCORRECT:
        {tom_inference or "Doesn't exist"}

        EXISTING USER REPRESENTATION - INCOMPLETE, TO BE UPDATED:
        {user_representation or "Doesn't exist"}
        """
    )


class InformationPiece(BaseModel):
    quote: str
    category: str
    explanation: str
    semantic_retrieval: str


class InformationExtraction(BaseModel):
    pieces: list[InformationPiece]
    challenge: str


class FactExtraction(BaseModel):
    information_extraction: InformationExtraction
    facts: list[str]


@ai_track("Fact Extraction")
@with_langfuse()
@llm.call(
    provider="google", model="gemini-2.0-flash-lite", response_model=FactExtraction
)
async def extract_facts_long_term(chat_history: str):
    return c(
        f"""
        You are an AI assistant specialized in extracting and formatting relevant information about users from conversations. Your task is to analyze a given conversation and create a list of concise, factual statements about the user. These statements will be stored in a vector embedding database to enhance future interactions.

        Here is the conversation you need to analyze:

        <conversation>
        {chat_history}
        </conversation>

        Instructions:

        1. Carefully read through the conversation. Extract only new facts, from only the last message sent by the user - treat the rest of the conversation only as context. Ignore facts in the last message that are already stated in the conversation.

        2. Identify key new pieces of information from the last message sent by the user that would be valuable for future interactions. Look for:  
        - Personal details (name, age, occupation, location, etc.)
        - Preferences (likes, dislikes, interests, hobbies)
        - Experiences (travel, education, work history)
        - Expressive style (writing style, tone, etc.)
        - Relationships (family, friends, pets)
        - Goals or aspirations
        - Challenges or problems they're facing
        - Opinions or beliefs

        3. For each piece of information you identify:
           a. Verify that it is factual and explicitly stated in the conversation, not inferred.
           b. Formulate it as a concise statement that would aid in semantic retrieval.
           c. Ensure it is not similar to information previously stated in the conversation.

        4. Before providing your final output, wrap your analysis in information_extraction. In this analysis:
           - List each piece of information you've identified.
           - For each piece of information:
             * Quote the relevant part of the conversation.
             * Categorize the information (e.g., personal detail, preference, experience).
             * Explain why you've included this information.
             * Show how you've formulated the fact for optimal semantic retrieval.
           - Discuss any challenges you encountered in extracting or formatting the information.

        5. After your analysis, provide your final output as a list of strings. Each string should be a single fact about the user.
        
        Remember to focus on clear, concise statements that capture key information about the user. Each fact should be worded in a way that will aid its semantic retrieval from a vector embedding database.
        """
    )

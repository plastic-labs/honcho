import logging
from enum import Enum

from mirascope import llm
from mirascope.integrations.langfuse import with_langfuse
from pydantic import BaseModel

from src.config import settings
from src.utils.clients import clients
from src.utils.types import track

logger = logging.getLogger(__name__)


# Enums for strongly typed fields
class InfoType(str, Enum):
    STYLE = "STYLE"
    STATEMENT = "STATEMENT"


class CertaintyLevel(str, Enum):
    LIKELY = "LIKELY"
    POTENTIAL = "POTENTIAL"
    SPECULATIVE = "SPECULATIVE"


# ToM Inference Output Models
class CurrentState(BaseModel):
    immediate_context: str
    active_goals: str
    present_mood: str


class SupportedObservation(BaseModel):
    detail: str
    source: str


class TentativeInference(BaseModel):
    interpretation: str
    basis: str


class KnowledgeGap(BaseModel):
    topic: str


class ExpectationViolation(BaseModel):
    possible_surprise: str
    reason: str
    confidence_level: float


class TomInferenceOutput(BaseModel):
    current_state: CurrentState
    tentative_inferences: list[TentativeInference]
    knowledge_gaps: list[KnowledgeGap]
    expectation_violations: list[ExpectationViolation]


# User Representation Output Models
class SourcedInfo(BaseModel):
    detail: str
    source: str


class UserCurrentState(BaseModel):
    active_context: SourcedInfo
    temporary_conditions: SourcedInfo
    present_mood_activity: SourcedInfo


class PersistentInfo(BaseModel):
    detail: str
    source: str
    info_type: InfoType


class TentativePattern(BaseModel):
    pattern: str
    source: str
    certainty_level: CertaintyLevel


class UserKnowledgeGap(BaseModel):
    missing_info: str


class UserExpectationViolation(BaseModel):
    potential_surprise: str
    reason: str
    confidence_level: float


class UpdateSection(BaseModel):
    new_information: list[SourcedInfo]
    changes: list[SourcedInfo]
    removals: list[SourcedInfo]


class UserRepresentationOutput(BaseModel):
    current_state: UserCurrentState
    persistent_information: list[PersistentInfo]
    tentative_patterns: list[TentativePattern]
    knowledge_gaps: list[UserKnowledgeGap]
    expectation_violations: list[UserExpectationViolation]
    updates: UpdateSection


@track("Tom Inference")
@with_langfuse()
@llm.call(
    provider=(
        settings.LLM.TOM_INFERENCE_PROVIDER
        if settings.LLM.TOM_INFERENCE_PROVIDER != "custom"
        else "openai"
    ),
    model=settings.LLM.TOM_INFERENCE_MODEL,
    response_model=TomInferenceOutput,
    client=clients[settings.LLM.TOM_INFERENCE_PROVIDER],
)
async def tom_inference(
    chat_history: str,
    user_representation: str | None = None,
):
    return f"""
You are a system for analyzing conversations to make evidence-based inferences about user mental states.

REQUIREMENTS:
1. Only make inferences that are directly supported by conversation evidence
2. For each inference, cite the specific message that supports it
3. Use uncertainty qualifiers (may, might, possibly) for speculative inferences
4. Do not make assumptions about demographics unless explicitly stated
5. Focus on current mental state and immediate context
6. Consider your own knowledge gaps and violations of expectations (what would surprise you)

OUTPUT FORMAT:
current_state:
- immediate_context: User's current situation
- active_goals: What user is trying to achieve  
- present_mood: Observable emotional state

tentative_inferences: list of objects with:
- interpretation: Possible but uncertain interpretation
- basis: Supporting message or evidence

knowledge_gaps: list of objects with:
- topic: Important unknown information or question

expectation_violations: list of objects with:
- possible_surprise: What content could surprise you in the next message
- reason: Why this would be surprising based on current information
- confidence_level: Float between 0.0 and 1.0 indicating confidence
- Include 3-5 possible surprises

<conversation>
{chat_history or "Not provided"}
</conversation>

<user_representation>
{user_representation or "Not provided"}
</user_representation>
"""


@track("User Representation")
@with_langfuse()
@llm.call(
    provider=(
        settings.LLM.TOM_INFERENCE_PROVIDER
        if settings.LLM.TOM_INFERENCE_PROVIDER != "custom"
        else "openai"
    ),
    model=settings.LLM.TOM_INFERENCE_MODEL,
    response_model=UserRepresentationOutput,
    client=clients[settings.LLM.TOM_INFERENCE_PROVIDER],
)
async def user_representation(
    chat_history: str,
    user_representation: str | None = None,
    tom_inference: str | None = None,
):
    return f"""
You are a system for maintaining factual user representations based on conversation history and theory of mind analysis.

Your job is to update the existing user representation (if provided) with the new information from the conversation history and theory of mind analysis.

Copy over information as-is from the existing user representation. Add new information as needed. Only remove content from this section if new information contradicts it. This is especially important for Persistent Information and Tentative Patterns.

REQUIREMENTS:
1. Distinguish between temporary states and persistent patterns
2. Only incorporate verified information into core profile
3. Track certainty levels for all information
4. Maintain areas of uncertainty explicitly
5. Update representation incrementally

OUTPUT FORMAT:
current_state:
- active_context: object with "detail" (current situation/activity/location) and "source" (exact message)
- temporary_conditions: object with "detail" (immediate circumstances) and "source" (exact message)
- present_mood_activity: object with "detail" (what user is doing right now) and "source" (exact message)

persistent_information: list of objects with:
- detail: The specific information or pattern
- source: Exact message that supports this
- info_type: Must be exactly "STYLE" for communication patterns or "STATEMENT" for explicit facts

tentative_patterns: list of objects with:
- pattern: The observed pattern
- source: Supporting evidence from specific message
- certainty_level: Must be exactly "LIKELY" (almost certain), "POTENTIAL" (possible), or "SPECULATIVE" (uncertain)

knowledge_gaps: list of objects with:
- missing_info: Key information that is missing or needs clarification

expectation_violations: list of objects with:
- potential_surprise: What could surprise you in the next message
- reason: Why this would be surprising based on current information
- confidence_level: Float between 0.0 and 1.0
- Include 3-5 possible surprises

updates:
- new_information: List of objects with "detail" (recent observation) and "source" (supporting message)
- changes: List of objects with "detail" (modified interpretation) and "source" (supporting message)
- removals: List of objects with "detail" (information no longer supported) and "source" (contradicting message)

<conversation>
{chat_history or "Not provided"}
</conversation>

<existing_user_representation>
{user_representation or "Not provided"}
</existing_user_representation>

<tom_analysis>
{tom_inference or "Not provided"}
</tom_analysis>
"""

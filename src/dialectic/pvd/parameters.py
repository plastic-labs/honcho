"""
PVD parameter generation using LLM.

Generates α, β, γ weights based on query classification.
"""

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.config import settings
from src.dialectic.pvd.classifier import QueryClassification
from src.utils.clients import honcho_llm_call

logger = logging.getLogger(__name__)


@dataclass
class PVDParameters:
    """PVD scoring parameters."""

    alpha: float  # Semantic similarity weight
    beta: float  # Entity-conditioned probability weight (session)
    gamma: float  # Anchor-conditioned probability weight (global)
    reasoning: str


class PVDParametersResponse(BaseModel):
    """Structured output for parameter generation."""

    alpha: float = Field(
        description="Weight for semantic similarity component (0.0-1.0)", ge=0.0, le=1.0
    )
    beta: float = Field(
        description="Weight for entity-conditioned probability component (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    gamma: float = Field(
        description="Weight for anchor-conditioned probability component (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation of parameter choices (1-2 sentences)"
    )


PARAMETER_GENERATION_PROMPT = """You are a retrieval parameter optimizer. Given a query and its classification, determine optimal weights for the PVD scoring function:

**score(v; q) = α*sim(q,v) + β*log p(v|e,a) - γ*log p(v|a)**

Where:
- **α**: Weight for semantic similarity (how relevant is observation v to query q?)
- **β**: Weight for entity-conditioned probability (how typical is v in this **session**?)
- **γ**: Weight for anchor-conditioned probability (how typical is v **globally** for this peer?)

Query: {query}
Classification: {query_type}
Session Context: {has_session}

## Task-Specific Guidelines

**TEMPORAL / EVENT_ORDERING / KNOW_UPDATE:**
- High α (0.6-0.8): Strong semantic relevance to time-related terms
- Low β (0.1-0.3): Session context less important
- High γ (0.5-0.7): Penalize globally common observations (we want novel events)
- Goal: Find time-sensitive, recently-changed information

**PREF_FOLLOW / SUMMARIZATION:**
- Moderate α (0.4-0.6): Semantic relevance important but not dominant
- High β (0.5-0.7): Session patterns matter (if session available)
- Low γ (0.1-0.3): Favor globally consistent patterns
- Goal: Find stable, repeated behaviors

**CONTRADICTION:**
- Moderate α (0.5-0.7): Need semantic relevance
- Moderate β (0.3-0.5): Session context helps identify contradictions
- Moderate γ (0.3-0.5): Need mix of session-specific and global views
- Goal: Find conflicting statements across contexts

**INFO_EXTRACT:**
- High α (0.7-0.9): Semantic relevance is paramount
- Low β (0.1-0.2): Session context less relevant for factual lookup
- Low γ (0.1-0.2): Just find the most relevant fact
- Goal: Maximize semantic matching

**MULTI_SESS:**
- Moderate α (0.5-0.6): Semantic relevance needed
- Low β (0.1-0.2): Explicitly spanning sessions, minimize session bias
- Moderate γ (0.3-0.4): Global patterns relevant
- Goal: Cross-session information retrieval

**ABSTENTION:**
- Moderate α (0.5-0.6): Need to find relevant context to determine if abstention needed
- Moderate β (0.3-0.4): Session context helps
- Low γ (0.2-0.3): Focus on recent context
- Goal: Find information gaps

**INST_FOLLOW:**
- High α (0.6-0.8): Must match instruction semantics closely
- Moderate β (0.2-0.4): Session context may contain relevant instructions
- Low γ (0.1-0.3): Instructions are typically session-specific
- Goal: Execute instructions accurately

## Constraints

1. **Sum to ~1.0**: α + β + γ should be approximately 1.0 (can vary ±0.3 for emphasis)
2. **All non-negative**: All weights must be >= 0.0
3. **Adaptive to session**: If no session context, β should be lower (0.0-0.2)

Respond with optimal α, β, γ and brief reasoning.
"""


async def generate_pvd_parameters(
    query: str, classification: QueryClassification, session_name: str | None
) -> PVDParameters:
    """
    Generate α, β, γ parameters using an LLM.

    The LLM considers:
    - Query classification
    - Whether this is a session-specific or global query
    - Retrieval optimization strategy for the task type

    Args:
        query: The user's query
        classification: Query classification result
        session_name: Session identifier (None if global query)

    Returns:
        PVDParameters with alpha, beta, gamma, and reasoning
    """
    try:
        # Format prompt
        has_session = "Yes" if session_name else "No"
        prompt = PARAMETER_GENERATION_PROMPT.format(
            query=query,
            query_type=classification.query_type,
            has_session=has_session,
        )

        # Create a simple LLM settings object for parameter generation
        from src.config import HonchoSettings

        class ParameterSettings(HonchoSettings):
            PROVIDER = settings.DIALECTIC.PVD.PROVIDER
            MODEL = settings.DIALECTIC.PVD.PARAMETER_MODEL
            BACKUP_PROVIDER = None
            BACKUP_MODEL = None

        # Call LLM with structured output
        response = await honcho_llm_call(
            llm_settings=ParameterSettings(),
            prompt=prompt,
            max_tokens=400,
            temperature=0.0,  # Deterministic
            tools=None,
            tool_executor=None,
            response_model=PVDParametersResponse,
            track_name="PVD Parameter Generator",
        )

        parameters = PVDParameters(
            alpha=response.output.alpha,
            beta=response.output.beta,
            gamma=response.output.gamma,
            reasoning=response.output.reasoning,
        )

        logger.debug(
            f"Generated PVD parameters for {classification.query_type}: "
            f"α={parameters.alpha:.2f}, β={parameters.beta:.2f}, γ={parameters.gamma:.2f} "
            f"- {parameters.reasoning}"
        )

        return parameters

    except Exception as e:
        logger.error(f"Parameter generation failed: {e}", exc_info=True)
        # Fallback to default parameters based on query type
        return _get_default_parameters(classification.query_type, session_name)


def _get_default_parameters(query_type: str, session_name: str | None) -> PVDParameters:
    """
    Get default parameters for a query type.

    Used as fallback when LLM parameter generation fails.

    Args:
        query_type: The query classification type
        session_name: Session identifier (None if global)

    Returns:
        Default PVDParameters for this query type
    """
    # Adjust beta based on session availability
    session_factor = 1.0 if session_name else 0.3

    default_params = {
        "temporal": (0.7, 0.1 * session_factor, 0.6),
        "event_ordering": (0.7, 0.2 * session_factor, 0.6),
        "know_update": (0.7, 0.2 * session_factor, 0.6),
        "pref_follow": (0.5, 0.6 * session_factor, 0.2),
        "summarization": (0.5, 0.6 * session_factor, 0.2),
        "contradiction": (0.6, 0.4 * session_factor, 0.4),
        "info_extract": (0.8, 0.1 * session_factor, 0.1),
        "multi_sess": (0.6, 0.1 * session_factor, 0.4),
        "abstention": (0.6, 0.3 * session_factor, 0.2),
        "inst_follow": (0.7, 0.3 * session_factor, 0.2),
    }

    alpha, beta, gamma = default_params.get(
        query_type,
        (
            settings.DIALECTIC.PVD.DEFAULT_ALPHA,
            settings.DIALECTIC.PVD.DEFAULT_BETA * session_factor,
            settings.DIALECTIC.PVD.DEFAULT_GAMMA,
        ),
    )

    logger.warning(
        f"Using default parameters for {query_type}: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}"
    )

    return PVDParameters(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        reasoning=f"Default parameters for {query_type} (fallback)",
    )

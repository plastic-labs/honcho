"""
Query classification for PVD retrieval.

Classifies queries into benchmark-aligned task types using an LLM.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings
from src.utils.clients import honcho_llm_call

logger = logging.getLogger(__name__)

# Query types aligned with benchmark tasks
QueryType = Literal[
    "abstention",
    "contradiction",
    "event_ordering",
    "info_extract",
    "inst_follow",
    "know_update",
    "multi_sess",
    "pref_follow",
    "summarization",
    "temporal",
]


@dataclass
class QueryClassification:
    """Result of query classification."""

    query_type: QueryType
    reasoning: str
    confidence: float  # 0-1


class QueryClassificationResponse(BaseModel):
    """Structured output for query classification."""

    query_type: QueryType = Field(description="The primary task type for this query")
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this classification was chosen"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0
    )


CLASSIFICATION_PROMPT = """You are a query classifier for a memory retrieval system. Classify the following query into ONE of these benchmark task types:

1. **abstention**: Query requires acknowledging uncertainty or insufficient information
   - Examples: "Do you know...", "Are you aware of...", questions asking about data availability

2. **contradiction**: Query asks about conflicting statements or inconsistencies
   - Examples: "Did they say X or Y?", "What are the contradictions about...", "They mentioned A but also B, which is correct?"

3. **event_ordering**: Query asks about temporal sequence of events
   - Examples: "What happened first, X or Y?", "In what order did...", "Before/after comparisons"

4. **info_extract**: Query asks for specific factual information
   - Examples: "What is...", "Tell me about...", "What are the details of...", factual lookups

5. **inst_follow**: Query contains explicit instructions to follow
   - Examples: "List all...", "Find every...", "Give me...", imperative commands

6. **know_update**: Query asks about changes or updates to information
   - Examples: "What changed about...", "Did they update...", "Has X been modified", "What's new with..."

7. **multi_sess**: Query explicitly spans multiple conversation sessions
   - Examples: "Across all conversations...", "In previous sessions...", "Throughout our chats..."

8. **pref_follow**: Query asks about preferences or typical behaviors
   - Examples: "What does X usually...", "What are their preferences for...", "How do they typically..."

9. **summarization**: Query asks for overview or summary
   - Examples: "Summarize...", "Give me an overview of...", "What's the general..."

10. **temporal**: Query asks about timing, recency, or when something happened
    - Examples: "When did...", "How long ago...", "Recently...", "Latest...", "Most recent..."

Query: {query}

{context_section}

Respond with:
1. Primary task type (one of the above)
2. Brief reasoning (1-2 sentences)
3. Confidence (0.0-1.0)

Note: If a query could fit multiple categories, choose the MOST specific one. For example, "When did X change?" is more specifically "know_update" than "temporal".
"""


async def classify_query(
    query: str, session_context: str | None = None
) -> QueryClassification:
    """
    Classify a dialectic query using an LLM.

    Uses Claude Haiku for speed (~150-250ms latency).

    Args:
        query: The user's query to classify
        session_context: Optional context about the session

    Returns:
        QueryClassification with query_type, reasoning, and confidence
    """
    try:
        # Build context section
        context_section = ""
        if session_context:
            context_section = f"\nSession Context: {session_context}\n"

        # Format prompt
        prompt = CLASSIFICATION_PROMPT.format(
            query=query, context_section=context_section
        )

        # Create a simple LLM settings object for classification
        from src.config import HonchoSettings

        class ClassifierSettings(HonchoSettings):
            PROVIDER = settings.DIALECTIC.PVD.PROVIDER
            MODEL = settings.DIALECTIC.PVD.CLASSIFIER_MODEL
            BACKUP_PROVIDER = None
            BACKUP_MODEL = None

        # Call LLM with structured output
        response = await honcho_llm_call(
            llm_settings=ClassifierSettings(),
            prompt=prompt,
            max_tokens=300,
            temperature=0.0,  # Deterministic
            tools=None,
            tool_executor=None,
            response_model=QueryClassificationResponse,
            track_name="PVD Query Classifier",
        )

        classification = QueryClassification(
            query_type=response.output.query_type,
            reasoning=response.output.reasoning,
            confidence=response.output.confidence,
        )

        logger.debug(
            f"Classified query as '{classification.query_type}' "
            f"(confidence: {classification.confidence:.2f}): {classification.reasoning}"
        )

        return classification

    except Exception as e:
        logger.error(f"Query classification failed: {e}", exc_info=True)
        # Fallback to info_extract as a reasonable default
        return QueryClassification(
            query_type="info_extract",
            reasoning="Classification failed, using default",
            confidence=0.5,
        )

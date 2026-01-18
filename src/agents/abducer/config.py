"""Configuration for the Abducer agent."""

from pydantic import BaseModel, Field


class AbducerConfig(BaseModel):
    """Configuration for hypothesis generation agent.

    The Abducer generates explanatory hypotheses from observations.
    """

    max_hypotheses_per_batch: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of hypotheses to generate in one execution",
    )

    min_premise_count: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Minimum number of premises required to generate a hypothesis",
    )

    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for hypotheses to be stored",
    )

    lookback_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="How many days of observations to consider for hypothesis generation",
    )

    max_premise_retrieval: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of premises to retrieve from database",
    )

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Semantic similarity threshold for grouping related observations",
    )

    enable_hypothesis_merging: bool = Field(
        default=True,
        description="Whether to merge similar hypotheses to avoid duplicates",
    )

    hypothesis_merge_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for merging hypotheses",
    )

"""Configuration for the Predictor agent."""

from pydantic import BaseModel, ConfigDict, Field


class PredictorConfig(BaseModel):
    """Configuration for prediction generation agent."""

    predictions_per_hypothesis: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of predictions to generate per hypothesis",
    )

    min_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum hypothesis confidence to generate predictions from",
    )

    specificity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum specificity score for predictions to be stored",
    )

    max_hypothesis_retrieval: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of hypotheses to retrieve for prediction generation",
    )

    novelty_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for detecting duplicate predictions (higher = more strict)",
    )

    is_blind: bool = Field(
        default=True,
        description="Whether predictions must be blind (no access to future observations)",
    )

    model_config = ConfigDict(frozen=True)

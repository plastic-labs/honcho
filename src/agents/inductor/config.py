"""Configuration for the Inductor agent."""

from pydantic import BaseModel, ConfigDict, Field


class InductorConfig(BaseModel):
    """Configuration for pattern extraction agent."""

    min_predictions_per_pattern: int = Field(
        default=3,
        ge=2,
        le=20,
        description="Minimum number of predictions required to extract a pattern",
    )

    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for grouping similar predictions",
    )

    stability_score_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum stability score for pattern to be stored",
    )

    max_predictions_retrieval: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum number of unfalsified predictions to retrieve",
    )

    pattern_types: list[str] = Field(
        default_factory=lambda: [
            "preference",
            "behavior",
            "personality",
            "tendency",
            "temporal",
            "conditional",
        ],
        description="Types of patterns to extract",
    )

    max_inductions_per_run: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of inductions to create per agent run",
    )

    model_config = ConfigDict(frozen=True)  # type: ignore[reportIncompatibleVariableOverride]

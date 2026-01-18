"""Configuration for the Falsifier agent."""

from pydantic import BaseModel, ConfigDict, Field


class FalsifierConfig(BaseModel):
    """Configuration for falsification agent."""

    max_search_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search iterations to find contradictions",
    )

    contradiction_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to mark prediction as falsified",
    )

    unfalsified_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to mark prediction as unfalsified",
    )

    search_efficiency_target: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Target efficiency score for search process (good searches / total searches)",
    )

    max_predictions_per_run: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of predictions to test per agent run",
    )

    search_result_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to retrieve per query",
    )

    model_config = ConfigDict(frozen=True)  # type: ignore[reportIncompatibleVariableOverride]

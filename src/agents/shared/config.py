"""
Configuration base classes for Honcho agents.

This module provides base configuration classes and utilities
for agent configuration management.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """
    Base configuration class for all Honcho agents.

    All agent-specific configurations should inherit from this class
    to ensure consistent configuration management.
    """

    model: str = Field(
        default="gpt-4o",
        description="LLM model to use for this agent",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM sampling",
    )

    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for LLM response (None = no limit)",
    )

    timeout: int = Field(
        default=60,
        gt=0,
        description="Timeout in seconds for agent execution",
    )

    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts on failure",
    )

    enable_tracing: bool = Field(
        default=True,
        description="Enable provenance tracing for this agent",
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"


class ExtractorConfig(AgentConfig):
    """Configuration for the Extractor agent."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.3  # Lower temperature for more consistent extraction


class AbducerConfig(AgentConfig):
    """Configuration for the Abducer agent (hypothesis generation)."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_hypotheses: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of hypotheses to generate",
    )
    min_hypotheses: int = Field(
        default=1,
        ge=1,
        description="Minimum number of hypotheses to generate",
    )


class PredictorConfig(AgentConfig):
    """Configuration for the Predictor agent (blind prediction generation)."""

    model: str = "gpt-4o"
    temperature: float = 0.8
    max_predictions_per_hypothesis: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Maximum predictions per hypothesis",
    )
    min_predictions_per_hypothesis: int = Field(
        default=2,
        ge=1,
        description="Minimum predictions per hypothesis",
    )
    enforce_blindness: bool = Field(
        default=True,
        description="Strictly enforce that predictor doesn't see premises",
    )


class FalsifierConfig(AgentConfig):
    """Configuration for the Falsifier agent (contradiction search)."""

    model: str = "gpt-4o"
    temperature: float = 0.5
    max_search_iterations: int = Field(
        default=7,
        ge=1,
        le=20,
        description="Maximum adversarial search iterations",
    )
    early_stop_on_contradiction: bool = Field(
        default=True,
        description="Stop searching once a contradiction is found",
    )
    search_creativity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Creativity level for adversarial search queries",
    )


class InductorConfig(AgentConfig):
    """Configuration for the Inductor agent (pattern extraction)."""

    model: str = "gpt-4o"
    temperature: float = 0.6
    min_sources_per_induction: int = Field(
        default=2,
        ge=2,
        description="Minimum source predictions needed for an induction",
    )
    clustering_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for prediction clustering",
    )


class DialecticConfig(AgentConfig):
    """Configuration for the Dialectic agent (query answering)."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tool_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tool calling iterations",
    )


class DreamerConfig(AgentConfig):
    """Configuration for the Dreamer agent (orchestration)."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    enable_surprisal: bool = Field(
        default=True,
        description="Enable surprisal-based sampling",
    )


def create_config_from_dict(
    config_class: type[AgentConfig],
    config_dict: Dict[str, Any],
) -> AgentConfig:
    """
    Create an agent configuration from a dictionary.

    Args:
        config_class: The configuration class to instantiate
        config_dict: Dictionary of configuration values

    Returns:
        Instantiated configuration object

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        return config_class(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

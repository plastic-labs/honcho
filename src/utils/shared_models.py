"""
Shared Pydantic models used by both dialectic and deriver modules.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class ReasoningLevel(str, Enum):
    EXPLICIT = "explicit"
    DEDUCTIVE = "deductive"


class ObservationMetadata(BaseModel):
    """Actual metadata structure from the database."""

    session_context: str = ""
    summary_id: str = ""
    message_id: str | None = None
    level: str | None = None
    session_name: str | None = None
    premises: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    """Observation matching the actual document structure."""

    content: str
    metadata: ObservationMetadata = Field(default_factory=ObservationMetadata)
    created_at: datetime

    def __str__(self) -> str:
        return self.content


class DeductiveObservation(BaseModel):
    """Deductive observation with multiple premises and one conclusion."""

    premises: list[str] = Field(
        description="Supporting premises or evidence for this conclusion",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")


class UnifiedObservation(BaseModel):
    """Unified observation model with conclusion and optional premises.

    This model separates the core observation (conclusion) from its supporting
    evidence (premises), enabling proper embedding generation from conclusions
    while preserving premise information in metadata.
    """

    conclusion: str = Field(description="The actual observation content")
    premises: list[str] = Field(
        description="Optional supporting premises or evidence", default_factory=list
    )
    level: str | None = Field(
        description="Reasoning level (explicit, deductive)", default=None
    )

    @property
    def has_premises(self) -> bool:
        """Check if this observation has premises."""
        return len(self.premises) > 0

    def to_deductive_observation(self) -> DeductiveObservation:
        """Convert to DeductiveObservation for backward compatibility."""
        return DeductiveObservation(conclusion=self.conclusion, premises=self.premises)

    @classmethod
    def from_deductive_observation(
        cls, deductive_obs: DeductiveObservation
    ) -> UnifiedObservation:
        """Create from DeductiveObservation."""
        return cls(conclusion=deductive_obs.conclusion, premises=deductive_obs.premises)

    @classmethod
    def from_string(
        cls, observation: str, level: str | None = None
    ) -> UnifiedObservation:
        """Create from simple string observation (no premises)."""
        return cls(conclusion=observation, level=level)


class ReasoningResponse(BaseModel):
    """Reasoning response with explicit and deductive observation types."""

    explicit: list[str] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog']",
        default_factory=list,
    )
    deductive: list[DeductiveObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities. Each deduction should have premises and a single conclusion.",
        default_factory=list,
    )


class ReasoningResponseWithThinking(ReasoningResponse):
    thinking: str | None = Field(
        description="Critical thinking about what it means to do explicit and deductive reasoning and how to apply it here",
        default=None,
    )


class ObservationContext(BaseModel):
    """Type-safe context container."""

    thinking: str | None = Field(default=None)
    explicit: list[Observation] = Field(default_factory=list)
    deductive: list[Observation] = Field(default_factory=list)

    @property
    def all_observations(self) -> list[Observation]:
        return self.explicit + self.deductive

    def get_by_level(self, level: ReasoningLevel) -> list[Observation]:
        return getattr(self, level.value)

    def add_observation(self, observation: Observation, level: ReasoningLevel) -> None:
        getattr(self, level.value).append(observation)

    @classmethod
    def from_reasoning_response(
        cls,
        response: ReasoningResponse,
        base_metadata: ObservationMetadata | None = None,
    ) -> ObservationContext:
        """Create ObservationContext from ReasoningResponse."""
        context = cls()

        # Add thinking trace if available
        context.thinking = getattr(response, "thinking", None)

        # Add explicit observations
        for conclusion in response.explicit:
            explicit_metadata: ObservationMetadata = (
                base_metadata.model_copy() if base_metadata else ObservationMetadata()
            )
            explicit_metadata.level = "explicit"

            obs = Observation(
                content=conclusion,
                metadata=explicit_metadata,
                created_at=datetime.now(),
            )
            context.add_observation(obs, ReasoningLevel.EXPLICIT)

        # Add deductive observations
        for level_name in ["deductive"]:
            level = ReasoningLevel(level_name)
            structured_obs_list: list[DeductiveObservation] = getattr(
                response, level_name
            )

            for structured_obs in structured_obs_list:
                deductive_metadata: ObservationMetadata = (
                    base_metadata.model_copy()
                    if base_metadata
                    else ObservationMetadata()
                )
                deductive_metadata.level = level_name
                deductive_metadata.premises = structured_obs.premises
                obs = Observation(
                    content=structured_obs.conclusion,
                    metadata=deductive_metadata,
                    created_at=datetime.now(),
                )
                context.add_observation(obs, level)

        return context


class SemanticQueries(BaseModel):
    """Model for semantic query generation responses."""

    queries: list[str] = Field(
        description="List of semantic search queries to retrieve relevant observations"
    )


class ObservationDict(TypedDict, total=False):
    """Type definition for observation dictionary structures."""

    conclusion: str
    content: str
    premises: list[str]
    created_at: str

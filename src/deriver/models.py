"""
Strong Pydantic models for the deriver and agentic systems.

Consolidated models from across the deriver system.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ReasoningLevel(str, Enum):
    EXPLICIT = "explicit"
    DEDUCTIVE = "deductive" 
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"


class ObservationMetadata(BaseModel):
    """Actual metadata structure from the database."""
    session_context: str = ""
    summary_id: str = ""
    last_accessed: str | None = None  # ISO datetime string
    access_count: int = 0
    accessed_sessions: list[str] = Field(default_factory=list)
    message_id: str | None = None
    level: str | None = None
    session_id: str | None = None
    premises: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    """Observation matching the actual document structure."""
    content: str
    metadata: ObservationMetadata = Field(default_factory=ObservationMetadata)
    created_at: datetime
    
    def __str__(self) -> str:
        return self.content


class StructuredObservation(BaseModel):
    """Structured observation for LLM reasoning output - has conclusion and premises."""
    conclusion: str = Field(description="The main conclusion or observation")
    premises: list[str] = Field(
        description="Supporting premises or evidence for this conclusion",
        default_factory=list
    )
    
    def to_observation(self, metadata: ObservationMetadata | None = None) -> Observation:
        """Convert to storage Observation format."""
        if metadata is None:
            metadata = ObservationMetadata()
        metadata.premises = self.premises
        
        return Observation(
            content=self.conclusion,
            metadata=metadata,
            created_at=datetime.now()
        )


class ReasoningResponse(BaseModel):
    """Complete reasoning response with all observation types."""
    thinking: str = Field(
        description="The thinking process of the LLM"
    )
    explicit: list[str] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference",
        default_factory=list
    )
    deductive: list[StructuredObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities",
        default_factory=list
    )
    inductive: list[StructuredObservation] = Field(
        description="Highly probable patterns based on multiple observations - strong generalizations with substantial support",
        default_factory=list
    )
    abductive: list[StructuredObservation] = Field(
        description="Best explanatory hypotheses for all observations - plausible theories about identity/motivations/context",
        default_factory=list
    )


class ObservationContext(BaseModel):
    """Type-safe context container."""
    thinking: str = Field(default="")
    explicit: list[Observation] = Field(default_factory=list)
    deductive: list[Observation] = Field(default_factory=list)
    inductive: list[Observation] = Field(default_factory=list)
    abductive: list[Observation] = Field(default_factory=list)
    
    @property
    def all_observations(self) -> list[Observation]:
        return self.explicit + self.deductive + self.inductive + self.abductive
    
    def get_by_level(self, level: ReasoningLevel) -> list[Observation]:
        return getattr(self, level.value)
    
    def add_observation(self, observation: Observation, level: ReasoningLevel) -> None:
        getattr(self, level.value).append(observation)
    
    @classmethod
    def from_reasoning_response(cls, response: ReasoningResponse, base_metadata: ObservationMetadata | None = None) -> 'ObservationContext':
        """Create ObservationContext from ReasoningResponse."""
        context = cls()

        # Add thinking trace
        context.thinking = response.thinking
        
        # Add explicit observations
        for explicit_content in response.explicit:
            metadata = base_metadata.model_copy() if base_metadata else ObservationMetadata()
            metadata.level = "explicit"
            obs = Observation(
                content=explicit_content,
                metadata=metadata,
                created_at=datetime.now()
            )
            context.add_observation(obs, ReasoningLevel.EXPLICIT)
        
        # Add structured observations
        for level_name in ["deductive", "inductive", "abductive"]:
            level = ReasoningLevel(level_name)
            structured_obs_list = getattr(response, level_name)
            
            for structured_obs in structured_obs_list:
                metadata = base_metadata.model_copy() if base_metadata else ObservationMetadata()
                metadata.level = level_name
                obs = structured_obs.to_observation(metadata)
                context.add_observation(obs, level)
        
        return context


class ReasoningTrace(BaseModel):
    """Clean trace model for debugging."""
    session_id: str
    iterations: int = 0
    success: bool = False
    error_message: str | None = None
    processing_time: float = 0.0


class SemanticQueries(BaseModel):
    """Model for semantic query generation responses."""
    queries: list[str] = Field(
        description="List of semantic search queries to retrieve relevant observations"
    )
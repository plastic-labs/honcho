from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src import models
from src.utils.formatting import parse_datetime_iso


def _strip_microseconds_and_timezone(timestamp: datetime) -> datetime:
    """
    Remove microseconds and timezone info from a datetime for stable string formatting.
    """
    return timestamp.replace(microsecond=0, tzinfo=None)


class ObservationMetadata(BaseModel):
    id: str = Field(default="", description="Document ID for this observation")
    created_at: datetime
    message_ids: list[int]
    session_name: str


class ExplicitObservationBase(BaseModel):
    content: str = Field(description="The explicit observation")


class DeductiveObservationBase(BaseModel):
    premise_ids: list[str] = Field(
        description="Document IDs of premise observations for tree traversal",
        default_factory=list,
    )
    premises: list[str] = Field(
        description="Human-readable premise text for display",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")


class InductiveObservationBase(BaseModel):
    """Base model for inductive observations - patterns, generalizations, and personality insights."""

    source_ids: list[str] = Field(
        description="Document IDs of source observations for tree traversal",
        default_factory=list,
    )
    sources: list[str] = Field(
        description="Human-readable source text for display",
        default_factory=list,
    )
    pattern_type: str = Field(
        description="Type of pattern: 'preference', 'behavior', 'personality', 'tendency', 'correlation'",
        default="pattern",
    )
    conclusion: str = Field(description="The inductive generalization or pattern")
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', 'low'",
        default="medium",
    )


class PromptRepresentation(BaseModel):
    """
    The representation format that is used when getting structured output from an LLM.
    """

    explicit: list[ExplicitObservationBase] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog named Rover']",
        default_factory=list,
    )

    @field_validator("explicit", mode="before")
    @classmethod
    def convert_none_to_empty_list(cls, v: Any) -> Any:
        """Convert None to empty list - handles LLMs returning null instead of []."""
        if v is None:
            return []
        return v


class ExplicitObservation(ExplicitObservationBase, ObservationMetadata):
    """Explicit observation with content and metadata."""

    def __str__(self) -> str:
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] {self.content}"

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        id_prefix = f"[id:{self.id}] " if self.id else ""
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] {self.content}"

    def __hash__(self) -> int:
        """
        Make ExplicitObservation hashable for use in sets.
        """
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for ExplicitObservation objects.
        Two observations are equal if all their fields match.
        """
        if not isinstance(other, ExplicitObservation):
            return False
        return (
            self.content == other.content
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class DeductiveObservation(DeductiveObservationBase, ObservationMetadata):
    """Deductive observation with multiple premises and one conclusion, plus metadata."""

    def __str__(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] {self.conclusion}\n{premises_text}"

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        id_prefix = f"[id:{self.id}] " if self.id else ""
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] {self.conclusion}\n{premises_text}"

    def str_no_timestamps(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        """
        Make DeductiveObservation hashable for use in sets. NOTE: premises are not included in the hash.
        """
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for DeductiveObservation objects.
        Two observations are equal if all their fields match -- NOTE: premises are not included in the equality check.
        """
        if not isinstance(other, DeductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class InductiveObservation(InductiveObservationBase, ObservationMetadata):
    """Inductive observation with sources, pattern type, and confidence, plus metadata."""

    def __str__(self) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"[{_strip_microseconds_and_timezone(self.created_at)}] [{self.confidence}] {self.conclusion}{sources_text}"

    def str_with_id(self) -> str:
        """Format with ID prefix for use by agents that need to reference observations."""
        id_prefix = f"[id:{self.id}] " if self.id else ""
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"{id_prefix}[{_strip_microseconds_and_timezone(self.created_at)}] [{self.confidence}] {self.conclusion}{sources_text}"

    def str_no_timestamps(self) -> str:
        sources_text = ""
        if self.sources:
            source_lines = [f"    - {source}" for source in self.sources]
            sources_text = "\n" + "\n".join(source_lines)
        return f"[{self.confidence}] {self.conclusion}{sources_text}"

    def __hash__(self) -> int:
        """Make InductiveObservation hashable for use in sets."""
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """Define equality for InductiveObservation objects."""
        if not isinstance(other, InductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class Representation(BaseModel):
    """
    A Representation is a traversable and diffable map of observations.
    At the base, we have a list of explicit observations, derived from a peer's messages.

    From there, deductive observations can be made by establishing logical relationships between explicit observations.

    In the future, we can add more levels of reasoning on top of these.

    All of a peer's observations are stored as documents in a collection. These documents can be queried in various ways
    to produce this Representation object.

    Additionally, a "working representation" is a version of this data structure representing the most recent observations
    within a single session.

    A representation can have a maximum number of observations, which is applied individually to each level of reasoning.
    If a maximum is set, observations are added and removed in FIFO order.
    """

    explicit: list[ExplicitObservation] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog']",
        default_factory=list,
    )
    deductive: list[DeductiveObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities. Each deduction should have premises and a single conclusion.",
        default_factory=list,
    )
    inductive: list[InductiveObservation] = Field(
        description="Patterns, generalizations, and personality insights inferred from multiple observations. Higher-level reasoning created by the Dreamer agent.",
        default_factory=list,
    )

    def is_empty(self) -> bool:
        """
        Check if the representation is empty.
        """
        return (
            len(self.explicit) == 0
            and len(self.deductive) == 0
            and len(self.inductive) == 0
        )

    def len(self) -> int:
        """
        Return the total number of observations in the representation.
        """
        return len(self.explicit) + len(self.deductive) + len(self.inductive)

    def diff_representation(self, other: "Representation") -> "Representation":
        """
        Given this and another representation, return a new representation with only observations that are unique to the other.
        Note that this only removes literal duplicates, not semantically equivalent ones.
        """
        diff = Representation()
        diff.explicit = [o for o in other.explicit if o not in self.explicit]
        diff.deductive = [o for o in other.deductive if o not in self.deductive]
        diff.inductive = [o for o in other.inductive if o not in self.inductive]
        return diff

    def merge_representation(
        self, other: "Representation", max_observations: int | None = None
    ):
        """
        Merge another representation object into this one.
        This will automatically deduplicate explicit, deductive, and inductive observations.
        This *preserves order* of observations so that they retain FIFO order.

        NOTE: observations with the *same* timestamp will not have order preserved.
        That's fine though, because they are from the same timestamp...
        """
        # removing duplicates by going list->set->list
        self.explicit = list(set(self.explicit + other.explicit))
        self.deductive = list(set(self.deductive + other.deductive))
        self.inductive = list(set(self.inductive + other.inductive))
        # sort by created_at
        self.explicit.sort(key=lambda x: x.created_at)
        self.deductive.sort(key=lambda x: x.created_at)
        self.inductive.sort(key=lambda x: x.created_at)

        if max_observations:
            self.explicit = self.explicit[-max_observations:]
            self.deductive = self.deductive[-max_observations:]
            self.inductive = self.inductive[-max_observations:]

    def __str__(self) -> str:
        """
        Format representation into a clean, readable string for LLM prompts.
        NOTE: we always strip subsecond precision from the timestamps.

        Returns:
            Formatted string with clear sections and bullet points including temporal metadata
            Example:
            EXPLICIT:
            1. [2025-01-01 12:00:00] The user has a dog named Rover
            2. [2025-01-01 12:01:00] The user's dog is 5 years old
            3. [2025-01-01 12:05:00] The user is 25 years old
            DEDUCTIVE:
            1. [2025-01-01 12:01:00] Rover is 5 years old
                - The user has a dog named Rover
                - The user's dog is 5 years old

        """

        parts: list[str] = []

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(f"{i}. {observation}")
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation}")
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(f"{i}. {observation}")
        parts.append("")

        return "\n".join(parts)

    def str_with_ids(self) -> str:
        """
        Format representation with observation IDs for agents that need to reference/delete observations.

        Returns:
            Formatted string with IDs included
            Example:
            EXPLICIT:
            1. [id:abc123] [2025-01-01 12:00:00] The user has a dog named Rover
            2. [id:def456] [2025-01-01 12:01:00] The user's dog is 5 years old
            DEDUCTIVE:
            1. [id:ghi789] [2025-01-01 12:01:00] Rover is 5 years old
                - The user has a dog named Rover
                - The user's dog is 5 years old
            INDUCTIVE:
            1. [id:jkl012] [2025-01-01 12:05:00] [high] User tends to be methodical
                - id:abc123
                - id:def456
        """
        parts: list[str] = []

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(f"{i}. {observation.str_with_id()}")
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation.str_with_id()}")
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(f"{i}. {observation.str_with_id()}")
        parts.append("")

        return "\n".join(parts)

    def str_no_timestamps(self) -> str:
        """
        Format representation into a clean, readable string for LLM prompts... but without timestamps.

        Returns:
            Formatted string with clear sections and bullet points including temporal metadata
            Example:
            EXPLICIT:
            1. The user has a dog named Rover
            2. The user's dog is 5 years old
            3. The user is 25 years old
            DEDUCTIVE:
            1. Rover is 5 years old
                - The user has a dog named Rover
                - The user's dog is 5 years old
            INDUCTIVE:
            1. [high] User tends to be methodical
                - id:abc123
                - id:def456

        """
        parts: list[str] = []

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(f"{i}. {observation.content}")
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation.str_no_timestamps()}")
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(f"{i}. {observation.str_no_timestamps()}")
        parts.append("")

        return "\n".join(parts)

    def format_as_markdown(self) -> str:
        """
        Format a Representation object as markdown.
        NOTE: we always strip subsecond precision from the timestamps.

        Returns:
            Formatted markdown string
        """

        parts: list[str] = []

        # Add explicit observations
        if self.explicit:
            parts.append("## Explicit Observations\n")
            for obs in self.explicit:
                parts.append(f"{obs}")
            parts.append("")

        # Add deductive observations
        if self.deductive:
            parts.append("## Deductive Observations\n")
            for obs in self.deductive:
                parts.append(f"- {obs.conclusion}")
                if obs.premises:
                    parts.append("   Premises:")
                    for premise in obs.premises:
                        parts.append(f"   - {premise}")
                parts.append("")
            parts.append("")

        # Add inductive observations
        if self.inductive:
            parts.append("## Inductive Observations\n")
            for obs in self.inductive:
                parts.append(f"**Pattern** [{obs.confidence}]: {obs.conclusion}")
                if obs.pattern_type:
                    parts.append(f"   **Type**: {obs.pattern_type}")
                if obs.sources:
                    parts.append("   **Sources**:")
                    for source in obs.sources[:5]:
                        parts.append(f"   - {source}")
                    if len(obs.sources) > 5:
                        parts.append(f"   - ... and {len(obs.sources) - 5} more")
                parts.append("")
            parts.append("")

        return "\n".join(parts)

    @classmethod
    def from_documents(cls, documents: Sequence[models.Document]) -> "Representation":
        return cls(
            explicit=[
                ExplicitObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", []),
                    session_name=doc.session_name,
                )
                for doc in documents
                if doc.level == "explicit"
            ],
            deductive=[
                DeductiveObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", []),
                    session_name=doc.session_name,
                    # Support both top-level and metadata locations for backward compatibility
                    premise_ids=doc.premise_ids
                    or doc.internal_metadata.get("premise_ids", []),
                    premises=doc.internal_metadata.get("premises", []),
                )
                for doc in documents
                if doc.level == "deductive"
            ],
            inductive=[
                InductiveObservation(
                    id=doc.id,
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", []),
                    session_name=doc.session_name,
                    # Support both top-level and metadata locations for backward compatibility
                    source_ids=doc.source_ids
                    or doc.internal_metadata.get("source_ids", []),
                    sources=doc.internal_metadata.get("sources", []),
                    pattern_type=doc.internal_metadata.get("pattern_type", "pattern"),
                    confidence=doc.internal_metadata.get("confidence", "medium"),
                )
                for doc in documents
                if doc.level == "inductive"
            ],
        )

    @classmethod
    def from_prompt_representation(
        cls,
        prompt_representation: "PromptRepresentation",
        message_ids: list[int],
        session_name: str,
        created_at: datetime,
    ) -> "Representation":
        """Convert PromptRepresentation to Representation."""
        return cls(
            explicit=[
                ExplicitObservation(
                    content=e.content,
                    created_at=created_at,
                    message_ids=message_ids,
                    session_name=session_name,
                )
                for e in prompt_representation.explicit
            ],
            deductive=[],
            inductive=[],
        )


def _safe_datetime_from_metadata(
    internal_metadata: dict[str, Any], fallback_datetime: datetime
) -> datetime:
    message_created_at = internal_metadata.get("message_created_at")
    if message_created_at is None:
        return _strip_microseconds_and_timezone(fallback_datetime)

    if isinstance(message_created_at, str):
        try:
            return _strip_microseconds_and_timezone(
                parse_datetime_iso(message_created_at)
            )
        except ValueError:
            return _strip_microseconds_and_timezone(fallback_datetime)

    if isinstance(message_created_at, datetime):
        return _strip_microseconds_and_timezone(message_created_at)
    return _strip_microseconds_and_timezone(fallback_datetime)

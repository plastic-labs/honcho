from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src import models
from src.utils.formatting import parse_datetime_iso


class ObservationMetadata(BaseModel):
    created_at: datetime
    message_ids: list[tuple[int, int]]
    session_name: str


class ExplicitObservationBase(BaseModel):
    content: str = Field(description="The explicit observation")


class ImplicitObservationBase(BaseModel):
    content: str = Field(description="The implicit observation - a fact clearly implied by the message")


class DeductiveObservationBase(BaseModel):
    premises: list[str] = Field(
        description="Supporting premises or evidence for this conclusion",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")


class InductiveObservationBase(BaseModel):
    premises: list[str] = Field(
        description="Supporting specific instances for this generalization",
        default_factory=list,
    )
    conclusion: str = Field(description="The inductive conclusion (probabilistic generalization)")


class AbductiveObservationBase(BaseModel):
    premises: list[str] = Field(
        description="Supporting observations that this explanation accounts for",
        default_factory=list,
    )
    conclusion: str = Field(description="The abductive conclusion (probable explanation)")


class DeductionItem(BaseModel):
    """A single deduction with its conclusion and premises."""

    conclusion: str = Field(description="The deductive conclusion")
    premises: list[str] = Field(
        description="The premises that support this conclusion",
        default_factory=list,
    )


class PromptRepresentation(BaseModel):
    """
    The representation format that is used when getting structured output from an LLM.
    Now follows the new prompt schema with explicit and implicit facts.
    Can also include deductive, inductive, and abductive reasoning results.
    """

    explicit: list[str] = Field(
        description="Facts DIRECTLY stated by the peer - atomic propositions with single truth values. Example: ['Maria is 25 years old', 'Maria has a dog named Rover']",
        default_factory=list,
    )
    implicit: list[str] = Field(
        description="Facts CLEARLY IMPLIED by the peer's message - atomic propositions derived through obvious implication. Example: ['Maria attended college' (from 'I graduated from college')]",
        default_factory=list,
    )
    deductions: list[DeductionItem] = Field(
        description="Deductive conclusions with their supporting premises",
        default_factory=list,
    )
    inductions: list["InductionItem"] = Field(
        description="Inductive generalizations with their supporting instances",
        default_factory=list,
    )
    abductions: list["AbductionItem"] = Field(
        description="Abductive explanations with their supporting observations",
        default_factory=list,
    )


class DeductiveReasoningResponse(BaseModel):
    """
    The response format for deductive reasoning from an LLM.
    Contains a list of deductions, each with a conclusion and supporting premises.
    """

    deductions: list[DeductionItem] = Field(
        description="List of deductive conclusions with their supporting premises",
        default_factory=list,
    )


class InductionItem(BaseModel):
    """A single induction with its conclusion and supporting premises."""

    conclusion: str = Field(description="The inductive conclusion (probabilistic generalization)")
    premises: list[str] = Field(
        description="The specific instances that support this generalization",
        default_factory=list,
    )


class AbductionItem(BaseModel):
    """A single abduction with its conclusion and supporting observations."""

    conclusion: str = Field(description="The abductive conclusion (probable explanation)")
    premises: list[str] = Field(
        description="The observations that this explanation accounts for",
        default_factory=list,
    )


class InductiveReasoningResponse(BaseModel):
    """
    The response format for inductive reasoning from an LLM.
    Contains a list of inductions, each with a probable conclusion and supporting instances.
    """

    inductions: list[InductionItem] = Field(
        description="List of inductive generalizations with their supporting instances",
        default_factory=list,
    )


class AbductiveReasoningResponse(BaseModel):
    """
    The response format for abductive reasoning from an LLM.
    Contains a list of abductions, each with a probable explanation and supporting observations.
    """

    abductions: list[AbductionItem] = Field(
        description="List of abductive explanations with their supporting observations",
        default_factory=list,
    )


class ExplicitObservation(ExplicitObservationBase, ObservationMetadata):
    """Explicit observation with content and metadata."""

    def __str__(self) -> str:
        return f"[{self.created_at.replace(microsecond=0)}] {self.content}"

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

class ImplicitObservation(ImplicitObservationBase, ObservationMetadata):
    """Implicit observation with content and metadata."""

    def __str__(self) -> str:
        return f"[{self.created_at.replace(microsecond=0)}] {self.content}"

    def __hash__(self) -> int:
        """
        Make ImplicitObservation hashable for use in sets.
        """
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for ImplicitObservation objects.
        Two observations are equal if all their fields match.
        """
        if not isinstance(other, ImplicitObservation):
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
        return f"[{self.created_at.replace(microsecond=0)}] {self.conclusion}\n{premises_text}"

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
    """Inductive observation with multiple supporting instances and one probable generalization, plus metadata."""

    def __str__(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"[{self.created_at.replace(microsecond=0)}] {self.conclusion}\n{premises_text}"

    def str_no_timestamps(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        """
        Make InductiveObservation hashable for use in sets. NOTE: premises are not included in the hash.
        """
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for InductiveObservation objects.
        Two observations are equal if all their fields match -- NOTE: premises are not included in the equality check.
        """
        if not isinstance(other, InductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class AbductiveObservation(AbductiveObservationBase, ObservationMetadata):
    """Abductive observation with multiple supporting observations and one probable explanation, plus metadata."""

    def __str__(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"[{self.created_at.replace(microsecond=0)}] {self.conclusion}\n{premises_text}"

    def str_no_timestamps(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        """
        Make AbductiveObservation hashable for use in sets. NOTE: premises are not included in the hash.
        """
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for AbductiveObservation objects.
        Two observations are equal if all their fields match -- NOTE: premises are not included in the equality check.
        """
        if not isinstance(other, AbductiveObservation):
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
    implicit: list[ImplicitObservation] = Field(
        description="Facts CLEARLY IMPLIED by the peer's message - atomic propositions derived through obvious implication. Example: ['Maria attended college' (from 'I graduated from college')]",
        default_factory=list,
    )
    deductive: list[DeductiveObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities. Each deduction should have premises and a single conclusion.",
        default_factory=list,
    )
    inductive: list[InductiveObservation] = Field(
        description="Probable generalizations derived from multiple specific instances - pattern recognition moving from specific to general. Each induction should have supporting instances and a probabilistic conclusion.",
        default_factory=list,
    )
    abductive: list[AbductiveObservation] = Field(
        description="Probable explanations for observed patterns and behaviors - inference to the best explanation. Each abduction should have supporting observations and an explanatory conclusion.",
        default_factory=list,
    )

    def is_empty(self) -> bool:
        """
        Check if the representation is empty.
        """
        return (
            len(self.explicit) == 0
            and len(self.implicit) == 0
            and len(self.deductive) == 0
            and len(self.inductive) == 0
            and len(self.abductive) == 0
        )

    def diff_representation(self, other: "Representation") -> "Representation":
        """
        Given this and another representation, return a new representation with only observations that are unique to the other.
        Note that this only removes literal duplicates, not semantically equivalent ones.
        """
        diff = Representation()
        diff.explicit = [o for o in other.explicit if o not in self.explicit]
        diff.implicit = [o for o in other.implicit if o not in self.implicit]
        diff.deductive = [o for o in other.deductive if o not in self.deductive]
        diff.inductive = [o for o in other.inductive if o not in self.inductive]
        diff.abductive = [o for o in other.abductive if o not in self.abductive]
        return diff

    def merge_representation(
        self, other: "Representation", max_observations: int | None = None
    ):
        """
        Merge another representation object into this one.
        This will automatically deduplicate explicit, implicit, deductive, and inductive observations.
        This *preserves order* of observations so that they retain FIFO order.

        NOTE: observations with the *same* timestamp will not have order preserved.
        That's fine though, because they are from the same timestamp...
        """
        # removing duplicates by going list->set->list
        self.explicit = list(set(self.explicit + other.explicit))
        self.implicit = list(set(self.implicit + other.implicit))
        self.deductive = list(set(self.deductive + other.deductive))
        self.inductive = list(set(self.inductive + other.inductive))
        self.abductive = list(set(self.abductive + other.abductive))

        # sort by created_at
        self.explicit.sort(key=lambda x: x.created_at)
        self.implicit.sort(key=lambda x: x.created_at)
        self.deductive.sort(key=lambda x: x.created_at)
        self.inductive.sort(key=lambda x: x.created_at)
        self.abductive.sort(key=lambda x: x.created_at)

        if max_observations:
            self.explicit = self.explicit[-max_observations:]
            self.implicit = self.implicit[-max_observations:]
            self.deductive = self.deductive[-max_observations:]
            self.inductive = self.inductive[-max_observations:]
            self.abductive = self.abductive[-max_observations:]

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

        parts.append("IMPLICIT:\n")
        for i, observation in enumerate(self.implicit, 1):
            parts.append(f"{i}. {observation}")
        parts.append("")

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation.conclusion}")
            for premise in observation.premises:
                parts.append(f"    - {premise}")
            parts.append("")
        parts.append("")

        parts.append("INDUCTIVE:\n")
        for i, observation in enumerate(self.inductive, 1):
            parts.append(f"{i}. {observation.conclusion}")
            for premise in observation.premises:
                parts.append(f"    - {premise}")
            parts.append("")
        parts.append("")

        parts.append("ABDUCTIVE:\n")
        for i, observation in enumerate(self.abductive, 1):
            parts.append(f"{i}. {observation.conclusion}")
            for premise in observation.premises:
                parts.append(f"    - {premise}")
            parts.append("")
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

        """
        parts: list[str] = []

        parts.append("EXPLICIT:\n")
        for i, observation in enumerate(self.explicit, 1):
            parts.append(f"{i}. {observation.content}")
        parts.append("")

        parts.append("IMPLICIT:\n")
        for i, observation in enumerate(self.implicit, 1):
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

        parts.append("ABDUCTIVE:\n")
        for i, observation in enumerate(self.abductive, 1):
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
        parts.append("## Explicit Observations\n")
        for i, obs in enumerate(self.explicit, 1):
            parts.append(f"{i}. {obs}")
        parts.append("")

        # Add implicit observations
        parts.append("## Implicit Observations\n")
        for i, obs in enumerate(self.implicit, 1):
            parts.append(f"{i}. {obs}")
        parts.append("")

        # Add deductive observations
        parts.append("## Deductive Observations\n")
        for i, obs in enumerate(self.deductive, 1):
            parts.append(f"{i}. **Conclusion**: {obs.conclusion}")
            if obs.premises:
                parts.append("   **Premises**:")
                for premise in obs.premises:
                    parts.append(f"   - {premise}")
            parts.append("")
        parts.append("")

        # Add inductive observations
        parts.append("## Inductive Observations\n")
        for i, obs in enumerate(self.inductive, 1):
            parts.append(f"{i}. **Conclusion**: {obs.conclusion}")
            if obs.premises:
                parts.append("   **Supporting Instances**:")
                for premise in obs.premises:
                    parts.append(f"   - {premise}")
            parts.append("")
        parts.append("")

        # Add abductive observations
        parts.append("## Abductive Observations\n")
        for i, obs in enumerate(self.abductive, 1):
            parts.append(f"{i}. **Explanation**: {obs.conclusion}")
            if obs.premises:
                parts.append("   **Supporting Observations**:")
                for premise in obs.premises:
                    parts.append(f"   - {premise}")
            parts.append("")
        parts.append("")

        return "\n".join(parts)

    @classmethod
    def from_documents(cls, documents: Sequence[models.Document]) -> "Representation":
        return cls(
            explicit=[
                ExplicitObservation(
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                    session_name=doc.session_name,
                )
                for doc in documents
                if doc.internal_metadata.get("level") == "explicit"
            ],
            implicit=[
                ImplicitObservation(
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    content=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                    session_name=doc.session_name,
                )
                for doc in documents
                if doc.internal_metadata.get("level") == "implicit"
            ],
            deductive=[
                DeductiveObservation(
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                    session_name=doc.session_name,
                    premises=doc.internal_metadata.get("premises", []),
                )
                for doc in documents
                if doc.internal_metadata.get("level") == "deductive"
            ],
            inductive=[
                InductiveObservation(
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                    session_name=doc.session_name,
                    premises=doc.internal_metadata.get("premises", []),
                )
                for doc in documents
                if doc.internal_metadata.get("level") == "inductive"
            ],
            abductive=[
                AbductiveObservation(
                    created_at=_safe_datetime_from_metadata(
                        doc.internal_metadata, doc.created_at
                    ),
                    conclusion=doc.content,
                    message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                    session_name=doc.session_name,
                    premises=doc.internal_metadata.get("premises", []),
                )
                for doc in documents
                if doc.internal_metadata.get("level") == "abductive"
            ],
        )

    @classmethod
    def from_prompt_representation(
        cls,
        prompt_representation: "PromptRepresentation",
        message_ids: tuple[int, int],
        session_name: str,
        created_at: datetime,
    ) -> "Representation":
        """
        Convert PromptRepresentation (explicit/implicit/deductions/inductions) to Representation.
        - Explicit facts become ExplicitObservation objects
        - Implicit facts become ImplicitObservation objects
        - Implicit facts are ALSO stored as DeductiveObservation with empty premises (for backward compatibility)
        - Deductions from deductive reasoning become DeductiveObservation with their premises
        - Inductions from inductive reasoning become InductiveObservation with their supporting instances
        """
        # Implicit facts as deductive observations with empty premises
        implicit_as_deductive = [
            DeductiveObservation(
                conclusion=imp,
                created_at=created_at,
                message_ids=[message_ids],
                session_name=session_name,
                premises=[],  # Implicit facts don't have explicit premises
            )
            for imp in prompt_representation.implicit
        ]

        # Proper deductions with premises
        proper_deductions = [
            DeductiveObservation(
                conclusion=ded.conclusion,
                created_at=created_at,
                message_ids=[message_ids],
                session_name=session_name,
                premises=ded.premises,
            )
            for ded in prompt_representation.deductions
        ]

        # Inductions with supporting instances
        inductions = [
            InductiveObservation(
                conclusion=ind.conclusion,
                created_at=created_at,
                message_ids=[message_ids],
                session_name=session_name,
                premises=ind.premises,
            )
            for ind in prompt_representation.inductions
        ]

        # Abductions with supporting observations
        abductions = [
            AbductiveObservation(
                conclusion=abd.conclusion,
                created_at=created_at,
                message_ids=[message_ids],
                session_name=session_name,
                premises=abd.premises,
            )
            for abd in prompt_representation.abductions
        ]

        return cls(
            explicit=[
                ExplicitObservation(
                    content=e,
                    created_at=created_at,
                    message_ids=[message_ids],
                    session_name=session_name,
                )
                for e in prompt_representation.explicit
            ],
            implicit=[
                ImplicitObservation(
                    content=imp,
                    created_at=created_at,
                    message_ids=[message_ids],
                    session_name=session_name,
                )
                for imp in prompt_representation.implicit
            ],
            deductive=implicit_as_deductive + proper_deductions,
            inductive=inductions,
            abductive=abductions,
        )


def _safe_datetime_from_metadata(
    internal_metadata: dict[str, Any], fallback_datetime: datetime
) -> datetime:
    message_created_at = internal_metadata.get("message_created_at")
    if message_created_at is None:
        return fallback_datetime.replace(microsecond=0)

    if isinstance(message_created_at, str):
        try:
            return parse_datetime_iso(message_created_at)
        except ValueError:
            return fallback_datetime.replace(microsecond=0)

    if isinstance(message_created_at, datetime):
        return message_created_at.replace(microsecond=0)
    return fallback_datetime.replace(microsecond=0)

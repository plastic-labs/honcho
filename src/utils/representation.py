from datetime import datetime

from pydantic import BaseModel, Field

from src.config import settings
from src.utils.formatting import parse_datetime_iso, utc_now_iso


class Observation(BaseModel):
    created_at: datetime
    message_id: int | None
    session_name: str | None


class ExplicitObservation(Observation):
    content: str = Field(description="The explicit observation")

    def __str__(self) -> str:
        return f"[{self.created_at}] {self.content}"

    def __hash__(self) -> int:
        """
        Make ExplicitObservation hashable for use in sets.
        """
        return hash((self.content, self.created_at, self.message_id, self.session_name))

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
            and self.message_id == other.message_id
            and self.session_name == other.session_name
        )


class DeductiveObservation(Observation):
    """Deductive observation with multiple premises and one conclusion."""

    premises: list[str] = Field(
        description="Supporting premises or evidence for this conclusion",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")

    def __str__(self) -> str:
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"[{self.created_at}] {self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        """
        Make DeductiveObservation hashable for use in sets.
        """
        return hash(
            (self.conclusion, self.created_at, self.message_id, self.session_name)
        )

    def __eq__(self, other: object) -> bool:
        """
        Define equality for DeductiveObservation objects.
        Two observations are equal if all their fields match.
        """
        if not isinstance(other, DeductiveObservation):
            return False
        return (
            self.conclusion == other.conclusion
            and self.created_at == other.created_at
            and self.message_id == other.message_id
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

    def is_empty(self) -> bool:
        """
        Check if the representation is empty.
        """
        return len(self.explicit) == 0 and len(self.deductive) == 0

    def diff_representation(self, other: "Representation") -> "Representation":
        """
        Given this and another representation, return a new representation with only observations that are unique to the other.
        Note that this only removes literal duplicates, not semantically equivalent ones.
        """
        diff = Representation()
        diff.explicit = [o for o in other.explicit if o not in self.explicit]
        diff.deductive = [o for o in other.deductive if o not in self.deductive]
        return diff

    def __str__(self) -> str:
        """
        Format representation into a clean, readable string for LLM prompts.

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

        parts.append("DEDUCTIVE:\n")
        for i, observation in enumerate(self.deductive, 1):
            parts.append(f"{i}. {observation}")
        parts.append("")

        return "\n".join(parts)

    def format_as_markdown(self) -> str:
        """
        Format a Representation object as markdown.

        Returns:
            Formatted markdown string
        """

        parts: list[str] = []

        # Add explicit observations
        parts.append("## Explicit Observations\n")
        for i, obs in enumerate(self.explicit, 1):
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

        return "\n".join(parts)


class PromptDeductiveObservation(BaseModel):
    """
    The deductive observation format that is used when getting structured output from an LLM.
    """

    conclusion: str = Field(description="The deductive conclusion")
    premises: list[str] = Field(description="The premises of the deductive observation")


class PromptRepresentation(BaseModel):
    """
    The representation format that is used when getting structured output from an LLM.
    """

    explicit: list[str] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference. Example: ['The user is 25 years old', 'The user has a dog named Rover']",
        default_factory=list,
    )
    deductive: list[PromptDeductiveObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities. Each deduction should have premises and a single conclusion.",
        default_factory=list,
    )

    def to_representation(self, message_id: int, session_name: str) -> Representation:
        """
        Convert a PromptRepresentation object to a Representation object.
        """
        return Representation(
            explicit=[
                ExplicitObservation(
                    content=e,
                    created_at=parse_datetime_iso(utc_now_iso()),
                    message_id=message_id,
                    session_name=session_name,
                )
                for e in self.explicit
            ],
            deductive=[
                DeductiveObservation(
                    created_at=parse_datetime_iso(utc_now_iso()),
                    message_id=message_id,
                    session_name=session_name,
                    conclusion=d.conclusion,
                    premises=d.premises,
                )
                for d in self.deductive
            ],
        )


class StoredRepresentation(Representation):
    """
    A representation that is stored in the database with its creation timestamp and the ID of the message that triggered its creation.
    """

    created_at: datetime
    message_id: str
    max_observations: int | None = Field(
        description="The maximum number of observations to store. If None, there is no limit. This limit is applied individually to each level of reasoning.",
        default=settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    )

    def add_single_observation(
        self, observation: ExplicitObservation | DeductiveObservation
    ) -> None:
        if isinstance(observation, ExplicitObservation):
            self.explicit.append(observation)
            if self.max_observations:
                self.explicit: list[ExplicitObservation] = self.explicit[
                    -self.max_observations :
                ]
        else:
            self.deductive.append(observation)
            if self.max_observations:
                self.deductive: list[DeductiveObservation] = self.deductive[
                    -self.max_observations :
                ]

    def merge_representation(self, other: "Representation"):
        """
        Merge another representation object into this one.
        This will automatically deduplicate explicit and deductive observations.
        This *preserves order* of observations so that they retain FIFO order.
        """
        # removing duplicates by going list->set->list
        self.explicit = list(set(self.explicit + other.explicit))
        self.deductive = list(set(self.deductive + other.deductive))
        # sort by created_at
        self.explicit.sort(key=lambda x: x.created_at)
        self.deductive.sort(key=lambda x: x.created_at)

        if self.max_observations:
            self.explicit = self.explicit[-self.max_observations :]
            self.deductive = self.deductive[-self.max_observations :]

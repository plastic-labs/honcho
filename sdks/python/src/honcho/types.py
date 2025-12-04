"""Shared types for the Honcho SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import Required, TypedDict

from pydantic import BaseModel, Field

# Re-export observation types from dedicated module
from .observations import AsyncObservationScope, Observation, ObservationScope

if TYPE_CHECKING:
    from .base import SessionBase

__all__ = [
    "AsyncObservationScope",
    "DeductiveObservation",
    "DeductiveObservationBase",
    "DialecticStreamResponse",
    "ExplicitObservation",
    "ExplicitObservationBase",
    "Observation",
    "ObservationCreateParam",
    "ObservationMetadata",
    "ObservationScope",
    "PeerContext",
    "Representation",
]


class ObservationCreateParam(TypedDict, total=False):
    """Parameters for creating an observation.

    Attributes:
        content: The observation content/text (required)
        session_id: The session this observation relates to (ID string or Session object) (required)
    """

    content: Required[str]
    session_id: "Required[str | SessionBase]"


class ObservationMetadata(BaseModel):
    """Metadata associated with an observation."""

    created_at: datetime
    message_ids: list[int]
    session_name: str


class ExplicitObservationBase(BaseModel):
    """Base model for explicit observations - facts literally stated."""

    content: str = Field(description="The explicit observation")


class DeductiveObservationBase(BaseModel):
    """Base model for deductive observations - logical conclusions."""

    premises: list[str] = Field(
        description="Supporting premises or evidence for this conclusion",
        default_factory=list,
    )
    conclusion: str = Field(description="The deductive conclusion")


class ExplicitObservation(ExplicitObservationBase, ObservationMetadata):
    """
    Explicit observation with content and metadata.
    Represents facts LITERALLY stated - direct quotes or clear paraphrases only.
    """

    def __str__(self) -> str:
        """Format observation with timestamp and content."""
        return f"[{self.created_at.replace(microsecond=0)}] {self.content}"

    def __hash__(self) -> int:
        """
        Make ExplicitObservation hashable for use in sets.
        """
        return hash((self.content, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for ExplicitObservation objects.
        Two observations are equal if content, created_at, and session_name match.
        NOTE: message_ids are not included in the equality check.
        """
        if not isinstance(other, ExplicitObservation):
            return False
        return (
            self.content == other.content
            and self.created_at == other.created_at
            and self.session_name == other.session_name
        )


class DeductiveObservation(DeductiveObservationBase, ObservationMetadata):
    """
    Deductive observation with multiple premises and one conclusion, plus metadata.
    Represents conclusions that MUST be true given explicit facts and premises.
    """

    def __str__(self) -> str:
        """Format observation with timestamp, conclusion, and premises."""
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"[{self.created_at.replace(microsecond=0)}] {self.conclusion}\n{premises_text}"

    def str_no_timestamps(self) -> str:
        """Format observation without timestamps."""
        premises_text = "\n".join(f"    - {premise}" for premise in self.premises)
        return f"{self.conclusion}\n{premises_text}"

    def __hash__(self) -> int:
        """
        Make DeductiveObservation hashable for use in sets.
        NOTE: premises are not included in the hash.
        """
        return hash((self.conclusion, self.created_at, self.session_name))

    def __eq__(self, other: object) -> bool:
        """
        Define equality for DeductiveObservation objects.
        Two observations are equal if all their fields match.
        NOTE: premises are not included in the equality check.
        """
        if not isinstance(other, DeductiveObservation):
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
    From there, deductive observations can be made by establishing logical relationships
    between explicit observations.

    All of a peer's observations are stored as documents in a collection. These documents
    can be queried in various ways to produce this Representation object.

    A "working representation" is a version of this data structure representing the most
    recent observations within a single session.

    A representation can have a maximum number of observations, which is applied
    individually to each level of reasoning. If a maximum is set, observations are
    added and removed in FIFO order.
    """

    explicit: list[ExplicitObservation] = Field(
        description="Facts LITERALLY stated by the user - direct quotes or clear paraphrases only, no interpretation or inference",
        default_factory=list,
    )
    deductive: list[DeductiveObservation] = Field(
        description="Conclusions that MUST be true given explicit facts and premises - strict logical necessities",
        default_factory=list,
    )

    def is_empty(self) -> bool:
        """
        Check if the representation is empty.
        """
        return len(self.explicit) == 0 and len(self.deductive) == 0

    def diff_representation(self, other: "Representation") -> "Representation":
        """
        Given this and another representation, return a new representation with only
        observations that are unique to the other.

        Note: This only removes literal duplicates, not semantically equivalent ones.

        Args:
            other: The representation to compare against

        Returns:
            A new Representation containing only observations unique to other
        """
        diff = Representation()
        diff.explicit = [o for o in other.explicit if o not in self.explicit]
        diff.deductive = [o for o in other.deductive if o not in self.deductive]
        return diff

    def merge_representation(
        self, other: "Representation", max_observations: int | None = None
    ) -> None:
        """
        Merge another representation object into this one.

        This will automatically deduplicate explicit and deductive observations.
        This *preserves order* of observations so that they retain FIFO order.

        NOTE: observations with the *same* timestamp will not have order preserved.
        That's fine though, because they are from the same timestamp...

        Args:
            other: The representation to merge into this one
            max_observations: Optional maximum number of observations to keep per type
        """
        # removing duplicates by going list->set->list
        self.explicit = list(set(self.explicit + other.explicit))
        self.deductive = list(set(self.deductive + other.deductive))
        # sort by created_at
        self.explicit.sort(key=lambda x: x.created_at)
        self.deductive.sort(key=lambda x: x.created_at)

        if max_observations:
            self.explicit = self.explicit[-max_observations:]
            self.deductive = self.deductive[-max_observations:]

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
            Formatted string with clear sections and bullet points without temporal metadata

        Example:
            EXPLICIT:
            1. The user has a dog named Rover
            2. The user's dog is 5 years old

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
            parts.append(f"{i}. {observation.str_no_timestamps()}")
        parts.append("")

        return "\n".join(parts)

    def format_as_markdown(self) -> str:
        """
        Format a Representation object as markdown.
        NOTE: we always strip subsecond precision from the timestamps.

        Returns:
            Formatted markdown string with headers and lists
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Representation":
        """
        Create a Representation from a dictionary (typically from API response).

        Args:
            data: Dictionary containing 'explicit' and 'deductive' observation lists

        Returns:
            A new Representation instance

        Raises:
            ValidationError: If observation data is missing required fields
        """
        explicit_data: Any = data.get("explicit", [])
        deductive_data: Any = data.get("deductive", [])

        explicit_list = cast(
            list[Any], explicit_data if isinstance(explicit_data, list) else []
        )
        deductive_list = cast(
            list[Any], deductive_data if isinstance(deductive_data, list) else []
        )

        return cls(
            explicit=[ExplicitObservation(**obs) for obs in explicit_list],
            deductive=[DeductiveObservation(**obs) for obs in deductive_list],
        )


class DialecticStreamResponse:
    """
    Iterator for streaming dialectic responses with utilities for accessing the final response.

    Similar to OpenAI and Anthropic streaming patterns, this allows you to:
    - Iterate over chunks as they arrive
    - Access the final accumulated response after streaming completes

    Works with both sync and async iterators.

    Example (sync):
        ```python
        stream = peer.chat("Hello", stream=True)

        # Stream chunks
        for chunk in stream:
            print(chunk, end="", flush=True)

        # Get final response object
        final = stream.get_final_response()
        print(f"\\nFull content: {final['content']}")
        ```

    Example (async):
        ```python
        stream = await peer.chat("Hello", stream=True)

        # Stream chunks
        async for chunk in stream:
            print(chunk, end="", flush=True)

        # Get final response object
        final = stream.get_final_response()
        print(f"\\nFull content: {final['content']}")
        ```
    """

    _iterator: Iterator[str] | AsyncIterator[str]
    _accumulated_content: list[str]
    _is_complete: bool

    def __init__(self, iterator: Iterator[str] | AsyncIterator[str]):
        self._iterator = iterator
        self._accumulated_content = []
        self._is_complete = False

    # Sync iterator protocol
    def __iter__(self):
        if isinstance(self._iterator, Iterator):
            return self
        else:
            raise TypeError("iterator must be an sync iterator, got async iterator")

    def __next__(self) -> str:
        try:
            if not isinstance(self._iterator, Iterator):
                raise TypeError("iterator must be an sync iterator, got async iterator")
            chunk = next(self._iterator)
            self._accumulated_content.append(chunk)
            return chunk
        except StopIteration:
            self._is_complete = True
            raise

    # Async iterator protocol
    def __aiter__(self):
        if isinstance(self._iterator, AsyncIterator):
            return self
        else:
            raise TypeError("iterator must be an async iterator, got sync iterator")

    async def __anext__(self) -> str:
        try:
            if not isinstance(self._iterator, AsyncIterator):
                raise TypeError("iterator must be an async iterator, got sync iterator")
            chunk = await self._iterator.__anext__()
            self._accumulated_content.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._is_complete = True
            raise

    def get_final_response(self) -> dict[str, str]:
        """
        Get the final accumulated response after streaming completes.

        Returns:
            A dictionary with the full content: {"content": "full accumulated text"}

        Note:
            This should be called after the stream has been fully consumed.
            If called before completion, it returns the content accumulated so far.
        """
        return {"content": "".join(self._accumulated_content)}

    @property
    def is_complete(self) -> bool:
        """Check if the stream has finished."""
        return self._is_complete


class PeerContext:
    """
    Context for a peer, including representation and peer card.

    This class holds both the working representation and peer card for a peer,
    typically returned from the get_context API call.

    Attributes:
        peer_id: The ID of the observer peer
        target_id: The ID of the target peer being observed
        representation: The working representation (may be None if no observations exist)
        peer_card: List of peer card strings (may be None if no card exists)
    """

    peer_id: str
    target_id: str
    representation: Representation | None
    peer_card: list[str] | None

    def __init__(
        self,
        peer_id: str,
        target_id: str,
        representation: Representation | None = None,
        peer_card: list[str] | None = None,
    ):
        self.peer_id = peer_id
        self.target_id = target_id
        self.representation = representation
        self.peer_card = peer_card

    @classmethod
    def from_api_response(cls, response: Any) -> "PeerContext":
        """
        Create a PeerContext from an API response.

        Args:
            response: API response object with peer_id, target_id, representation, and peer_card

        Returns:
            A new PeerContext instance
        """
        peer_id = getattr(response, "peer_id", "") or ""
        target_id = getattr(response, "target_id", "") or ""

        representation = None
        rep_data = getattr(response, "representation", None)
        if rep_data is not None:
            if isinstance(rep_data, dict):
                representation = Representation.from_dict(
                    cast(dict[str, Any], rep_data)
                )
            elif hasattr(rep_data, "explicit") and hasattr(rep_data, "deductive"):
                representation = Representation.from_dict(
                    {
                        "explicit": rep_data.explicit,
                        "deductive": rep_data.deductive,
                    }
                )

        peer_card = getattr(response, "peer_card", None)

        return cls(
            peer_id=peer_id,
            target_id=target_id,
            representation=representation,
            peer_card=peer_card,
        )

    def __repr__(self) -> str:
        has_rep = self.representation is not None
        has_card = self.peer_card is not None and len(self.peer_card) > 0
        return (
            f"PeerContext(peer_id={self.peer_id!r}, target_id={self.target_id!r}, "
            f"has_representation={has_rep}, has_peer_card={has_card})"
        )

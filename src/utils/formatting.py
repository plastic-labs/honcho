"""
Shared formatting utility functions for both dialectic and deriver modules.

This module contains helper functions for processing observations, formatting context,
and handling temporal metadata for the reasoning system.
"""

from datetime import datetime, timezone
from typing import Any, Protocol, cast, runtime_checkable

from src.utils.logging import conditional_observe
from src.utils.shared_models import ReasoningResponse


def format_datetime_utc(dt: datetime) -> str:
    """
    Format datetime to ISO 8601 string with Z suffix for UTC timezone.

    This ensures consistent datetime formatting across the entire backend,
    using the Z format which is the ISO 8601 standard for UTC and matches
    Pydantic's JSON serialization behavior.

    Args:
        dt: datetime object (should be timezone-aware)

    Returns:
        ISO 8601 formatted string with Z suffix for UTC

    Example:
        >>> dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        >>> format_datetime_utc(dt)
        '2023-01-01T12:00:00Z'
    """
    if dt.tzinfo is None:
        # If no timezone info, assume UTC
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC if not already
    if dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    # Format and replace +00:00 with Z
    return dt.isoformat().replace("+00:00", "Z")


def utc_now_iso() -> str:
    """
    Get current UTC time as ISO 8601 string with Z suffix.

    Returns:
        Current UTC time in ISO 8601 format with Z suffix

    Example:
        >>> utc_now_iso()
        '2023-01-01T12:34:56.789123Z'
    """
    return format_datetime_utc(datetime.now(timezone.utc))


def parse_datetime_iso(iso_string: str) -> datetime:
    """
    Parse ISO 8601 datetime string, handling both Z and +00:00 UTC formats.

    This function handles the fact that Python's fromisoformat() doesn't
    directly support the 'Z' suffix, which is the standard ISO 8601 way
    to represent UTC timezone.

    Args:
        iso_string: ISO 8601 formatted datetime string

    Returns:
        datetime object with timezone information

    Example:
        >>> parse_datetime_iso('2023-01-01T12:00:00Z')
        datetime.datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
        >>> parse_datetime_iso('2023-01-01T12:00:00+00:00')
        datetime.datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    # Convert Z format to +00:00 format for Python's fromisoformat
    normalized_string = iso_string.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized_string)


@runtime_checkable
class StructuredObservation(Protocol):
    """Protocol for observations that have conclusion and premises attributes."""

    conclusion: str
    premises: list[str]


REASONING_LEVELS: list[str] = ["explicit", "deductive"]
LEVEL_LABELS: dict[str, str] = {
    "explicit": "Explicit (Literal facts directly stated by the user)",
    "deductive": "Deductive (Logically necessary conclusions from explicit facts)",
}


def format_premises_for_display(premises: list[str]) -> str:
    """
    Format premises as a clean bulleted list for display.

    Args:
        premises: List of premise strings

    Returns:
        Formatted premises text with newlines and bullets, or empty string if no premises
    """
    if not premises:
        return ""

    premises_formatted: list[str] = []
    for premise in premises:
        premises_formatted.append(f"    - {premise}")
    return "\n" + "\n".join(premises_formatted)


def format_structured_observation(conclusion: str, premises: list[str]) -> str:
    """
    Format a structured observation with conclusion and premises for display.

    Args:
        conclusion: The main conclusion
        premises: List of supporting premises

    Returns:
        Formatted observation string
    """
    premises_text = format_premises_for_display(premises)
    return f"{conclusion}{premises_text}"


def extract_observation_content(observation: str | dict[str, Any] | Any) -> str:
    """Extract content string from an observation (dict or string)."""
    # Handle StructuredObservation objects (Pydantic models)
    if isinstance(observation, StructuredObservation):
        return format_structured_observation(
            observation.conclusion, observation.premises
        )

    # Handle explicit observations as simple strings
    if isinstance(observation, str):
        return observation

    if isinstance(observation, dict):
        # For explicit observations with conclusions
        if "conclusions" in observation:
            conclusions_value: str = cast(str, observation["conclusions"])
            if isinstance(conclusions_value, list):
                return "; ".join(cast(list[str], conclusions_value))
            return conclusions_value
        # For structured observations, return conclusion with premises formatted
        if "conclusion" in observation:
            conclusion: str = cast(str, observation["conclusion"])
            premises: list[str] = cast(list[str], observation.get("premises", []))  # pyright: ignore
            return format_structured_observation(conclusion, premises)
        # Fallback to content field or string representation
        content_value: str | None = observation.get("content")  # pyright: ignore
        return content_value if content_value is not None else str(observation)  # pyright: ignore
    return str(observation)


def format_new_turn_with_timestamp(
    new_turn: str, current_time: str | datetime, speaker: str
) -> str:
    """
    Format new turn message with optional timestamp.

    Args:
        new_turn: The message content
        current_time: Timestamp string or "unknown"
        speaker: The speaker's name

    Returns:
        Formatted string like "2023-05-08 13:56:00 speaker: hello" or "speaker: hello"
    """
    if isinstance(current_time, datetime):
        current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    if current_time and current_time != "unknown":
        return f"{current_time} {speaker}: {new_turn}"
    return f"{speaker}: {new_turn}"


def format_context_for_prompt(
    context: ReasoningResponse | dict[str, Any] | None,
) -> str:
    """
    Format context into a clean, readable string for LLM prompts.

    Args:
        context: ReasoningResponse object or dict with reasoning levels as keys and observation lists as values
                Observations can be strings or dicts - will be normalized

    Returns:
        Formatted string with clear sections and bullet points including temporal metadata
    """
    if not context:
        return "No context available."

    formatted_sections: list[str] = []

    # Handle both ReasoningResponse objects and dicts
    if isinstance(context, ReasoningResponse):
        # It's a ReasoningResponse object
        observations_by_level = {
            "explicit": context.explicit,
            "deductive": context.deductive,
        }
    else:
        # It's a dict
        observations_by_level = context
    # Process each level in a consistent order
    for level in REASONING_LEVELS:
        observations = observations_by_level.get(level, [])
        if not observations:
            continue

        label = LEVEL_LABELS.get(level, level.title())
        formatted_sections.append(f"{label}:")

        # Format observations with temporal metadata when available
        for observation in observations:
            observation_content = extract_observation_content(observation)
            formatted_sections.append(f"  • {observation_content}")

        formatted_sections.append("")  # Blank line between sections

    # Remove trailing blank line if exists
    if formatted_sections and formatted_sections[-1] == "":
        formatted_sections.pop()

    return (
        "\n".join(formatted_sections)
        if formatted_sections
        else "No relevant context available."
    )


def normalize_observations_for_comparison(observations: list[Any]) -> set[str]:
    """Convert observations to normalized strings for comparison."""
    normalized: set[str] = set()
    for observation in observations:
        observation_content = extract_observation_content(observation)
        normalized.add(observation_content.strip().lower())
    return normalized


@conditional_observe
def find_new_observations(
    original_context: ReasoningResponse, revised_observations: ReasoningResponse
) -> dict[str, Any]:
    """
    Find observations that are new in revised_observations compared to original_context.

    Args:
        original_context: Original observation context
        revised_observations: Revised observation context

    Returns:
        Dictionary with new observations by level
    """
    new_observations_by_level: dict[str, Any] = {}

    for level in REASONING_LEVELS:
        original_observations = normalize_observations_for_comparison(
            getattr(original_context, level, [])
        )
        revised_list = getattr(revised_observations, level, [])

        # Find genuinely new observations
        new_observations: list[Any] = []
        for observation in revised_list:
            normalized_observation = (
                extract_observation_content(observation).strip().lower()
            )
            if normalized_observation not in original_observations:
                new_observations.append(observation)

        new_observations_by_level[level] = new_observations

    return new_observations_by_level

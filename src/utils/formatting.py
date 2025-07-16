"""
Shared formatting utility functions for both dialectic and deriver modules.

This module contains helper functions for processing observations, formatting context,
and handling temporal metadata for the reasoning system.
"""

from datetime import datetime
from typing import Any, Protocol, cast, runtime_checkable

from langfuse.decorators import observe  # pyright: ignore

from src.utils.shared_models import ReasoningResponse


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
    else:
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


def format_datetime_simple(dt: datetime | str | Any) -> str:
    """
    Format datetime object to simple format matching new turn formatting.
    Converts from ISO format like '2025-06-02T19:43:41.392640+00:00'
    to simple format like '2025-06-03 20:23:43'

    Args:
        dt: datetime object or datetime string

    Returns:
        Formatted datetime string in simple format
    """
    if isinstance(dt, datetime):
        # It's a datetime object
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(dt, str):
        # It's a string - try to parse it first
        try:
            # Handle ISO format strings
            if "T" in dt:
                parsed_dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Already in simple format
                return dt
        except ValueError:
            # If parsing fails, return as-is
            return dt
    else:
        # Fallback
        return str(dt)


def normalize_observations_for_comparison(observations: list[Any]) -> set[str]:
    """Convert observations to normalized strings for comparison."""
    normalized: set[str] = set()
    for observation in observations:
        observation_content = extract_observation_content(observation)
        normalized.add(observation_content.strip().lower())
    return normalized


@observe()
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

    # Helper function to get observations from either ReasoningResponse or dict
    def get_observations(
        context: ReasoningResponse | dict[str, Any], level: str
    ) -> list[Any]:
        if isinstance(context, ReasoningResponse):
            # It's a ReasoningResponse object
            return getattr(context, level, [])
        else:
            # It's a dict
            return context.get(level, [])

    for level in REASONING_LEVELS:
        original_observations = normalize_observations_for_comparison(
            get_observations(original_context, level)
        )
        revised_list = get_observations(revised_observations, level)

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

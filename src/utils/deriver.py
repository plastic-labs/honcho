"""
Utility functions for deriver/reasoning operations.

This module contains helper functions for processing observations, formatting context,
and handling temporal metadata for the reasoning system.
"""

from typing import Dict, List, Any
import logging
from datetime import datetime

from src.deriver.models import ReasoningResponse

REASONING_LEVELS = ["explicit", "deductive", "inductive", "abductive"]
LEVEL_LABELS = {
    "explicit": "Explicit (Literal facts directly stated by the user)",
    "deductive": "Deductive (Logically necessary conclusions from explicit facts)",
    "inductive": "Inductive (Highly probable generalizations from patterns)",
    "abductive": "Abductive (Best explanatory hypotheses for all observations)",
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
    
    premises_formatted = []
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


def extract_observation_content(observation) -> str:
    """Extract content string from an observation (dict or string)."""
    # Handle StructuredObservation objects (Pydantic models)
    if hasattr(observation, 'conclusion') and hasattr(observation, 'premises'):
        return observation.conclusion 
    
    if isinstance(observation, dict):
        # For structured observations, return only the conclusion
        if "conclusion" in observation:
            conclusion = observation["conclusion"]
            return conclusion
        # Fallback to content field or string representation
        return observation.get("content", str(observation))
    return str(observation)


def ensure_context_structure(context: dict) -> dict:
    """Ensure context has all reasoning levels with empty lists as defaults."""
    return {level: context.get(level, []) for level in REASONING_LEVELS}


def format_new_turn_with_timestamp(new_turn: str, current_time: str) -> str:
    """
    Format new turn message with optional timestamp.

    Args:
        new_turn: The message content
        current_time: Timestamp string or "unknown"

    Returns:
        Formatted string like "2023-05-08 13:56:00 user: hello" or "user: hello"
    """
    if current_time and current_time != "unknown":
        return f"{current_time} user: {new_turn}"
    else:
        return f"user: {new_turn}"


def format_context_for_prompt(context) -> str:
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

    formatted_sections = []

    # Handle both ReasoningResponse objects and dicts
    if hasattr(context, "explicit"):
        # It's a ReasoningResponse object
        observations_by_level = {
            "explicit": context.explicit,
            "deductive": context.deductive,
            "inductive": context.inductive,
            "abductive": context.abductive,
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

            # Check if observation is a dict with temporal metadata
            if isinstance(observation, dict) and "access_count" in observation:
                temporal_info = format_temporal_metadata(observation)
                formatted_sections.append(f"  • {observation_content}{temporal_info}")
            else:
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


def format_datetime_simple(dt) -> str:
    """
    Format datetime object to simple format matching new turn formatting.
    Converts from ISO format like '2025-06-02T19:43:41.392640+00:00'
    to simple format like '2025-06-03 20:23:43'

    Args:
        dt: datetime object or datetime string

    Returns:
        Formatted datetime string in simple format
    """
    if hasattr(dt, "strftime"):
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


def format_temporal_metadata(observation: dict) -> str:
    """
    Format temporal metadata for display in reasoning prompts.
    Shows both observation genesis (when derived) and ongoing relevance (access patterns).

    Args:
        observation: Observation dictionary containing temporal metadata

    Returns:
        Formatted temporal context string showing inception and access patterns
    """
    temporal_parts = []

    # Observation inception - when originally derived
    created_at = observation.get("created_at")
    if created_at:
        formatted_time = format_datetime_simple(created_at)
        temporal_parts.append(f"derived {formatted_time}")

    # Access frequency and session spread
    access_count = observation.get("access_count", 0)
    accessed_sessions = observation.get("accessed_sessions", [])

    if access_count > 0:
        if accessed_sessions:
            session_count = len(accessed_sessions)
            if session_count > 1:
                temporal_parts.append(
                    f"accessed {access_count}x across {session_count} sessions"
                )
            else:
                temporal_parts.append(f"accessed {access_count}x in 1 session")
        else:
            temporal_parts.append(f"accessed {access_count}x")

    # Last accessed timestamp
    last_accessed = observation.get("last_accessed")
    if last_accessed:
        formatted_last_accessed = format_datetime_simple(last_accessed)
        temporal_parts.append(f"last accessed {formatted_last_accessed}")

    if temporal_parts:
        return f" [{', '.join(temporal_parts)}]"
    else:
        return ""


def normalize_observations_for_comparison(observations: list) -> set:
    """Convert observations to normalized strings for comparison."""
    normalized = set()
    for observation in observations:
        observation_content = extract_observation_content(observation)
        normalized.add(observation_content.strip().lower())
    return normalized


def format_context_for_trace(
    context: ReasoningResponse, include_similarity_scores: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Format context for trace capture with optional similarity scores.

    Args:
        context: Context dictionary with reasoning levels
        include_similarity_scores: Whether to include similarity scores in the output

    Returns:
        Formatted context dictionary for trace
    """
    # formatted: Dict[str, List[Dict[str, Any]]] = {}
    # for level in REASONING_LEVELS:
    #     observations = context.get(level, [])
    #     formatted[level] = []
    #     for observation in observations:
    #         if isinstance(observation, dict):
    #             entry: Dict[str, Any] = {
    #                 "content": extract_observation_content(observation),
    #                 "document_id": observation.get("document_id", "unknown"),
    #             }
    #             if include_similarity_scores:
    #                 entry["similarity_score"] = observation.get("similarity_score", 0.0)
    #             formatted[level].append(entry)
    #         else:
    #             entry = {"content": str(observation), "document_id": "unknown"}
    #             if include_similarity_scores:
    #                 entry["similarity_score"] = 0.0
    #             formatted[level].append(entry)
    # return formatted
    return context.model_dump()


def analyze_observation_changes(
    original_context: ReasoningResponse,
    revised_observations: ReasoningResponse,
    significance_threshold: float,
    include_details: bool = False,
):
    """
    Analyze changes between original and revised observations.

    Args:
        original_context: Original observation context
        revised_observations: Revised observation context
        significance_threshold: Threshold for determining significance
        include_details: Whether to include detailed change information

    Returns:
        If include_details=False: bool (is_significant)
        If include_details=True: tuple(is_significant, changes_detected, significance_score)
    """
    logger = logging.getLogger(__name__)

    total_original_observations = 0
    total_changed_observations = 0
    changes_detected = {} if include_details else None

    # Helper function to get observations from either ReasoningResponse or dict
    def get_observations(context, level):
        if hasattr(context, "explicit"):
            # It's a ReasoningResponse object
            return getattr(context, level, [])
        else:
            # It's a dict
            return context.get(level, [])

    for level in REASONING_LEVELS:
        # Get normalized observation sets for comparison
        original_observations = normalize_observations_for_comparison(
            get_observations(original_context, level)
        )
        revised_observations_list = normalize_observations_for_comparison(
            get_observations(revised_observations, level)
        )

        level_original_count = len(original_observations)
        total_original_observations += level_original_count

        # Count significant changes (only additions for deriver)
        added_observations = revised_observations_list - original_observations
        level_changes = len(added_observations)
        total_changed_observations += level_changes

        # Log changes for debugging
        if level_changes > 0:
            logger.debug(f"Changes in {level} level:")
            logger.debug(f"  Original: {level_original_count} observations")
            logger.debug(f"  Added: {len(added_observations)} observations")
            if added_observations:
                logger.debug(
                    f"  New observations: {list(added_observations)[:3]}..."
                )  # Show first 3

        # Format changes for trace if detailed output requested
        if include_details and changes_detected is not None:
            changes_detected[level] = {
                "added": list(added_observations),
            }

    # Calculate change percentage
    if total_original_observations == 0:
        change_percentage = 1.0 if total_changed_observations > 0 else 0.0
    else:
        change_percentage = total_changed_observations / total_original_observations

    # Check if changes meet significance threshold
    is_significant = change_percentage >= significance_threshold

    logger.debug(
        f"Change analysis: {total_changed_observations}/{total_original_observations} observations changed ({change_percentage:.1%})"
    )
    logger.debug(
        f"Significance threshold: {significance_threshold:.1%} - {'MET' if is_significant else 'NOT MET'}"
    )

    if not is_significant and total_changed_observations > 0:
        logger.debug(
            "Changes detected but not significant enough to continue recursion"
        )

    if include_details:
        return is_significant, changes_detected, change_percentage
    else:
        return is_significant


def find_new_observations(
    original_context: ReasoningResponse, revised_observations: ReasoningResponse
) -> dict:
    """
    Find observations that are new in revised_observations compared to original_context.

    Args:
        original_context: Original observation context
        revised_observations: Revised observation context

    Returns:
        Dictionary with new observations by level
    """
    new_observations_by_level = {}

    # Helper function to get observations from either ReasoningResponse or dict
    def get_observations(context, level):
        if hasattr(context, "explicit"):
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
        new_observations = []
        for observation in revised_list:
            normalized_observation = (
                extract_observation_content(observation).strip().lower()
            )
            if normalized_observation not in original_observations:
                new_observations.append(observation)

        new_observations_by_level[level] = new_observations

    return new_observations_by_level

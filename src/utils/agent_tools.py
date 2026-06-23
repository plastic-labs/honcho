import asyncio
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.models import Document
from src.schemas import ResolvedConfiguration
from src.telemetry.events import (
    AgentToolConclusionsCreatedEvent,
    AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
    EmbeddingCallPurpose,
    emit,
)
from src.utils import summarizer
from src.utils.formatting import format_new_turn_with_timestamp, utc_now_iso
from src.utils.representation import Representation
from src.utils.types import ToolResult, embedding_call_purpose, get_current_iteration

logger = logging.getLogger(__name__)

# Hard cap to prevent unbounded peer card growth from repeated agent updates.
MAX_PEER_CARD_FACTS = 40

# Identity-marker prefixes allowed on the peer card. Anything else is rejected
# structurally — see `_validate_peer_card_entry`.
PEER_CARD_ALLOWED_PREFIXES: tuple[str, ...] = (
    "IDENTITY:",
    "ATTRIBUTE:",
    "RELATIONSHIP:",
    "INSTRUCTION:",
)

# Per-entry character cap to block evidence-bundle dumps and runaway lines.
MAX_PEER_CARD_ENTRY_LENGTH = 200


def _validate_peer_card_entry(line: str) -> bool:
    """Structural validation for a single peer card entry.

    Returns True when the line starts with one of the allowed prefixes followed
    by a space, has a non-empty body after the prefix, and fits within the per-
    entry length cap. Subject-substance correctness (is this actually about the
    observed peer?) is left to the prompt — this is form-only.
    """
    if not line or len(line) > MAX_PEER_CARD_ENTRY_LENGTH:
        return False
    for prefix in PEER_CARD_ALLOWED_PREFIXES:
        prefix_with_space = f"{prefix} "
        if line.startswith(prefix_with_space):
            body = line[len(prefix_with_space) :].strip()
            return bool(body)
    return False


def _normalized_observation_input(
    obs: schemas.ObservationInput,
) -> schemas.ObservationInput:
    """Return an observation input with content normalized for persistence/embedding."""
    return obs.model_copy(update={"content": obs.content.strip()})


def _base_observation_properties() -> dict[str, Any]:
    return {
        "content": {
            "type": "string",
            "description": "The observation content",
        },
        "level": {
            "type": "string",
            "enum": [
                "explicit",
                "deductive",
                "inductive",
                "contradiction",
            ],
            "description": (
                "Level: 'explicit' for direct facts, 'deductive' for logical "
                + "necessities, 'inductive' for patterns, 'contradiction' for "
                + "conflicting statements"
            ),
        },
        "source_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Document IDs of source or premise observations. Required and "
                + "must be non-empty for deductive, inductive, and contradiction "
                + "observations."
            ),
        },
        "premises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "(For deductive) Human-readable premise text for display",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "(For inductive/contradiction) Human-readable source text for display",
        },
        "pattern_type": {
            "type": "string",
            "enum": [
                "preference",
                "behavior",
                "personality",
                "tendency",
                "correlation",
            ],
            "description": "(For inductive only) Type of pattern being identified",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": (
                "(For inductive only) Confidence level: 'high' for 5+ sources, "
                + "'medium' for 3-4, 'low' for 2"
            ),
        },
    }


def _generic_observation_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": _base_observation_properties(),
        "required": ["content", "level"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"level": {"const": "deductive"}}},
                "then": {
                    "required": ["source_ids", "premises"],
                    "properties": {
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "premises": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                },
            },
            {
                "if": {"properties": {"level": {"const": "inductive"}}},
                "then": {
                    "required": [
                        "source_ids",
                        "sources",
                        "pattern_type",
                        "confidence",
                    ],
                    "properties": {
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                    },
                },
            },
            {
                "if": {"properties": {"level": {"const": "contradiction"}}},
                "then": {
                    "required": ["source_ids", "sources"],
                    "properties": {
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                    },
                },
            },
        ],
    }


def _deductive_observation_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The deductive conclusion as a self-contained statement",
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Required non-empty list of source observation IDs supporting the deduction",
            },
            "premises": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Required human-readable premise text matching the source observations",
            },
        },
        "required": ["content", "source_ids", "premises"],
        "additionalProperties": False,
    }


def _inductive_observation_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The inductive pattern or generalization as a self-contained statement",
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Required list of at least two source observation IDs supporting the pattern",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Required human-readable evidence text matching the source observations",
            },
            "pattern_type": {
                "type": "string",
                "enum": [
                    "preference",
                    "behavior",
                    "personality",
                    "tendency",
                    "correlation",
                ],
                "description": "Required pattern category",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Required confidence level based on evidence count",
            },
        },
        "required": ["content", "source_ids", "sources", "pattern_type", "confidence"],
        "additionalProperties": False,
    }


def _safe_int(value: Any, default: int) -> int:
    """Coerce a tool input value to int, returning default on failure.

    LLMs sometimes pass non-numeric strings (e.g. 'Infinity') for integer
    parameters which would crash ``min()`` comparisons.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


# Module-level lock registry for thread-safe observation creation.
# Keyed by (workspace_name, observer, observed) to ensure all tool executors
# operating on the same data share the same lock.
#
# Uses WeakValueDictionary so entries are automatically removed when no
# ToolContext holds a reference to the lock (i.e., all executors for that
# key have finished and been garbage collected). This prevents unbounded
# growth over the lifetime of a long-running deriver process.
_observation_locks: weakref.WeakValueDictionary[tuple[str, str, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)
_registry_lock = asyncio.Lock()


async def get_observation_lock(
    workspace_name: str, observer: str, observed: str
) -> asyncio.Lock:
    """
    Get or create a lock for a specific workspace/observer/observed combination.

    This ensures that concurrent tool executors operating on the same observation
    space share a lock, preventing race conditions during document creation.

    The lock is stored as a weak reference — it stays alive as long as at least
    one ToolContext (via create_tool_executor) holds a strong reference. Once all
    executors for a key finish and are garbage collected, the entry is
    automatically removed from the registry.

    Args:
        workspace_name: Workspace identifier
        observer: The observing peer
        observed: The peer being observed

    Returns:
        An asyncio.Lock shared by all executors for this combination
    """
    key = (workspace_name, observer, observed)
    async with _registry_lock:
        lock = _observation_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _observation_locks[key] = lock
        return lock


@dataclass
class ObservationFailure:
    """Records a single observation that failed during creation."""

    content_preview: str
    error: str


@dataclass
class ObservationsCreatedResult:
    """Result of a batch create_observations call."""

    created_count: int
    created_levels: list[str]
    failed: list[ObservationFailure]


def _truncate_tool_output(
    output: str, max_chars: int | None = None
) -> tuple[str, int, bool]:
    """Truncate tool output to prevent token explosion.

    Returns (text, original_chars, was_truncated). Callers thread the
    truncation signal into `ToolResult.metadata` so
    `AgentToolCallCompletedEvent` can report `was_truncated` and
    `result_chars_before_truncation` instead of always emitting them as
    None/False.
    """
    if max_chars is None:
        max_chars = settings.LLM.MAX_TOOL_OUTPUT_CHARS
    original_chars = len(output)
    if original_chars <= max_chars:
        return output, original_chars, False
    truncated = (
        output[:max_chars]
        + f"\n\n[OUTPUT TRUNCATED - showing {max_chars:,} of {original_chars:,} characters]"
    )
    return truncated, original_chars, True


def _maybe_truncated_result(output: str) -> "str | ToolResult":
    """Run `_truncate_tool_output` and wrap in `ToolResult` only when the
    output was actually clamped, so the truncation signal reaches the
    `AgentToolCallCompletedEvent` emitter (which reads `was_truncated` /
    `result_chars_before_truncation` from `ToolResult.metadata`). Returns a
    bare `str` in the common no-truncation case to keep the handler
    contract unchanged.
    """
    content, original_chars, was_truncated = _truncate_tool_output(output)
    if not was_truncated:
        return content
    return ToolResult(
        content=content,
        metadata={
            "was_truncated": True,
            "result_chars_before_truncation": original_chars,
        },
    )


def _truncate_message_content(content: str, max_chars: int | None = None) -> str:
    """Truncate individual message content (simple beginning truncation)."""
    if max_chars is None:
        max_chars = settings.LLM.MAX_MESSAGE_CONTENT_CHARS
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def _extract_pattern_snippet(
    content: str, pattern: str, max_chars: int | None = None
) -> str:
    """Extract snippet around a regex pattern match.

    For grep/exact text search, finds the pattern and extracts context around it.
    """
    import re

    if max_chars is None:
        max_chars = settings.LLM.MAX_MESSAGE_CONTENT_CHARS
    if len(content) <= max_chars:
        return content

    match = re.search(re.escape(pattern), content, re.IGNORECASE)
    if not match:
        # No match, return beginning
        return content[:max_chars] + "..."

    match_start = match.start()
    match_end = match.end()

    # Calculate window around match
    match_len = match_end - match_start
    remaining = max_chars - match_len
    before = remaining // 2
    after = remaining - before

    start = max(0, match_start - before)
    end = min(len(content), match_end + after)

    # Adjust if we hit boundaries
    if start == 0:
        end = min(len(content), max_chars)
    elif end == len(content):
        start = max(0, len(content) - max_chars)

    snippet = content[start:end]

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""

    return f"{prefix}{snippet}{suffix}"


TOOLS: dict[str, dict[str, Any]] = {
    "create_observations": {
        "name": "create_observations",
        "description": "Create observations at any level: explicit (facts), deductive (logical necessities), inductive (patterns), or contradiction (conflicting statements). For deductive, inductive, and contradiction observations, missing or empty source_ids are invalid and will be rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of observations to create",
                    "items": _generic_observation_item_schema(),
                },
            },
            "required": ["observations"],
        },
    },
    "create_observations_deductive": {
        "name": "create_observations_deductive",
        "description": "Create new deductive observations discovered while answering the query. Every observation must include non-empty source_ids and premise text. Use this only for novel deductions grounded in existing observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new deductive observations to create",
                    "items": _deductive_observation_item_schema(),
                },
            },
            "required": ["observations"],
        },
    },
    "create_observations_inductive": {
        "name": "create_observations_inductive",
        "description": "Create new inductive observations discovered while answering the query. Every observation must include source_ids, source text, pattern_type, and confidence. Use this only for patterns supported by multiple observations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new inductive observations to create",
                    "items": _inductive_observation_item_schema(),
                },
            },
            "required": ["observations"],
        },
    },
    "update_peer_card": {
        "name": "update_peer_card",
        "description": (
            "Update the peer card with stable identity markers about the observed peer. "
            "An identity marker distinguishes the peer from others of its kind and persists across interactions. "
            "The peer may be any entity with identity that changes over time (human, agent, codebase, team, organization) — do not assume the peer is human. "
            "Each entry must start with one of four prefixes: `IDENTITY:` (canonical name, kind, aliases, IDs), "
            "`ATTRIBUTE:` (stable durable property, including explicitly stated standing preferences), "
            "`RELATIONSHIP:` (durable link to another entity), or "
            "`INSTRUCTION:` (standing rule of engagement the peer has explicitly stated). "
            "Do not write `TRAIT:` or behavioral `PREFERENCE:` entries, one-off observations, transient state, "
            "inferred facts not directly supported by evidence, evidence bundles / `e.g.` clauses, or entries about co-occurring peers. "
            "Entries without an allowed prefix or that exceed the per-entry length cap are rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "array",
                    "description": (
                        "Complete deduplicated peer card list (max 40 entries). "
                        "Each entry must start with one of the allowed prefixes "
                        "(`IDENTITY: `, `ATTRIBUTE: `, `RELATIONSHIP: `, `INSTRUCTION: `) "
                        "followed by one concise identity marker. Entries without an allowed prefix are rejected."
                    ),
                    "items": {"type": "string"},
                },
            },
            "required": ["content"],
        },
    },
    "get_recent_history": {
        "name": "get_recent_history",
        "description": "Retrieve recent conversation history to get more context about the conversation.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    "search_memory": {
        "name": "search_memory",
        "description": "Search for observations in memory using semantic similarity. Use this to find relevant information about the peer when you need to recall specific details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "top_k": {
                    "type": "integer",
                    "description": "(Optional) number of results to return (default: 20, max: 40)",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    "get_observation_context": {
        "name": "get_observation_context",
        "description": "Retrieve messages for given message IDs along with surrounding context. Takes message IDs (from an observation's message_ids field) and retrieves those messages plus the messages immediately before and after each one to provide conversation context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of message IDs to retrieve (get these from observation.message_ids in search results)",
                },
            },
            "required": ["message_ids"],
        },
    },
    "search_messages": {
        "name": "search_messages",
        "description": "Search for messages using semantic similarity and retrieve conversation snippets. Returns matching messages with surrounding context (2 messages before and after). Nearby matches within the same session are merged into a single snippet to avoid repetition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to find relevant messages",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching messages to return (default: 10, max: 20)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    "grep_messages": {
        "name": "grep_messages",
        "description": "Search for messages containing specific text (case-insensitive). Unlike semantic search, this finds EXACT text matches. Use for finding specific names, dates, phrases, or keywords mentioned in conversations. Returns messages with surrounding context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to search for (case-insensitive substring match)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default: 10, max: 30)",
                    "default": 10,
                },
                "context_window": {
                    "type": "integer",
                    "description": "Number of messages before/after each match to include (default: 2)",
                    "default": 2,
                },
            },
            "required": ["text"],
        },
    },
    "get_messages_by_date_range": {
        "name": "get_messages_by_date_range",
        "description": "Get messages from a specific date range. Use this to find what was discussed during a particular time period, or to compare information before vs after a date. Essential for knowledge update questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "after_date": {
                    "type": "string",
                    "description": "Start date (ISO format, e.g., '2024-01-15'). Returns messages after this date.",
                },
                "before_date": {
                    "type": "string",
                    "description": "End date (ISO format). Returns messages before this date.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default: 20, max: 50)",
                    "default": 20,
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order: 'asc' for oldest first, 'desc' for newest first (default: desc)",
                    "default": "desc",
                },
            },
        },
    },
    "search_messages_temporal": {
        "name": "search_messages_temporal",
        "description": "Semantic search for messages with optional date filtering. Combines the power of semantic search with time constraints. Use after_date to find recent mentions of a topic, or before_date to find what was said about something before a certain point. Best for knowledge update questions where you need to find the MOST RECENT discussion of a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query",
                },
                "after_date": {
                    "type": "string",
                    "description": "Only return messages after this date (ISO format, e.g., '2024-01-15')",
                },
                "before_date": {
                    "type": "string",
                    "description": "Only return messages before this date (ISO format)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default: 10, max: 20)",
                    "default": 10,
                },
                "context_window": {
                    "type": "integer",
                    "description": "Messages before/after each match (default: 2)",
                    "default": 2,
                },
            },
            "required": ["query"],
        },
    },
    "get_recent_observations": {
        "name": "get_recent_observations",
        "description": "Get the most recent observations about the peer. Useful for understanding what's been learned recently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of observations to return (default: 10)",
                    "default": 10,
                },
                "session_only": {
                    "type": "boolean",
                    "description": "If true, only return observations from the current session (default: false)",
                    "default": False,
                },
            },
        },
    },
    "get_most_derived_observations": {
        "name": "get_most_derived_observations",
        "description": "Get observations that have been reinforced most frequently across conversations. These represent the most established facts about the peer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of observations to return (default: 10)",
                    "default": 10,
                },
            },
        },
    },
    "get_session_summary": {
        "name": "get_session_summary",
        "description": "Get the session summary (short or long form). Useful for understanding the overall conversation context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary_type": {
                    "type": "string",
                    "enum": ["short", "long"],
                    "description": "Type of summary to retrieve (default: short)",
                    "default": "short",
                },
            },
        },
    },
    "get_peer_card": {
        "name": "get_peer_card",
        "description": "Get the peer card containing known biographical information about the peer (name, age, location, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    "delete_observations": {
        "name": "delete_observations",
        "description": "Delete observations by their IDs. Use the exact ID shown in [id:xxx] format from search results. Example: if observation shows '[id:abc123XYZ]', pass 'abc123XYZ' to delete it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of observation IDs to delete (use the exact ID from [id:xxx] in search results)",
                },
            },
            "required": ["observation_ids"],
        },
    },
    "finish_consolidation": {
        "name": "finish_consolidation",
        "description": "Signal that consolidation is complete. Call this when you have finished your consolidation work and are ready to stop. You MUST call this tool when done - do not keep exploring indefinitely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished (peer card updates, observations consolidated, observations deleted)",
                },
            },
            "required": ["summary"],
        },
    },
    "extract_preferences": {
        "name": "extract_preferences",
        "description": "Extract user preferences and standing instructions from conversation history. This tool performs both semantic and text searches for preferences, instructions, and communication style preferences, then returns them for adding to the peer card. Call this FIRST during consolidation.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    "get_reasoning_chain": {
        "name": "get_reasoning_chain",
        "description": "Get the reasoning chain for an observation - traverse the tree to find premises (for deductive) or sources (for inductive), and/or find conclusions derived from this observation. Use this to understand how an observation was derived or what conclusions depend on it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_id": {
                    "type": "string",
                    "description": "The document ID of the observation to get the reasoning chain for",
                },
                "direction": {
                    "type": "string",
                    "enum": ["premises", "conclusions", "both"],
                    "description": "'premises' to get what this observation is based on, 'conclusions' to get what depends on it, 'both' for full context",
                    "default": "both",
                },
            },
            "required": ["observation_id"],
        },
    },
}

# Tools for the dialectic agent (analysis)
DIALECTIC_TOOLS: list[dict[str, Any]] = [
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    TOOLS["get_observation_context"],
    # TOOLS["create_observations_deductive"],
    TOOLS["grep_messages"],  # For exact text search (names, dates, keywords)
    TOOLS["get_messages_by_date_range"],  # For temporal/date-based queries
    TOOLS["search_messages_temporal"],  # Semantic search + date filtering
    TOOLS["get_reasoning_chain"],  # Traverse reasoning trees
]

# Minimal tools for dialectic agent at "minimal" reasoning level
# Reduces cost by limiting tool definitions in context
DIALECTIC_TOOLS_MINIMAL: list[dict[str, Any]] = [
    TOOLS["search_memory"],
    TOOLS["search_messages"],
]

# Tools for the dreamer agent (consolidation + peer card + deduplication)
DREAMER_TOOLS: list[dict[str, Any]] = [
    # Preference extraction (should be called first)
    TOOLS["extract_preferences"],
    TOOLS["get_recent_observations"],
    TOOLS["get_most_derived_observations"],
    TOOLS["search_memory"],
    TOOLS["get_peer_card"],
    TOOLS["create_observations"],
    TOOLS["delete_observations"],
    TOOLS["update_peer_card"],
    # Message access tools for context verification
    TOOLS["search_messages"],
    TOOLS["get_observation_context"],
    # Tree traversal
    TOOLS["get_reasoning_chain"],
    # Completion signal
    TOOLS["finish_consolidation"],
]

# Tools for the deduction specialist (dreamer phase 1)
# Creates deductive observations from explicit observations, can delete duplicates
# Includes message access for context and self-directed exploration
# Note: get_peer_card is not included - peer card is injected into the prompt directly
DEDUCTION_SPECIALIST_TOOLS: list[dict[str, Any]] = [
    # Discovery tools
    TOOLS["get_recent_observations"],
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    # Action tools
    TOOLS["create_observations_deductive"],
    TOOLS["delete_observations"],
    TOOLS["update_peer_card"],
]

# Tools for the induction specialist (dreamer phase 2)
# Creates inductive observations from explicit and deductive observations
# Includes message access for context and self-directed exploration
# Induction does not write to the peer card — that is deduction's responsibility.
INDUCTION_SPECIALIST_TOOLS: list[dict[str, Any]] = [
    # Discovery tools
    TOOLS["get_recent_observations"],
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    # Action tools
    TOOLS["create_observations_inductive"],
]


async def create_observations(
    observations: list[schemas.ObservationInput],
    observer: str,
    observed: str,
    session_name: str | None,
    workspace_name: str,
    message_ids: list[int],
    message_created_at: str,
    run_id: str | None = None,
    parent_category: str | None = None,
) -> ObservationsCreatedResult:
    """
    Create multiple observations (documents) in the memory system in a single call.

    Uses short-lived DB sessions to avoid holding connections during embedding API calls.

    Args:
        observations: List of validated observation inputs
        observer: The peer making the observation
        observed: The peer being observed
        session_name: Session identifier
        workspace_name: Workspace identifier
        message_ids: List of message IDs these observations are based on
        message_created_at: Timestamp of the message that triggered these observations
        run_id: Agent run id, threaded onto the embedding-call ContextVar so
            EmbeddingCallCompletedEvents emitted here can be joined back to
            the originating agent run.

    Returns:
        ObservationsCreatedResult with created count and any per-observation failures
    """
    if not observations:
        logger.warning("create_observations called with empty list")
        return ObservationsCreatedResult(created_count=0, created_levels=[], failed=[])

    normalized_observations = [
        _normalized_observation_input(obs)
        for obs in observations
        if obs.content.strip()
    ]
    if not normalized_observations:
        logger.info("No non-empty observations to create")
        return ObservationsCreatedResult(created_count=0, created_levels=[], failed=[])

    # Ensure collection exists (short DB scope)
    async with tracked_db("create_observations.collection") as db:
        await crud.get_or_create_collection(
            db,
            workspace_name,
            observer=observer,
            observed=observed,
        )

    # Compute embeddings (no DB needed)
    contents = [obs.content for obs in normalized_observations]
    embeddings_by_index: dict[int, list[float]] | None = None
    try:
        with embedding_call_purpose(
            EmbeddingCallPurpose.CREATE_OBSERVATIONS.value,
            workspace_name=workspace_name,
            run_id=run_id,
            parent_category=parent_category,
        ):
            embeddings = await embedding_client.simple_batch_embed(contents)
        embeddings_by_index = dict(
            zip(range(len(normalized_observations)), embeddings, strict=True)
        )
    except Exception as e:
        logger.warning(
            "Batch embedding failed for create_observations; falling back to per-observation embedding: %s",
            e,
        )

    # Build document objects with pre-computed embeddings
    documents: list[schemas.DocumentCreate] = []
    failed: list[ObservationFailure] = []
    for i, obs in enumerate(normalized_observations):
        embedding: list[float]
        if embeddings_by_index is not None:
            embedding = embeddings_by_index[i]
        else:
            try:
                with embedding_call_purpose(
                    EmbeddingCallPurpose.CREATE_OBSERVATIONS.value,
                    workspace_name=workspace_name,
                    run_id=run_id,
                    parent_category=parent_category,
                ):
                    embedding = await embedding_client.embed(obs.content)
            except Exception as e:
                logger.warning(
                    "Error embedding observation content for level '%s': %s",
                    obs.level,
                    e,
                )
                failed.append(
                    ObservationFailure(
                        content_preview=obs.content[:50],
                        error=f"Embedding failed: {e}",
                    )
                )
                continue

        # Build metadata with level-specific fields
        metadata = schemas.DocumentMetadata(
            message_ids=message_ids,
            message_created_at=message_created_at,
            source_ids=obs.source_ids
            if obs.level in ("deductive", "inductive", "contradiction")
            else None,
            premises=obs.premises if obs.level == "deductive" else None,
            sources=obs.sources
            if obs.level in ("inductive", "contradiction")
            else None,
            pattern_type=obs.pattern_type if obs.level == "inductive" else None,
            confidence=(obs.confidence or "medium")
            if obs.level == "inductive"
            else None,
        )

        doc = schemas.DocumentCreate(
            content=obs.content,
            session_name=session_name,
            level=obs.level,
            metadata=metadata,
            embedding=embedding,
            source_ids=obs.source_ids
            if obs.level in ("deductive", "inductive", "contradiction")
            else None,
        )
        documents.append(doc)

    # Bulk create all documents (short DB scope)
    accepted: list[schemas.DocumentCreate] = []
    if documents:
        async with tracked_db("create_observations.save") as db:
            accepted = await crud.create_documents(
                db,
                documents=documents,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                deduplicate=True,
            )
        logger.info(
            "Created %d observations in %s/%s/%s",
            len(accepted),
            workspace_name,
            observer,
            observed,
        )

    return ObservationsCreatedResult(
        created_count=len(accepted),
        created_levels=[doc.level for doc in accepted],
        failed=failed,
    )


async def get_recent_history(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    observed: str | None = None,
    token_limit: int = 8192,
) -> list[models.Message]:
    """
    Retrieve recent conversation history.

    If session_name is provided, retrieves messages from that session.
    If session_name is None but observed is provided, retrieves recent messages
    sent by the observed peer across all their sessions.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        observed: Peer name to filter by when no session specified (optional)
        token_limit: Maximum tokens to retrieve (default: 8192)

    Returns:
        List of messages in chronological order
    """
    if session_name:
        # Get messages from a specific session
        messages_stmt = await crud.get_messages(
            workspace_name=workspace_name,
            session_name=session_name,
            token_limit=token_limit,
            reverse=True,  # Get most recent first
        )
        result = await db.execute(messages_stmt)
        messages = result.scalars().all()
        # Return in chronological order
        return list(reversed(messages))
    elif observed:
        # Get recent messages from the observed peer across all sessions
        stmt = (
            select(models.Message)
            .where(models.Message.workspace_name == workspace_name)
            .where(models.Message.peer_name == observed)
            .order_by(models.Message.created_at.desc())
            .limit(50)  # Limit to recent messages
        )
        result = await db.execute(stmt)
        messages = list(result.scalars().all())
        # Return in chronological order
        return list(reversed(messages))
    else:
        # No session and no observed peer - can't retrieve history
        return []


async def search_memory(
    workspace_name: str,
    observer: str,
    observed: str,
    query: str,
    limit: int,
    levels: list[str] | None = None,
    embedding: list[float] | None = None,
) -> Representation:
    """
    Search for observations in memory using semantic similarity.

    Does not require a DB session — ``query_documents`` manages its own
    short-lived sessions so no connection is held during external calls.

    Args:
        workspace_name: Workspace identifier
        observer: The peer who made the observations
        observed: The peer who was observed
        query: Search query text
        limit: Maximum number of results
        levels: Optional list of observation levels to filter by
                (e.g., ["explicit"], ["deductive", "inductive", "contradiction"])
        embedding: Optional pre-computed embedding to avoid redundant API calls

    Returns:
        Representation object containing relevant observations
    """
    # Build filter for levels if specified
    filters: dict[str, Any] | None = None
    if levels:
        filters = {"level": {"in": levels}}

    documents = await crud.query_documents(
        db=None,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        query=query,
        top_k=limit,
        filters=filters,
        embedding=embedding,
    )

    return Representation.from_documents(documents)


async def get_observation_context(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    message_ids: list[str],
    observer: str | None = None,
) -> list[models.Message]:
    """
    Retrieve messages for given message IDs along with surrounding context.

    Takes message IDs (from an observation's message_ids field) and retrieves those
    messages plus the messages immediately before and after each one to provide
    conversation context.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        message_ids: List of message IDs to retrieve
        observer: When provided and session_name is None, scope results
            to sessions this peer belongs to

    Returns:
        List of messages in chronological order, including the requested messages and surrounding context
    """
    if not message_ids:
        return []

    # Pre-fetch peer session scope if needed
    allowed_session_names: list[str] | None = None
    if observer and not session_name:
        from src.crud.message import get_peer_session_names

        allowed_session_names = await get_peer_session_names(
            db, workspace_name, observer
        )
        if not allowed_session_names:
            return []

    # Use a CTE to get seq_in_session values for target messages
    stmt = (
        select(models.Message.seq_in_session)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.public_id.in_(message_ids))
    )

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)
    elif allowed_session_names is not None:
        stmt = stmt.where(models.Message.session_name.in_(allowed_session_names))

    target_seqs_cte = stmt.cte("target_seqs")

    # Query messages where seq_in_session is within ±1 of any target sequence
    # We use EXISTS with arithmetic to check if the message is adjacent to any target
    stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(
            select(target_seqs_cte.c.seq_in_session)
            .where(
                (
                    target_seqs_cte.c.seq_in_session - models.Message.seq_in_session
                ).between(-1, 1)
            )
            .exists()
        )
        .order_by(models.Message.seq_in_session.asc())
    )

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)
    elif allowed_session_names is not None:
        stmt = stmt.where(models.Message.session_name.in_(allowed_session_names))

    result = await db.execute(stmt)
    messages = list(result.scalars().all())

    return messages


async def extract_preferences(
    workspace_name: str,
    session_name: str | None,
    observed: str,
    observer: str | None = None,
) -> dict[str, list[str]]:
    """
    Extract user preferences and standing instructions from conversation history.

    Uses semantic search to find messages that might contain preferences or instructions.
    This is language-agnostic and doesn't rely on keyword matching.

    Args:
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        observed: The peer whose preferences to extract
        observer: When provided and session_name is None, scope results
            to sessions this peer belongs to

    Returns:
        Dict with 'messages' list containing potentially relevant messages
    """
    messages: list[str] = []
    seen_content: set[str] = set()  # Dedupe by content hash

    # Semantic queries to find preference-like content
    semantic_queries = [
        "user preferences and communication style",
        "standing instructions and rules to follow",
        "how user wants responses formatted",
        "user requirements and constraints",
        "things user wants or does not want",
    ]

    # Batch embed all queries in a single API call (no DB needed).
    # If batching fails, each search call will generate its own embedding.
    query_embeddings_by_query: dict[str, list[float]] | None = None
    try:
        query_embeddings = await embedding_client.simple_batch_embed(semantic_queries)
        query_embeddings_by_query = dict(
            zip(semantic_queries, query_embeddings, strict=True)
        )
    except Exception as e:
        logger.warning(
            "Batch embedding failed for extract_preferences; falling back to per-query embedding in search_messages: %s",
            e,
        )

    for query in semantic_queries:
        try:
            snippets = await crud.search_messages(
                workspace_name=workspace_name,
                session_name=session_name,
                query=query,
                limit=10,
                context_window=0,
                embedding=(
                    query_embeddings_by_query.get(query)
                    if query_embeddings_by_query is not None
                    else None
                ),
                observer=observer,
            )
            for matches, _ in snippets:
                for msg in matches:
                    if msg.peer_name == observed:
                        content_key = msg.content[:100].lower()
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            messages.append(f"'{msg.content.strip()}'")
        except Exception as e:
            logger.warning("Error in semantic search for '%s': %s", query, e)

    return {
        "instructions": [],  # Deprecated - LLM will categorize
        "preferences": [],  # Deprecated - LLM will categorize
        "messages": messages[:30],  # Raw messages for LLM to process
    }


@dataclass
class ToolContext:
    """Context object passed to tool handlers."""

    workspace_name: str
    observer: str
    observed: str
    session_name: str | None
    current_messages: list[models.Message] | None
    include_observation_ids: bool
    history_token_limit: int
    # Shared lock for serializing writes to the same workspace/observer/observed.
    # This lock is obtained from the module-level registry to ensure all concurrent
    # tool executors for the same data share the same lock.
    db_lock: asyncio.Lock
    # Optional resolved configuration for checking feature flags
    configuration: ResolvedConfiguration | None = None
    # Telemetry context fields
    run_id: str | None = None
    agent_type: str | None = None  # "dialectic", "deriver", "dreamer"
    parent_category: str | None = None  # Parent category for CloudEvents


async def _handle_create_observations_impl(
    ctx: ToolContext,
    tool_input: dict[str, Any],
    *,
    forced_level: str | None = None,
) -> "str | ToolResult":
    """Handle create_observations tool."""
    raw_observations = tool_input.get("observations", [])

    if not raw_observations:
        return "ERROR: observations list is empty"

    # Set context-specific default level before Pydantic validation
    default_level = "explicit" if ctx.current_messages else "deductive"
    for obs in raw_observations:
        if forced_level is not None:
            obs["level"] = forced_level
        else:
            obs.setdefault("level", default_level)

    # Validate observations individually so valid ones are still processed
    observations: list[schemas.ObservationInput] = []
    validation_failures: list[ObservationFailure] = []
    for obs in raw_observations:
        try:
            validated = schemas.ObservationInput.model_validate(obs)
        except ValidationError as e:
            validation_failures.append(
                ObservationFailure(
                    content_preview=str(obs.get("content", ""))[:50],
                    error=f"Validation failed: {e}",
                )
            )
            continue
        # Deriver can only create explicit observations
        if ctx.current_messages and validated.level != "explicit":
            validation_failures.append(
                ObservationFailure(
                    content_preview=validated.content[:50],
                    error=f"Deriver can only create 'explicit' observations, got '{validated.level}'",
                )
            )
            continue
        observations.append(validated)

    if not observations:
        failure_details = "; ".join(
            f"'{f.content_preview}': {f.error}" for f in validation_failures
        )
        return f"ERROR: All observations failed validation: {failure_details}"

    # Determine message context
    if ctx.current_messages:
        message_ids = [msg.id for msg in ctx.current_messages]
        message_created_at = str(ctx.current_messages[-1].created_at)
    else:
        message_ids = []
        message_created_at = utc_now_iso()

    # Use lock to serialize database writes (prevents concurrent commit issues)
    async with ctx.db_lock:
        result = await create_observations(
            observations=observations,
            observer=ctx.observer,
            observed=ctx.observed,
            session_name=ctx.session_name,
            workspace_name=ctx.workspace_name,
            message_ids=message_ids,
            message_created_at=message_created_at,
            run_id=ctx.run_id,
            parent_category=ctx.parent_category,
        )

    # Merge validation and embedding failures
    all_failures = validation_failures + result.failed

    # Count levels from actually-created observations
    levels = result.created_levels
    explicit_count = levels.count("explicit")
    deductive_count = levels.count("deductive")
    inductive_count = levels.count("inductive")
    contradiction_count = levels.count("contradiction")

    # Emit telemetry event if context is available
    if ctx.run_id and ctx.agent_type and ctx.parent_category:
        emit(
            AgentToolConclusionsCreatedEvent(
                run_id=ctx.run_id,
                iteration=get_current_iteration(),
                parent_category=ctx.parent_category,
                agent_type=ctx.agent_type,
                workspace_name=ctx.workspace_name,
                observer=ctx.observer,
                observed=ctx.observed,
                conclusion_count=result.created_count,
                levels=levels,
            )
        )

    response = (
        f"Created {result.created_count} observations for {ctx.observed} by {ctx.observer} "
        f"({explicit_count} explicit, {deductive_count} deductive, "
        f"{inductive_count} inductive, {contradiction_count} contradiction)"
    )

    if all_failures:
        failure_details = "; ".join(
            f"'{f.content_preview}': {f.error}" for f in all_failures
        )
        response += f"\nFailed {len(all_failures)}: {failure_details}"

    # +5: surface created_count so DreamSpecialistEvent can sum actual
    # observations across the run rather than just counting create_observations
    # calls (which would conflate "1 call that made 5 observations" with
    # "5 calls that each made 1").
    from src.utils.types import ToolResult

    return ToolResult(
        content=response,
        metadata={"created_count": result.created_count, "levels": levels},
    )


async def _handle_create_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    return await _handle_create_observations_impl(ctx, tool_input)


async def _handle_create_observations_deductive(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    return await _handle_create_observations_impl(
        ctx,
        tool_input,
        forced_level="deductive",
    )


async def _handle_create_observations_inductive(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    return await _handle_create_observations_impl(
        ctx,
        tool_input,
        forced_level="inductive",
    )


async def _handle_update_peer_card(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle update_peer_card tool."""
    # Check if peer card creation is disabled via configuration
    if ctx.configuration is not None and not ctx.configuration.peer_card.create:
        logger.info(
            "Peer card creation disabled for %s, skipping update",
            ctx.workspace_name,
        )
        return (
            "Peer card creation is disabled for this workspace/session configuration."
        )

    raw_peer_card_content = tool_input.get("content")

    # Guard against None or empty content — keep the existing peer card.
    if raw_peer_card_content is None:
        logger.warning(
            "Peer card update called with None content for %s, keeping existing card",
            ctx.workspace_name,
        )
        return "Peer card content was empty, no update performed."

    # Normalize, validate structure, and deduplicate to keep peer cards bounded
    # and on-spec.
    normalized_peer_card: list[str] = []
    seen: set[str] = set()
    rejected_count = 0
    # Keep a small sample of rejected entries to surface back to the model so it
    # can self-correct on a retry. Capped to avoid bloating the tool response.
    rejected_samples: list[str] = []
    _REJECTED_SAMPLE_CAP = 3
    _REJECTED_SAMPLE_LINE_LIMIT = 120
    items = (
        cast(list[str], raw_peer_card_content)
        if isinstance(raw_peer_card_content, list)
        else [str(raw_peer_card_content)]
    )
    for item in items:
        line = str(item).strip()
        if not line:
            continue

        if not _validate_peer_card_entry(line):
            rejected_count += 1
            if len(rejected_samples) < _REJECTED_SAMPLE_CAP:
                rejected_samples.append(line[:_REJECTED_SAMPLE_LINE_LIMIT])
            logger.info(
                "Rejecting peer card entry (no allowed prefix, empty body, or over length cap): %r",
                line[:80],
            )
            continue

        # Case-insensitive dedupe with whitespace normalization.
        normalized_key = " ".join(line.lower().split())
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        normalized_peer_card.append(line)

    if rejected_count:
        logger.info(
            "Peer card update for %s/%s/%s rejected %d structurally invalid entries",
            ctx.workspace_name,
            ctx.observer,
            ctx.observed,
            rejected_count,
        )

    def _format_rejection_feedback(scope: str) -> str:
        """Build a self-correction hint for the model. `scope` is grammar glue:
        either "all" (every entry rejected) or e.g. "3 of 12" (partial)."""
        samples_block = ""
        if rejected_samples:
            sample_lines = "\n".join(f"  - {s!r}" for s in rejected_samples)
            extra = (
                f" (+{rejected_count - len(rejected_samples)} more)"
                if rejected_count > len(rejected_samples)
                else ""
            )
            samples_block = f" Examples of rejected entries{extra}:\n{sample_lines}"
        return (
            f"Rejected {scope} entries for failing structural validation. "
            "Each entry must start with one of `IDENTITY: `, `ATTRIBUTE: `, "
            "`RELATIONSHIP: `, or `INSTRUCTION: ` and stay under the per-entry "
            f"length cap.{samples_block}"
        )

    # Don't clear the peer card if all content normalized to empty or every
    # entry was structurally invalid.
    if not normalized_peer_card:
        logger.warning(
            "Peer card update normalized to empty for %s (rejected=%d), keeping existing card",
            ctx.workspace_name,
            rejected_count,
        )
        if rejected_count:
            return _format_rejection_feedback(f"all {rejected_count}")
        return "Peer card content was empty after normalization, no update performed."

    if len(normalized_peer_card) > MAX_PEER_CARD_FACTS:
        logger.warning(
            "Peer card update exceeded max facts (%s), truncating from %s to %s",
            MAX_PEER_CARD_FACTS,
            len(normalized_peer_card),
            MAX_PEER_CARD_FACTS,
        )
        normalized_peer_card = normalized_peer_card[:MAX_PEER_CARD_FACTS]

    async with ctx.db_lock, tracked_db("tool.update_peer_card") as db:
        await crud.set_peer_card(
            db,
            workspace_name=ctx.workspace_name,
            peer_card=normalized_peer_card,
            observer=ctx.observer,
            observed=ctx.observed,
        )
    logger.info(
        f"Updated peer card for {ctx.workspace_name}/{ctx.observer}/{ctx.observed}"
    )

    # Emit telemetry event if context is available
    if ctx.run_id and ctx.agent_type and ctx.parent_category:
        # Count facts in peer card (content is a list of strings per tool schema)
        emit(
            AgentToolPeerCardUpdatedEvent(
                run_id=ctx.run_id,
                iteration=get_current_iteration(),
                parent_category=ctx.parent_category,
                agent_type=ctx.agent_type,
                workspace_name=ctx.workspace_name,
                observer=ctx.observer,
                observed=ctx.observed,
                facts_count=len(normalized_peer_card),
            )
        )

    # signal a successful peer_card update so DreamSpecialistEvent
    # can set its `peer_card_updated` flag without name-counting.
    from src.utils.types import ToolResult

    success_content = (
        f"Updated peer card for {ctx.observed} by {ctx.observer} "
        f"with {len(normalized_peer_card)} entries."
    )
    if rejected_count:
        # Partial reject: surface the rejection so the model can re-emit the
        # dropped entries (with correct prefixes) on a retry instead of
        # silently losing them.
        accepted = len(normalized_peer_card)
        total = accepted + rejected_count
        success_content = f"{success_content} {_format_rejection_feedback(f'{rejected_count} of {total}')}"
    return ToolResult(
        content=success_content,
        metadata={
            "peer_card_updated": True,
            "facts_count": len(normalized_peer_card),
            "rejected_count": rejected_count,
        },
    )


async def _handle_get_recent_history(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle get_recent_history tool."""
    _ = tool_input
    async with tracked_db("tool.get_recent_history", read_only=True) as db:
        history: list[models.Message] = await get_recent_history(
            db,
            workspace_name=ctx.workspace_name,
            session_name=ctx.session_name,
            observed=ctx.observed,
            token_limit=ctx.history_token_limit,
        )
        if not history:
            return "No conversation history available"
        history_text = "\n".join(
            [f"{m.peer_name}: {_truncate_message_content(m.content)}" for m in history]
        )
    scope = (
        f"from session {ctx.session_name}"
        if ctx.session_name
        else f"from {ctx.observed} across sessions"
    )
    output = f"Conversation history ({len(history)} messages {scope}):\n{history_text}"
    return _maybe_truncated_result(output)


async def _handle_search_memory(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle search_memory tool."""
    from src.utils.types import ToolResult

    top_k = min(_safe_int(tool_input.get("top_k"), 20), 40)
    query = tool_input["query"]
    try:
        with embedding_call_purpose(
            EmbeddingCallPurpose.SEARCH_MEMORY.value,
            workspace_name=ctx.workspace_name,
            run_id=ctx.run_id,
            parent_category=ctx.parent_category,
        ):
            query_embedding = await embedding_client.embed(query)
    except ValueError:
        return (
            "ERROR: Query exceeds maximum token limit of "
            + f"{settings.EMBEDDING.MAX_INPUT_TOKENS}. Please use a shorter query."
        )

    # Base telemetry metadata; results_count gets filled in below.
    search_meta: dict[str, Any] = {
        "top_k": top_k,
        "used_embedding": True,
        "embedding_query_count": 1,
        "query_tokens": _estimate_tokens_safe(query),
    }

    documents = await crud.query_documents(
        db=None,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        query=query,
        top_k=top_k,
        embedding=query_embedding,
    )
    mem = Representation.from_documents(documents)
    total_count = mem.len()
    if total_count == 0:
        # Empty-memory fallback: if the memory is *empty*, that means we're
        # quite early in a workspace/peer/session -- in order to give good
        # answers in this stage, and be efficient with tool calls, and make
        # sure the model doesn't short-circuit and think there's nothing
        # here, we automatically search the message history for relevant
        # information.
        zero_hit_meta = {**search_meta, "results_count": 0}
        if ctx.agent_type == "dialectic":
            limit = min(_safe_int(tool_input.get("top_k"), 20), 20)
            message_output = None
            snippets = await crud.search_messages(
                workspace_name=ctx.workspace_name,
                session_name=ctx.session_name,
                query=query,
                limit=limit,
                context_window=0,
                embedding=query_embedding,
                observer=ctx.observer,
            )
            if snippets:
                message_output = _format_message_snippets(
                    snippets, f"for query '{query}'"
                )
            if message_output:
                fallback_meta = {**zero_hit_meta, "results_count": len(snippets)}
                return ToolResult(
                    content=f"No observations yet. Message search results:\n\n{message_output}",
                    metadata=fallback_meta,
                )
            return ToolResult(
                content=(
                    f"No observations found for query '{query}', and no messages found in "
                    "history. Try a different phrasing or use grep_messages for exact text."
                ),
                metadata=zero_hit_meta,
            )
        return ToolResult(
            content=f"No observations found for query '{query}'",
            metadata=zero_hit_meta,
        )
    mem_str = mem.str_with_ids() if ctx.include_observation_ids else str(mem)
    search_meta["results_count"] = total_count
    return ToolResult(
        content=f"Found {total_count} observations for query '{query}':\n\n{mem_str}",
        metadata=search_meta,
    )


async def _handle_get_observation_context(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle get_observation_context tool."""
    async with tracked_db("tool.get_observation_context", read_only=True) as db:
        messages = await get_observation_context(
            db,
            workspace_name=ctx.workspace_name,
            session_name=ctx.session_name,
            message_ids=tool_input["message_ids"],
            observer=ctx.observer,
        )
        if not messages:
            return f"No messages found for IDs {tool_input['message_ids']}"
        messages_text = "\n".join(
            [
                format_new_turn_with_timestamp(
                    _truncate_message_content(m.content),
                    m.created_at,
                    m.peer_name,
                )
                for m in messages
            ]
        )
    output = f"Retrieved {len(messages)} messages with context:\n{messages_text}"
    return _maybe_truncated_result(output)


async def _handle_search_messages(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle search_messages tool."""
    from src.utils.types import ToolResult

    query = tool_input["query"]
    limit = min(_safe_int(tool_input.get("limit"), 10), 20)  # Cap at 20
    # Pre-compute embedding outside DB session to avoid holding a connection
    # during the external API call (same pattern as _handle_search_memory).
    with embedding_call_purpose(
        EmbeddingCallPurpose.SEARCH_MESSAGES.value,
        workspace_name=ctx.workspace_name,
        run_id=ctx.run_id,
        parent_category=ctx.parent_category,
    ):
        query_embedding = await embedding_client.embed(query)
    snippets = await crud.search_messages(
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        query=query,
        limit=limit,
        context_window=2,
        embedding=query_embedding,
        observer=ctx.observer,
    )
    search_meta: dict[str, Any] = {
        "top_k": limit,
        "used_embedding": True,
        "embedding_query_count": 1,
        "query_tokens": _estimate_tokens_safe(query),
        "results_count": len(snippets),
    }
    if not snippets:
        return ToolResult(
            content=f"No messages found for query '{query}'",
            metadata=search_meta,
        )
    formatted = _format_message_snippets(snippets, f"for query '{query}'")
    return ToolResult(content=formatted, metadata=search_meta)


async def _handle_grep_messages(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle grep_messages tool."""
    text = tool_input.get("text", "")
    if not text:
        return "ERROR: 'text' parameter is required"
    limit = min(_safe_int(tool_input.get("limit"), 10), 30)  # Cap at 30
    context_window = min(
        _safe_int(tool_input.get("context_window"), 2), 2
    )  # Cap context

    snippets = await crud.grep_messages(
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        text=text,
        limit=limit,
        context_window=context_window,
        observer=ctx.observer,
    )
    if not snippets:
        return f"No messages found containing '{text}'"

    # Format with pattern-based snippet extraction
    snippet_texts: list[str] = []
    total_matches = sum(len(matches) for matches, _ in snippets)
    for i, (matches, context) in enumerate(snippets, 1):
        lines: list[str] = []
        for msg in context:
            truncated = _extract_pattern_snippet(msg.content, text)
            lines.append(
                format_new_turn_with_timestamp(truncated, msg.created_at, msg.peer_name)
            )
        sess = context[0].session_name if context else "unknown"
        snippet_texts.append(
            f"--- Snippet {i} (session: {sess}, {len(matches)} match(es)) ---\n"
            + "\n".join(lines)
        )

    output = (
        f"Found {total_matches} messages containing '{text}' in {len(snippets)} conversation snippets:\n\n"
        + "\n\n".join(snippet_texts)
    )
    return _maybe_truncated_result(output)


def _parse_date(date_str: str | None, param_name: str) -> datetime | None | str:
    """Parse a date string, returning datetime, None, or error string."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return f"ERROR: Invalid {param_name} format '{date_str}'. Use ISO format (e.g., '2024-01-15')"


async def _handle_get_messages_by_date_range(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle get_messages_by_date_range tool."""
    after_date_str = tool_input.get("after_date")
    before_date_str = tool_input.get("before_date")
    limit = min(_safe_int(tool_input.get("limit"), 20), 20)
    order = tool_input.get("order", "desc")

    after_date = _parse_date(after_date_str, "after_date")
    if isinstance(after_date, str):
        return after_date  # Error message

    before_date = _parse_date(before_date_str, "before_date")
    if isinstance(before_date, str):
        return before_date  # Error message

    async with tracked_db("tool.get_messages_by_date_range", read_only=True) as db:
        messages = await crud.get_messages_by_date_range(
            db,
            workspace_name=ctx.workspace_name,
            session_name=ctx.session_name,
            after_date=after_date,
            before_date=before_date,
            limit=limit,
            order=order,
            observer=ctx.observer,
        )
        msg_count = len(messages)
        messages_text = (
            "\n".join(
                [
                    format_new_turn_with_timestamp(
                        _truncate_message_content(m.content), m.created_at, m.peer_name
                    )
                    for m in messages
                ]
            )
            if messages
            else ""
        )

    date_range: list[str] = []
    if after_date_str:
        date_range.append(f"after {after_date_str}")
    if before_date_str:
        date_range.append(f"before {before_date_str}")

    if not msg_count:
        range_desc = " and ".join(date_range) if date_range else "specified range"
        return f"No messages found {range_desc}"

    range_desc = " and ".join(date_range) if date_range else "all time"
    order_desc = "oldest first" if order == "asc" else "newest first"

    output = (
        f"Found {msg_count} messages ({range_desc}, {order_desc}):\n\n{messages_text}"
    )
    return _maybe_truncated_result(output)


async def _handle_search_messages_temporal(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle search_messages_temporal tool."""
    query = tool_input.get("query", "")
    if not query:
        return "ERROR: 'query' parameter is required"

    after_date_str = tool_input.get("after_date")
    before_date_str = tool_input.get("before_date")
    limit = min(_safe_int(tool_input.get("limit"), 10), 10)
    context_window = min(_safe_int(tool_input.get("context_window"), 2), 2)

    after_date = _parse_date(after_date_str, "after_date")
    if isinstance(after_date, str):
        return after_date

    before_date = _parse_date(before_date_str, "before_date")
    if isinstance(before_date, str):
        return before_date

    # Pre-compute embedding outside DB session to avoid holding a connection
    # during the external API call.
    with embedding_call_purpose(
        EmbeddingCallPurpose.SEARCH_MESSAGES.value,
        workspace_name=ctx.workspace_name,
        run_id=ctx.run_id,
        parent_category=ctx.parent_category,
    ):
        query_embedding = await embedding_client.embed(query)
    snippets = await crud.search_messages_temporal(
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        query=query,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        context_window=context_window,
        embedding=query_embedding,
        observer=ctx.observer,
    )
    date_filter: list[str] = []
    if after_date_str:
        date_filter.append(f"after {after_date_str}")
    if before_date_str:
        date_filter.append(f"before {before_date_str}")
    filter_desc = f" ({' and '.join(date_filter)})" if date_filter else ""

    # Matches the search_messages metadata shape so analytics can filter
    # AgentToolCallCompletedEvent uniformly across all embedding-backed
    # search tools (search_memory / search_messages / search_messages_temporal).
    search_meta: dict[str, Any] = {
        "top_k": limit,
        "used_embedding": True,
        "embedding_query_count": 1,
        "query_tokens": _estimate_tokens_safe(query),
        "results_count": len(snippets),
    }

    if not snippets:
        return ToolResult(
            content=f"No messages found for query '{query}'{filter_desc}",
            metadata=search_meta,
        )

    formatted = _format_message_snippets(snippets, f"for query '{query}'{filter_desc}")
    return ToolResult(content=formatted, metadata=search_meta)


async def _handle_get_recent_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_recent_observations tool."""
    session_only = tool_input.get("session_only", False)
    async with tracked_db("tool.get_recent_observations", read_only=True) as db:
        documents = await crud.query_documents_recent(
            db=db,
            workspace_name=ctx.workspace_name,
            observer=ctx.observer,
            observed=ctx.observed,
            limit=min(_safe_int(tool_input.get("limit"), 10), 100),
            session_name=ctx.session_name if session_only else None,
        )
        representation = Representation.from_documents(documents)
    total_count = representation.len()
    if total_count == 0:
        return "No recent observations found"
    scope = "this session" if session_only else "all sessions"
    repr_str = (
        representation.str_with_ids()
        if ctx.include_observation_ids
        else str(representation)
    )
    return f"Found {total_count} recent observations from {scope}:\n\n{repr_str}"


async def _handle_get_most_derived_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_most_derived_observations tool."""
    async with tracked_db("tool.get_most_derived_observations", read_only=True) as db:
        documents = await crud.query_documents_most_derived(
            db=db,
            workspace_name=ctx.workspace_name,
            observer=ctx.observer,
            observed=ctx.observed,
            limit=min(_safe_int(tool_input.get("limit"), 10), 100),
        )
        representation = Representation.from_documents(documents)
    total_count = representation.len()
    if total_count == 0:
        return "No established observations found"
    repr_str = (
        representation.str_with_ids()
        if ctx.include_observation_ids
        else str(representation)
    )
    return f"Found {total_count} established (frequently reinforced) observations:\n\n{repr_str}"


async def _handle_get_session_summary(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_session_summary tool."""
    if not ctx.session_name:
        return "ERROR: No session available for summary"
    summary_type = tool_input.get("summary_type", "short")
    st = (
        summarizer.SummaryType.LONG
        if summary_type == "long"
        else summarizer.SummaryType.SHORT
    )
    async with tracked_db("tool.get_session_summary", read_only=True) as db:
        summary = await summarizer.get_summary(
            db, ctx.workspace_name, ctx.session_name, st
        )
    if not summary:
        return "No session summary available yet"
    return f"Session summary ({summary['summary_type']}):\n{summary['content']}"


async def _handle_get_peer_card(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_peer_card tool."""
    _ = tool_input
    async with tracked_db("tool.get_peer_card", read_only=True) as db:
        peer_card = await crud.get_peer_card(
            db,
            workspace_name=ctx.workspace_name,
            observer=ctx.observer,
            observed=ctx.observed,
        )
    if not peer_card:
        return f"No peer card available for {ctx.observed}"
    return f"Peer card for {ctx.observed}:\n" + "\n".join(
        f"- {fact}" for fact in peer_card
    )


async def _handle_delete_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> "str | ToolResult":
    """Handle delete_observations tool."""
    observation_ids = tool_input.get("observation_ids", [])
    if not observation_ids:
        return "ERROR: observation_ids list is empty"

    async with ctx.db_lock, tracked_db("tool.delete_observations") as db:
        deleted = await crud.delete_documents(
            db,
            workspace_name=ctx.workspace_name,
            document_ids=observation_ids,
            observer=ctx.observer,
            observed=ctx.observed,
        )

    deleted_ids = {doc_id for doc_id, _ in deleted}
    for obs_id in observation_ids:
        if obs_id not in deleted_ids:
            logger.warning(
                "Failed to delete observation %s (not found, already deleted, or wrong scope)",
                obs_id,
            )

    deleted_count = len(deleted)
    if deleted_count > 0 and ctx.run_id and ctx.agent_type and ctx.parent_category:
        emit(
            AgentToolConclusionsDeletedEvent(
                run_id=ctx.run_id,
                iteration=get_current_iteration(),
                parent_category=ctx.parent_category,
                agent_type=ctx.agent_type,
                workspace_name=ctx.workspace_name,
                observer=ctx.observer,
                observed=ctx.observed,
                conclusion_count=deleted_count,
                levels=[level for _, level in deleted],
            )
        )

    # +5: surface deleted_count + levels for DreamSpecialistEvent rollups.
    from src.utils.types import ToolResult

    return ToolResult(
        content=f"Deleted {deleted_count} observations",
        metadata={
            "deleted_count": deleted_count,
            "levels": [level for _, level in deleted],
        },
    )


async def _handle_finish_consolidation(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle finish_consolidation tool."""
    _ = ctx
    summary = tool_input.get("summary", "Consolidation complete")
    return f"CONSOLIDATION_COMPLETE: {summary}"


async def _handle_extract_preferences(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle extract_preferences tool."""
    _ = tool_input
    # Wrap so the batch-embed + downstream search_messages embedding calls
    # all carry preference-extraction attribution.
    with embedding_call_purpose(
        EmbeddingCallPurpose.PREFERENCE_EXTRACTION.value,
        workspace_name=ctx.workspace_name,
        run_id=ctx.run_id,
        parent_category=ctx.parent_category,
    ):
        results = await extract_preferences(
            workspace_name=ctx.workspace_name,
            session_name=ctx.session_name,
            observed=ctx.observed,
            observer=ctx.observer,
        )

    messages = results.get("messages", [])

    if not messages:
        return "No potentially relevant preference or instruction messages found in conversation history."

    output_parts: list[str] = [
        f"**Potentially Relevant Messages ({len(messages)}):**",
        "\n".join(f"- {msg}" for msg in messages),
        "\n**Action Required:** Review these messages and extract any preferences or standing instructions to add to the peer card using `update_peer_card`. "
        + "Summarize as clear rules (e.g., 'INSTRUCTION: Always include cultural context') or preferences (e.g., 'PREFERENCE: Brief responses').",
    ]

    return "\n\n".join(output_parts)


def _format_message_snippets(
    snippets: list[tuple[list[models.Message], list[models.Message]]], desc: str
) -> str:
    """Format message snippets for output.

    Returns bare `str` because callers concatenate it into other strings
    or place it into `ToolResult.content`. Callers that need the
    truncation telemetry signal route their own output through
    `_maybe_truncated_result` themselves.
    """
    snippet_texts: list[str] = []
    total_matches = sum(len(matches) for matches, _ in snippets)
    for i, (matches, context) in enumerate(snippets, 1):
        lines: list[str] = []
        for msg in context:
            truncated = _truncate_message_content(msg.content)
            lines.append(
                format_new_turn_with_timestamp(truncated, msg.created_at, msg.peer_name)
            )
        sess = context[0].session_name if context else "unknown"
        snippet_texts.append(
            f"--- Snippet {i} (session: {sess}, {len(matches)} match(es)) ---\n"
            + "\n".join(lines)
        )

    output = (
        f"Found {total_matches} matching messages in {len(snippets)} conversation snippets {desc}:\n\n"
        + "\n\n".join(snippet_texts)
    )
    # `[0]` extracts the truncated text — telemetry signal is discarded here
    # because callers wrap the result into ToolResult themselves (and so any
    # downstream truncation telemetry should come from the caller's path).
    return _truncate_tool_output(output)[0]


async def _handle_get_reasoning_chain(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_reasoning_chain tool."""
    observation_id = tool_input.get("observation_id")
    if not observation_id:
        return "ERROR: 'observation_id' is required"

    direction = tool_input.get("direction", "both")
    if direction not in ("premises", "conclusions", "both"):
        return f"ERROR: Invalid direction '{direction}'. Must be 'premises', 'conclusions', or 'both'"

    # Get the observation itself
    async with tracked_db("tool.get_reasoning_chain", read_only=True) as db:
        docs = await crud.get_documents_by_ids(db, ctx.workspace_name, [observation_id])
        if not docs or not docs[0]:
            return f"ERROR: Observation '{observation_id}' not found"

        doc: Document = docs[0]

        output_parts: list[str] = []

        # Format the main observation
        level = doc.level or "explicit"
        output_parts.append(f"**Observation [id:{doc.id}] ({level}):**\n{doc.content}")

        # Get premises/sources if requested
        if direction in ("premises", "both"):
            if level == "deductive" and doc.source_ids:
                premises = await crud.get_documents_by_ids(
                    db, ctx.workspace_name, doc.source_ids
                )
                if premises:
                    premise_lines: list[Any] = []
                    for p in premises:
                        p_level = p.level or "explicit"
                        premise_lines.append(f" - [id:{p.id}] ({p_level}): {p.content}")
                    output_parts.append(
                        f"\n**Premises ({len(premises)}):**\n"
                        + "\n".join(premise_lines)
                    )
                else:
                    output_parts.append(
                        f"\n**Premises:** Referenced {len(doc.source_ids)} premise IDs but none found in database"
                    )
            elif level == "inductive" and doc.source_ids:
                sources = await crud.get_documents_by_ids(
                    db, ctx.workspace_name, doc.source_ids
                )
                if sources:
                    source_lines: list[Any] = []
                    for s in sources:
                        s_level = s.level or "explicit"
                        source_lines.append(f" - [id:{s.id}] ({s_level}): {s.content}")
                    output_parts.append(
                        f"\n**Sources ({len(sources)}):**\n" + "\n".join(source_lines)
                    )
                else:
                    output_parts.append(
                        f"\n**Sources:** Referenced {len(doc.source_ids)} source IDs but none found in database"
                    )
            elif level == "explicit":
                output_parts.append(
                    "\n**Premises/Sources:** N/A (explicit observations have no premises)"
                )
            else:
                output_parts.append("\n**Premises/Sources:** None recorded")

        # Get conclusions if requested
        if direction in ("conclusions", "both"):
            children = await crud.get_child_observations(
                db,
                ctx.workspace_name,
                observation_id,
                observer=ctx.observer,
                observed=ctx.observed,
            )
            if children:
                child_lines: list[Any] = []
                for c in children:
                    c_level = c.level or "explicit"
                    child_lines.append(f" - [id:{c.id}] ({c_level}): {c.content}")
                output_parts.append(
                    f"\n**Derived Conclusions ({len(children)}):**\n"
                    + "\n".join(child_lines)
                )
            else:
                output_parts.append("\n**Derived Conclusions:** None found")

    return "\n".join(output_parts)


# Tool handler dispatch table
_TOOL_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], Any]] = {
    "create_observations": _handle_create_observations,
    "create_observations_deductive": _handle_create_observations_deductive,
    "create_observations_inductive": _handle_create_observations_inductive,
    "update_peer_card": _handle_update_peer_card,
    "get_recent_history": _handle_get_recent_history,
    "search_memory": _handle_search_memory,
    "get_observation_context": _handle_get_observation_context,
    "search_messages": _handle_search_messages,
    "grep_messages": _handle_grep_messages,
    "get_messages_by_date_range": _handle_get_messages_by_date_range,
    "search_messages_temporal": _handle_search_messages_temporal,
    "get_recent_observations": _handle_get_recent_observations,
    "get_most_derived_observations": _handle_get_most_derived_observations,
    "get_session_summary": _handle_get_session_summary,
    "get_peer_card": _handle_get_peer_card,
    "delete_observations": _handle_delete_observations,
    "finish_consolidation": _handle_finish_consolidation,
    "extract_preferences": _handle_extract_preferences,
    "get_reasoning_chain": _handle_get_reasoning_chain,
}


async def create_tool_executor(
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    current_messages: list[models.Message] | None = None,
    include_observation_ids: bool = False,
    history_token_limit: int = 8192,
    configuration: ResolvedConfiguration | None = None,
    run_id: str | None = None,
    agent_type: str | None = None,
    parent_category: str | None = None,
) -> Callable[[str, dict[str, Any]], Any]:
    """
    Create a unified tool executor function for all agent operations.

    This factory function captures the agent's context and returns an async callable
    that can execute any tool from AGENT_TOOLS or DIALECTIC_AGENT_TOOLS.

    Each tool handler manages its own short-lived DB sessions via tracked_db(),
    so no long-lived database session is needed.

    Args:
        workspace_name: Workspace identifier
        observer: The peer making observations/queries
        observed: The peer being observed/queried about
        session_name: Session identifier (optional for global queries)
        current_messages: List of current messages being processed (optional, for deriver)
        include_observation_ids: If True, include observation IDs in output (for dreamer agent)
        history_token_limit: Maximum tokens for get_recent_history (default: 8192)
        configuration: Resolved configuration for checking feature flags (optional)
        run_id: Optional run ID for telemetry correlation
        agent_type: Optional agent type for telemetry (dialectic, deriver, dreamer)
        parent_category: Optional parent category for CloudEvents

    Returns:
        An async callable that executes tools with the captured context
    """
    # Get shared lock from registry to prevent race conditions when multiple
    # tool executors operate on the same workspace/observer/observed concurrently
    shared_lock = await get_observation_lock(workspace_name, observer, observed)

    ctx = ToolContext(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        session_name=session_name,
        current_messages=current_messages,
        include_observation_ids=include_observation_ids,
        history_token_limit=history_token_limit,
        db_lock=shared_lock,
        configuration=configuration,
        run_id=run_id,
        agent_type=agent_type,
        parent_category=parent_category,
    )

    async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
        """
        Execute a tool and return result for LLM.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input arguments

        Returns:
            String result describing what was done
        """
        import time

        from src.utils.types import (
            ToolResult,
            get_current_iteration,
            get_current_provider_tool_call_id,
            get_current_tool_call_seq,
            set_last_tool_metadata,
        )

        # Log nondisclosive call shape only. Raw `tool_input` can carry user
        # content (search queries, peer-card text, etc.); the param keys are
        # enough to reconstruct the call shape from telemetry without leaking
        # content to log sinks.
        logger.info("[tool call] %s keys=%s", tool_name, sorted(tool_input.keys()))

        start = time.perf_counter()
        # Defaults populated even on early returns / error paths so the
        # AgentToolCallCompletedEvent emission below can fire consistently.
        result_str: str = ""
        metadata: dict[str, Any] = {}
        is_error: bool = False

        # Langfuse tool observation; auto-parents under the active step span.
        # Closed in the finally below with output + level.
        tool_obs = _begin_tool_observation(tool_name, tool_input)

        try:
            handler = _TOOL_HANDLERS.get(tool_name)
            if handler:
                handler_result = await handler(ctx, tool_input)
                # Handlers return either a plain str (existing contract) or a
                # ToolResult(content, metadata) carrying structured fields for
                # telemetry and specialist rollups.
                if isinstance(handler_result, ToolResult):
                    result_str = handler_result.content
                    metadata = handler_result.metadata
                else:
                    result_str = handler_result
                # Log shape, not contents — `result_str` can carry retrieved
                # observations, message snippets, peer-card text, etc. The
                # AgentToolCallCompletedEvent telemetry captures the
                # structured metadata for analytics.
                logger.info(
                    "[tool result] %s len=%d metadata_keys=%s",
                    tool_name,
                    len(result_str),
                    sorted(metadata.keys()),
                )
            else:
                result_str = f"Unknown tool: {tool_name}"
                is_error = True
                logger.warning(result_str)

        except asyncio.CancelledError:
            # Cancellation (client disconnect, server shutdown) — populate
            # telemetry fields so the finally-block emit records an accurate
            # event, then re-raise so cancellation propagates to the caller.
            # CancelledError extends BaseException, so the broader except
            # clauses below do not catch it.
            result_str = f"Tool {tool_name} cancelled"
            is_error = True
            raise
        except ValueError as e:
            # Recoverable errors (bad input, validation failures) - return to LLM
            result_str = f"Tool {tool_name} failed with invalid input: {e}"
            is_error = True
            logger.warning(result_str)
        except KeyError as e:
            # Missing required parameters - return to LLM
            result_str = f"Tool {tool_name} missing required parameter: {e}"
            is_error = True
            logger.warning(result_str)
        except Exception as e:
            # Unexpected errors - log with full traceback but still return to LLM
            # We don't re-raise because the LLM should be able to continue with other tools
            result_str = (
                f"Tool {tool_name} failed unexpectedly: {type(e).__name__}: {e}"
            )
            is_error = True
            logger.error(result_str, exc_info=True)
            # No explicit rollback needed — each handler uses tracked_db() which
            # handles rollback in its finally block
        finally:
            # Emit in finally so CancelledError (and any other BaseException)
            # still produces an AgentToolCallCompletedEvent before propagating.
            duration_ms = (time.perf_counter() - start) * 1000

            # Publish ToolResult.metadata for tool_loop to stash on all_tool_calls.
            # Reset to {} (rather than leaving stale metadata) so a non-ToolResult
            # handler doesn't appear to have leaked metadata from a prior call.
            set_last_tool_metadata(metadata)

            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name=tool_name,
                duration_ms=duration_ms,
                result_str=result_str,
                metadata=metadata,
                is_error=is_error,
                iteration=get_current_iteration(),
                tool_call_seq=get_current_tool_call_seq(),
                provider_tool_call_id=get_current_provider_tool_call_id(),
            )

            _finish_tool_observation(tool_obs, result_str, is_error)

        return result_str

    return execute_tool


def _begin_tool_observation(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Open a non-current Langfuse "tool" observation for one tool execution.

    Auto-parents under the active step span (else standalone). Returns a handle
    (closed by `_finish_tool_observation`) or None when disabled/setup fails.
    All tools are ``as_type="tool"`` — they share one generic dispatcher.
    """
    if not settings.LANGFUSE_PUBLIC_KEY:
        return None
    try:
        from langfuse import get_client

        return get_client().start_observation(
            as_type="tool", name=tool_name, input=tool_input
        )
    except Exception:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to open Langfuse tool observation", exc_info=True)
        return None


def _finish_tool_observation(tool_obs: Any, result_str: str, is_error: bool) -> None:
    """Close a Langfuse tool observation opened by `_begin_tool_observation`."""
    if tool_obs is None:
        return
    try:
        tool_obs.update(output=result_str, level="ERROR" if is_error else None)
        tool_obs.end()
    except Exception:  # pragma: no cover - best-effort telemetry
        logger.debug("Failed to close Langfuse tool observation", exc_info=True)


def _emit_agent_tool_call_completed(
    *,
    ctx: "ToolContext",
    tool_name: str,
    duration_ms: float,
    result_str: str,
    metadata: dict[str, Any],
    is_error: bool,
    iteration: int,
    tool_call_seq: int,
    provider_tool_call_id: str | None,
) -> None:
    """Build and emit AgentToolCallCompletedEvent. Best-effort; swallows errors.

    Skipped when the executor was constructed without agent identifiers
    (run_id / agent_type / parent_category) — telemetry attribution requires
    all three.
    """
    if not (ctx.run_id and ctx.agent_type and ctx.parent_category):
        return
    try:
        from src.telemetry.events import AgentToolCallCompletedEvent, emit

        emit(
            AgentToolCallCompletedEvent(
                run_id=ctx.run_id,
                iteration=iteration,
                tool_call_seq=tool_call_seq,
                provider_tool_call_id=provider_tool_call_id,
                parent_category=ctx.parent_category,
                agent_type=ctx.agent_type,
                workspace_name=ctx.workspace_name,
                tool_name=tool_name,
                duration_ms=duration_ms,
                is_error=is_error,
                result_chars=len(result_str),
                result_chars_before_truncation=metadata.get(
                    "result_chars_before_truncation"
                ),
                result_tokens_estimate=_estimate_tokens(result_str),
                was_truncated=bool(metadata.get("was_truncated", False)),
                query_tokens=metadata.get("query_tokens"),
                top_k=metadata.get("top_k"),
                results_count=metadata.get("results_count"),
                used_embedding=metadata.get("used_embedding"),
                embedding_query_count=int(metadata.get("embedding_query_count") or 0),
            )
        )
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("Failed to emit AgentToolCallCompletedEvent", exc_info=True)


def _estimate_tokens(text: str) -> int:
    """Tiktoken-based size proxy for tool result strings. Best-effort."""
    if not text:
        return 0
    try:
        import tiktoken

        # Use cl100k_base as a stable default — matches the embedding-client
        # fallback. Exact accuracy isn't required; this is a size proxy.
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fall back to a rough char→token ratio so the field is always populated.
        return max(1, len(text) // 4)


def _estimate_tokens_safe(text: str | None) -> int | None:
    """Wrapper around `_estimate_tokens` that returns None on falsy input.

    Used by search-handler metadata where we want `query_tokens=None`
    when the query is empty rather than 0 (which could be confused with a
    real measurement).
    """
    if not text:
        return None
    return _estimate_tokens(text)

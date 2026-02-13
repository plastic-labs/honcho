import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.models import Document
from src.schemas import ResolvedConfiguration
from src.telemetry.events import (
    AgentToolConclusionsCreatedEvent,
    AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
    emit,
)
from src.utils import summarizer
from src.utils.formatting import format_new_turn_with_timestamp, utc_now_iso
from src.utils.representation import Representation
from src.utils.types import get_current_iteration

logger = logging.getLogger(__name__)


# Module-level lock registry for thread-safe observation creation.
# Keyed by (workspace_name, observer, observed) to ensure all tool executors
# operating on the same data share the same lock.
_observation_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


async def get_observation_lock(
    workspace_name: str, observer: str, observed: str
) -> asyncio.Lock:
    """
    Get or create a lock for a specific workspace/observer/observed combination.

    This ensures that concurrent tool executors operating on the same observation
    space share a lock, preventing race conditions during document creation.

    Args:
        workspace_name: Workspace identifier
        observer: The observing peer
        observed: The peer being observed

    Returns:
        An asyncio.Lock shared by all executors for this combination
    """
    key = (workspace_name, observer, observed)
    async with _registry_lock:
        if key not in _observation_locks:
            _observation_locks[key] = asyncio.Lock()
        return _observation_locks[key]


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


def _truncate_tool_output(output: str, max_chars: int | None = None) -> str:
    """Truncate tool output to prevent token explosion."""
    if max_chars is None:
        max_chars = settings.LLM.MAX_TOOL_OUTPUT_CHARS
    if len(output) <= max_chars:
        return output
    truncated = output[:max_chars]
    return (
        truncated
        + f"\n\n[OUTPUT TRUNCATED - showing {max_chars:,} of {len(output):,} characters]"
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
        "description": "Create observations at any level: explicit (facts), deductive (logical necessities), inductive (patterns), or contradiction (conflicting statements). Use this to record facts, logical inferences, patterns, or note when the user has said contradictory things.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of observations to create",
                    "items": {
                        "type": "object",
                        "properties": {
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
                                "description": "Level: 'explicit' for direct facts, 'deductive' for logical necessities, 'inductive' for patterns, 'contradiction' for conflicting statements",
                            },
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "(For deductive/inductive/contradiction) Document IDs of source/premise observations - REQUIRED",
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
                                "description": "(For inductive only) Confidence level: 'high' for 3+ sources, 'medium' for 2+, 'low' for tentative",
                            },
                        },
                        "required": ["content", "level"],
                    },
                },
            },
            "required": ["observations"],
        },
    },
    "create_observations_deductive": {
        "name": "create_observations",
        "description": "Create new deductive observations discovered while answering the query. Use this when you infer something new about the peer that isn't already captured in existing observations. Only use for novel deductions - not for restating existing facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "description": "List of new deductive observations to create",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The observation content - should be a self-contained statement about the peer",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["observations"],
        },
    },
    "update_peer_card": {
        "name": "update_peer_card",
        "description": "Update the peer card with facts about the observed peer. The peer card is a summary of key information about the peer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "array",
                    "description": "List of facts about the peer",
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
    TOOLS["create_observations"],
    TOOLS["delete_observations"],
    TOOLS["update_peer_card"],
]

# Tools for the induction specialist (dreamer phase 2)
# Creates inductive observations from explicit and deductive observations
# Includes message access for context and self-directed exploration
# Note: get_peer_card is not included - peer card is injected into the prompt directly
INDUCTION_SPECIALIST_TOOLS: list[dict[str, Any]] = [
    # Discovery tools
    TOOLS["get_recent_observations"],
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    # Action tools
    TOOLS["create_observations"],
    TOOLS["update_peer_card"],
]


async def create_observations(
    db: AsyncSession,
    observations: list[schemas.ObservationInput],
    observer: str,
    observed: str,
    session_name: str | None,
    workspace_name: str,
    message_ids: list[int],
    message_created_at: str,
) -> ObservationsCreatedResult:
    """
    Create multiple observations (documents) in the memory system in a single call.

    Args:
        db: Database session
        observations: List of validated observation inputs
        observer: The peer making the observation
        observed: The peer being observed
        session_name: Session identifier
        workspace_name: Workspace identifier
        message_ids: List of message IDs these observations are based on
        message_created_at: Timestamp of the message that triggered these observations

    Returns:
        ObservationsCreatedResult with created count and any per-observation failures
    """
    if not observations:
        logger.warning("create_observations called with empty list")
        return ObservationsCreatedResult(created_count=0, created_levels=[], failed=[])

    # Get or create collection
    await crud.get_or_create_collection(
        db,
        workspace_name,
        observer=observer,
        observed=observed,
    )

    contents = [obs.content for obs in observations]

    # Batch embed all observation contents.
    # If batching fails, fall back to per-observation embedding.
    embeddings_by_index: dict[int, list[float]] | None = None
    try:
        embeddings = await embedding_client.simple_batch_embed(contents)
        embeddings_by_index = dict(
            zip(range(len(observations)), embeddings, strict=True)
        )
    except Exception as e:
        logger.warning(
            "Batch embedding failed for create_observations; falling back to per-observation embedding: %s",
            e,
        )

    # Build document objects with pre-computed embeddings
    documents: list[schemas.DocumentCreate] = []
    failed: list[ObservationFailure] = []
    for i, obs in enumerate(observations):
        embedding: list[float]
        if embeddings_by_index is not None:
            embedding = embeddings_by_index[i]
        else:
            try:
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

    # Bulk create all documents
    if documents:
        await crud.create_documents(
            db,
            documents=documents,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            deduplicate=True,
        )
        logger.info(
            "Created %d observations in %s/%s/%s",
            len(documents),
            workspace_name,
            observer,
            observed,
        )

    return ObservationsCreatedResult(
        created_count=len(documents),
        created_levels=[doc.level for doc in documents],
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
    db: AsyncSession,
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

    Args:
        db: Database session
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
        db=db,
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

    Returns:
        List of messages in chronological order, including the requested messages and surrounding context
    """
    if not message_ids:
        return []

    # Use a CTE to get seq_in_session values for target messages
    stmt = (
        select(models.Message.seq_in_session)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.public_id.in_(message_ids))
    )

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)

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

    result = await db.execute(stmt)
    messages = list(result.scalars().all())

    return messages


async def extract_preferences(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    observed: str,
) -> dict[str, list[str]]:
    """
    Extract user preferences and standing instructions from conversation history.

    Uses semantic search to find messages that might contain preferences or instructions.
    This is language-agnostic and doesn't rely on keyword matching.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        observed: The peer whose preferences to extract

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

    # Batch embed all queries in a single API call.
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
                db,
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

    db: AsyncSession
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


async def _handle_create_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle create_observations tool."""
    raw_observations = tool_input.get("observations", [])

    if not raw_observations:
        return "ERROR: observations list is empty"

    # Set context-specific default level before Pydantic validation
    default_level = "explicit" if ctx.current_messages else "deductive"
    for obs in raw_observations:
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
            ctx.db,
            observations=observations,
            observer=ctx.observer,
            observed=ctx.observed,
            session_name=ctx.session_name,
            workspace_name=ctx.workspace_name,
            message_ids=message_ids,
            message_created_at=message_created_at,
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

    return response


async def _handle_update_peer_card(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
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

    peer_card_content = tool_input["content"]
    async with ctx.db_lock:
        await crud.set_peer_card(
            ctx.db,
            workspace_name=ctx.workspace_name,
            peer_card=peer_card_content,
            observer=ctx.observer,
            observed=ctx.observed,
        )
    logger.info(
        f"Updated peer card for {ctx.workspace_name}/{ctx.observer}/{ctx.observed}"
    )

    # Emit telemetry event if context is available
    if ctx.run_id and ctx.agent_type and ctx.parent_category:
        # Count facts in peer card (content is a list of strings per tool schema)
        facts_count: int
        if isinstance(peer_card_content, list):
            content_list = cast(list[str], peer_card_content)
            facts_count = len([line for line in content_list if line.strip()])
        else:
            # Fallback for string (defensive)
            facts_count = len(
                [line for line in str(peer_card_content).split("\n") if line.strip()]
            )
        emit(
            AgentToolPeerCardUpdatedEvent(
                run_id=ctx.run_id,
                iteration=get_current_iteration(),
                parent_category=ctx.parent_category,
                agent_type=ctx.agent_type,
                workspace_name=ctx.workspace_name,
                observer=ctx.observer,
                observed=ctx.observed,
                facts_count=facts_count,
            )
        )

    return f"Updated peer card for {ctx.observed} by {ctx.observer}"


async def _handle_get_recent_history(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_recent_history tool."""
    _ = tool_input
    history: list[models.Message] = await get_recent_history(
        ctx.db,
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
    return _truncate_tool_output(output)


async def _handle_search_memory(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle search_memory tool."""
    top_k = min(tool_input.get("top_k", 20), 40)
    query = tool_input["query"]
    try:
        query_embedding = await embedding_client.embed(query)
    except ValueError:
        return f"ERROR: Query exceeds maximum token limit of {settings.MAX_EMBEDDING_TOKENS}. Please use a shorter query."

    documents = await crud.query_documents(
        db=ctx.db,
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
        # fallback behavior: if the memory is *empty*, that means we're quite
        # early in a workspace/peer/session -- in order to give good answers in
        # this stage, and be efficient with tool calls, and make sure the model
        # doesn't short-circuit and think there's nothing here, we
        # automatically search the message history for relevant information.
        if ctx.agent_type == "dialectic":
            limit = min(tool_input.get("top_k", 20), 20)
            snippets = await crud.search_messages(
                ctx.db,
                workspace_name=ctx.workspace_name,
                session_name=ctx.session_name,
                query=query,
                limit=limit,
                context_window=0,
                embedding=query_embedding,
            )
            if snippets:
                message_output = _format_message_snippets(
                    snippets, f"for query '{query}'"
                )
                return (
                    f"No observations yet. Message search results:\n\n{message_output}"
                )
            return (
                f"No observations found for query '{query}', and no messages found in "
                "history. Try a different phrasing or use grep_messages for exact text."
            )
        return f"No observations found for query '{query}'"
    mem_str = mem.str_with_ids() if ctx.include_observation_ids else str(mem)
    return f"Found {total_count} observations for query '{query}':\n\n{mem_str}"


async def _handle_get_observation_context(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_observation_context tool."""
    messages = await get_observation_context(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        message_ids=tool_input["message_ids"],
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
    return _truncate_tool_output(output)


async def _handle_search_messages(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle search_messages tool."""
    query = tool_input["query"]
    limit = min(tool_input.get("limit", 10), 20)  # Cap at 20
    snippets = await crud.search_messages(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        query=query,
        limit=limit,
        context_window=2,
    )
    if not snippets:
        return f"No messages found for query '{query}'"

    return _format_message_snippets(snippets, f"for query '{query}'")


async def _handle_grep_messages(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle grep_messages tool."""
    text = tool_input.get("text", "")
    if not text:
        return "ERROR: 'text' parameter is required"
    limit = min(tool_input.get("limit", 10), 30)  # Cap at 30
    context_window = min(tool_input.get("context_window", 2), 2)  # Cap context

    snippets = await crud.grep_messages(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        text=text,
        limit=limit,
        context_window=context_window,
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
    return _truncate_tool_output(output)


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
) -> str:
    """Handle get_messages_by_date_range tool."""
    after_date_str = tool_input.get("after_date")
    before_date_str = tool_input.get("before_date")
    limit = min(tool_input.get("limit", 20), 20)
    order = tool_input.get("order", "desc")

    after_date = _parse_date(after_date_str, "after_date")
    if isinstance(after_date, str):
        return after_date  # Error message

    before_date = _parse_date(before_date_str, "before_date")
    if isinstance(before_date, str):
        return before_date  # Error message

    messages = await crud.get_messages_by_date_range(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        order=order,
    )

    date_range: list[str] = []
    if after_date_str:
        date_range.append(f"after {after_date_str}")
    if before_date_str:
        date_range.append(f"before {before_date_str}")

    if not messages:
        range_desc = " and ".join(date_range) if date_range else "specified range"
        return f"No messages found {range_desc}"

    messages_text = "\n".join(
        [
            format_new_turn_with_timestamp(
                _truncate_message_content(m.content), m.created_at, m.peer_name
            )
            for m in messages
        ]
    )

    range_desc = " and ".join(date_range) if date_range else "all time"
    order_desc = "oldest first" if order == "asc" else "newest first"

    output = f"Found {len(messages)} messages ({range_desc}, {order_desc}):\n\n{messages_text}"
    return _truncate_tool_output(output)


async def _handle_search_messages_temporal(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle search_messages_temporal tool."""
    query = tool_input.get("query", "")
    if not query:
        return "ERROR: 'query' parameter is required"

    after_date_str = tool_input.get("after_date")
    before_date_str = tool_input.get("before_date")
    limit = min(tool_input.get("limit", 10), 10)
    context_window = min(tool_input.get("context_window", 2), 2)

    after_date = _parse_date(after_date_str, "after_date")
    if isinstance(after_date, str):
        return after_date

    before_date = _parse_date(before_date_str, "before_date")
    if isinstance(before_date, str):
        return before_date

    snippets = await crud.search_messages_temporal(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        query=query,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        context_window=context_window,
    )

    date_filter: list[str] = []
    if after_date_str:
        date_filter.append(f"after {after_date_str}")
    if before_date_str:
        date_filter.append(f"before {before_date_str}")
    filter_desc = f" ({' and '.join(date_filter)})" if date_filter else ""

    if not snippets:
        return f"No messages found for query '{query}'{filter_desc}"

    return _format_message_snippets(snippets, f"for query '{query}'{filter_desc}")


async def _handle_get_recent_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle get_recent_observations tool."""
    session_only = tool_input.get("session_only", False)
    documents = await crud.query_documents_recent(
        db=ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        limit=tool_input.get("limit", 10),
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
    documents = await crud.query_documents_most_derived(
        db=ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        limit=tool_input.get("limit", 10),
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
    summary = await summarizer.get_summary(
        ctx.db, ctx.workspace_name, ctx.session_name, st
    )
    if not summary:
        return "No session summary available yet"
    return f"Session summary ({summary['summary_type']}):\n{summary['content']}"


async def _handle_get_peer_card(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_peer_card tool."""
    _ = tool_input
    peer_card = await crud.get_peer_card(
        ctx.db,
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
) -> str:
    """Handle delete_observations tool."""
    observation_ids = tool_input.get("observation_ids", [])
    if not observation_ids:
        return "ERROR: observation_ids list is empty"

    deleted_count = 0
    async with ctx.db_lock:
        for obs_id in observation_ids:
            try:
                await crud.delete_document(
                    ctx.db,
                    workspace_name=ctx.workspace_name,
                    document_id=obs_id,
                    observer=ctx.observer,
                    observed=ctx.observed,
                )
                deleted_count += 1
            except Exception as e:
                logger.warning("Failed to delete observation %s: %s", obs_id, e)

    # Emit telemetry event if context is available
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
            )
        )

    return f"Deleted {deleted_count} observations"


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
    results = await extract_preferences(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        observed=ctx.observed,
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
    """Format message snippets for output."""
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
    return _truncate_tool_output(output)


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
    docs = await crud.get_documents_by_ids(ctx.db, ctx.workspace_name, [observation_id])
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
                ctx.db, ctx.workspace_name, doc.source_ids
            )
            if premises:
                premise_lines: list[Any] = []
                for p in premises:
                    p_level = p.level or "explicit"
                    premise_lines.append(f"  - [id:{p.id}] ({p_level}): {p.content}")
                output_parts.append(
                    f"\n**Premises ({len(premises)}):**\n" + "\n".join(premise_lines)
                )
            else:
                output_parts.append(
                    f"\n**Premises:** Referenced {len(doc.source_ids)} premise IDs but none found in database"
                )
        elif level == "inductive" and doc.source_ids:
            sources = await crud.get_documents_by_ids(
                ctx.db, ctx.workspace_name, doc.source_ids
            )
            if sources:
                source_lines: list[Any] = []
                for s in sources:
                    s_level = s.level or "explicit"
                    source_lines.append(f"  - [id:{s.id}] ({s_level}): {s.content}")
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
            ctx.db,
            ctx.workspace_name,
            observation_id,
            observer=ctx.observer,
            observed=ctx.observed,
        )
        if children:
            child_lines: list[Any] = []
            for c in children:
                c_level = c.level or "explicit"
                child_lines.append(f"  - [id:{c.id}] ({c_level}): {c.content}")
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
    db: AsyncSession,
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

    Args:
        db: Database session
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
        db=db,
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
        logger.info("[tool call] %s %s", tool_name, tool_input)

        try:
            handler = _TOOL_HANDLERS.get(tool_name)
            if handler:
                result = await handler(ctx, tool_input)
                logger.info("[tool result] %s %s", tool_name, result)
                return result
            return f"Unknown tool: {tool_name}"

        except ValueError as e:
            # Recoverable errors (bad input, validation failures) - return to LLM
            error_msg = f"Tool {tool_name} failed with invalid input: {e}"
            logger.warning(error_msg)
            return error_msg
        except KeyError as e:
            # Missing required parameters - return to LLM
            error_msg = f"Tool {tool_name} missing required parameter: {e}"
            logger.warning(error_msg)
            return error_msg
        except Exception as e:
            # Unexpected errors - log with full traceback but still return to LLM
            # We don't re-raise because the LLM should be able to continue with other tools
            error_msg = f"Tool {tool_name} failed unexpectedly: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            # Rollback the transaction to clear any failed state
            # This is critical for PostgreSQL which blocks subsequent queries on failed transactions
            await ctx.db.rollback()
            return error_msg

    return execute_tool

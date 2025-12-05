import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.embedding_client import embedding_client
from src.utils import summarizer
from src.utils.formatting import format_new_turn_with_timestamp, utc_now_iso
from src.utils.representation import Representation
from src.utils.types import DocumentLevel

logger = logging.getLogger(__name__)

# Maximum characters for tool output to prevent token explosion
MAX_TOOL_OUTPUT_CHARS = 30000  # ~7500 tokens


def _truncate_tool_output(output: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output to prevent token explosion."""
    if len(output) <= max_chars:
        return output
    truncated = output[:max_chars]
    return (
        truncated
        + f"\n\n[OUTPUT TRUNCATED - showing {max_chars:,} of {len(output):,} characters]"
    )


def _truncate_message_content(content: str, max_chars: int = 2000) -> str:
    """Truncate individual message content (simple beginning truncation)."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def _extract_pattern_snippet(content: str, pattern: str, max_chars: int = 2000) -> str:
    """Extract snippet around a regex pattern match.

    For grep/exact text search, finds the pattern and extracts context around it.
    """
    import re

    if len(content) <= max_chars:
        return content

    match = re.search(pattern, content, re.IGNORECASE)
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
        "description": "Create observations, the core unit of information in the memory system, about the observed peer. Use this to record explicit facts mentioned in conversation or deductive inferences about the peer's preferences, behaviors, or characteristics.",
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
                                "enum": ["explicit", "deductive"],
                                "description": "Level of observation: 'explicit' for directly stated facts, 'deductive' for inferred information",
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
        "description": "Search for observations in memory using semantic similarity. Use this to find relevant information about the peer when you need to recall specific details. For enumeration questions, call this MULTIPLE times with different query terms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 20). Use higher values for enumeration questions.",
                    "default": 5,
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
                    "items": {"type": "integer"},
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
}

# Tools for the deriver agent (ingestion)
DERIVER_TOOLS: list[dict[str, Any]] = [
    # TOOLS["search_memory"],
    # TOOLS["search_messages"],
    TOOLS["create_observations"],
    TOOLS["update_peer_card"],
]

# Tools for the dialectic agent (analysis)
DIALECTIC_TOOLS: list[dict[str, Any]] = [
    TOOLS["search_memory"],
    TOOLS["search_messages"],
    TOOLS["get_observation_context"],
    TOOLS["get_peer_card"],
    TOOLS["create_observations_deductive"],
    TOOLS["grep_messages"],  # For exact text search (names, dates, keywords)
    TOOLS["get_messages_by_date_range"],  # For temporal/date-based queries
    TOOLS["search_messages_temporal"],  # Semantic search + date filtering
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
    # Completion signal
    TOOLS["finish_consolidation"],
]


async def create_observations(
    db: AsyncSession,
    observations: list[dict[str, str]],
    observer: str,
    observed: str,
    session_name: str,
    workspace_name: str,
    message_ids: list[int],
    message_created_at: str,
) -> None:
    """
    Create multiple observations (documents) in the memory system in a single call.

    Args:
        db: Database session
        observations: List of observations, each with 'content' and 'level' keys
        observer: The peer making the observation
        observed: The peer being observed
        session_name: Session identifier
        workspace_name: Workspace identifier
        message_ids: List of message IDs these observations are based on
        message_created_at: Timestamp of the message that triggered these observations
    """
    if not observations:
        logger.warning("create_observations called with empty list")
        return

    # Get or create collection
    await crud.get_or_create_collection(
        db,
        workspace_name,
        observer=observer,
        observed=observed,
    )

    # Generate embeddings and create document objects for all observations
    documents: list[schemas.DocumentCreate] = []
    for obs in observations:
        content = obs.get("content", "")
        level_str = obs.get("level", "explicit")

        if not content:
            logger.warning("Skipping observation with empty content")
            continue

        # Validate and cast level
        level: DocumentLevel = "deductive" if level_str == "deductive" else "explicit"

        # Generate embedding for the observation
        embedding = await embedding_client.embed(content)

        # Create document
        doc = schemas.DocumentCreate(
            content=content,
            session_name=session_name,
            level=level,
            metadata=schemas.DocumentMetadata(
                message_ids=message_ids,
                message_created_at=message_created_at,
            ),
            embedding=embedding,
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
            f"Created {len(documents)} observations in {workspace_name}/{observer}/{observed}"
        )


async def update_peer_card(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    content: list[str],
) -> None:
    """
    Update the peer card for an observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer maintaining the card
        observed: The peer the card is about
        content: List of facts/information about the observed peer
    """
    await crud.set_peer_card(
        db,
        workspace_name=workspace_name,
        peer_card=content,
        observer=observer,
        observed=observed,
    )
    logger.info(f"Updated peer card for {workspace_name}/{observer}/{observed}")


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
    limit: int = 5,
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

    Returns:
        Representation object containing relevant observations
    """
    documents = await crud.query_documents(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        query=query,
        top_k=limit,
    )

    return Representation.from_documents(documents)


async def get_observation_context(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    message_ids: list[int],
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
        .where(models.Message.id.in_(message_ids))
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


async def search_messages(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    query: str,
    limit: int = 10,
    context_window: int = 2,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity and return conversation snippets.

    Overlapping snippets within the same session are merged to avoid repetition.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        query: Search query text
        limit: Maximum number of matching messages to return
        context_window: Number of messages before/after each match

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    return await crud.search_messages(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        query=query,
        limit=limit,
        context_window=context_window,
    )


async def grep_messages(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    text: str,
    limit: int = 10,
    context_window: int = 2,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages containing specific text (case-insensitive).

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        text: Text to search for
        limit: Maximum messages to return
        context_window: Number of messages before/after each match

    Returns:
        List of tuples: (matched_messages, context_messages)
    """
    return await crud.grep_messages(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        text=text,
        limit=limit,
        context_window=context_window,
    )


async def get_messages_by_date_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 20,
    order: str = "desc",
) -> list[models.Message]:
    """
    Get messages within a date range.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        after_date: Return messages after this datetime
        before_date: Return messages before this datetime
        limit: Maximum messages to return
        order: Sort order - 'asc' or 'desc'

    Returns:
        List of messages within the date range
    """
    return await crud.get_messages_by_date_range(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        order=order,
    )


async def search_messages_temporal(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    query: str,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int = 10,
    context_window: int = 2,
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity with optional date filtering.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        query: Search query text
        after_date: Only return messages after this datetime
        before_date: Only return messages before this datetime
        limit: Maximum messages to return
        context_window: Number of messages before/after each match

    Returns:
        List of tuples: (matched_messages, context_messages)
    """
    return await crud.search_messages_temporal(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        query=query,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        context_window=context_window,
    )


async def get_recent_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    limit: int = 10,
    session_name: str | None = None,
) -> Representation:
    """
    Get the most recent observations about a peer.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer who made the observations
        observed: The peer who was observed
        limit: Maximum number of observations
        session_name: Optional session name to filter by

    Returns:
        Representation object containing recent observations
    """
    documents = await crud.query_documents_recent(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        limit=limit,
        session_name=session_name,
    )

    return Representation.from_documents(documents)


async def get_most_derived_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    limit: int = 10,
) -> Representation:
    """
    Get observations that have been reinforced most frequently.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer who made the observations
        observed: The peer who was observed
        limit: Maximum number of observations

    Returns:
        Representation object containing most-derived observations
    """
    documents = await crud.query_documents_most_derived(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        limit=limit,
    )

    return Representation.from_documents(documents)


async def get_session_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    summary_type: str = "short",
) -> summarizer.Summary | None:
    """
    Get the session summary.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        summary_type: "short" or "long"

    Returns:
        Summary dict or None if no summary exists
    """
    st = (
        summarizer.SummaryType.LONG
        if summary_type == "long"
        else summarizer.SummaryType.SHORT
    )
    return await summarizer.get_summary(db, workspace_name, session_name, st)


async def get_peer_card(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> list[str] | None:
    """
    Get the peer card containing biographical information.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer who maintains the card
        observed: The peer the card is about

    Returns:
        List of facts about the peer, or None
    """
    return await crud.get_peer_card(
        db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )


async def delete_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    observation_ids: list[str],
) -> int:
    """
    Delete observations by their IDs.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: The peer who made the observations
        observed: The peer who was observed
        observation_ids: List of observation IDs to delete

    Returns:
        Number of observations deleted
    """
    deleted_count = 0
    for obs_id in observation_ids:
        try:
            await crud.delete_document(
                db,
                workspace_name=workspace_name,
                document_id=obs_id,
                observer=observer,
                observed=observed,
            )
            deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to delete observation {obs_id}: {e}")
    return deleted_count


async def extract_preferences(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    observed: str,
) -> dict[str, list[str]]:
    """
    Extract user preferences and standing instructions from conversation history.

    Performs both semantic and text searches to find:
    - Standing instructions ("always do X", "never mention Y")
    - Communication preferences ("I prefer brief responses")
    - Content preferences ("include examples", "use bullet points")
    - Decision-making preferences ("I prefer logical approaches")

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier (optional)
        observed: The peer whose preferences to extract

    Returns:
        Dict with 'instructions' and 'preferences' lists
    """
    instructions: list[str] = []
    preferences: list[str] = []
    seen_content: set[str] = set()  # Dedupe by content hash

    # Text patterns to search for standing instructions
    instruction_patterns = [
        "always",
        "never",
        "don't ever",
        "make sure to",
        "remember to",
        "when I ask",
        "whenever I",
    ]

    # Text patterns for preferences
    preference_patterns = [
        "I prefer",
        "I like",
        "I want",
        "I'd rather",
        "I would rather",
        "I enjoy",
    ]

    # Semantic queries for broader coverage
    semantic_queries = [
        "user preferences and communication style",
        "standing instructions and rules to follow",
        "how user wants responses formatted",
        "things user always or never wants",
    ]

    # 1. Text search for instruction patterns
    for pattern in instruction_patterns:
        try:
            snippets = await crud.grep_messages(
                db,
                workspace_name=workspace_name,
                session_name=session_name,
                text=pattern,
                limit=10,
                context_window=0,  # Just the matching message
            )
            for matches, _ in snippets:
                for msg in matches:
                    if msg.peer_name == observed:
                        content_key = msg.content[:100].lower()
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            # Extract the instruction
                            instructions.append(f"'{msg.content.strip()}'")
        except Exception as e:
            logger.warning(f"Error searching for pattern '{pattern}': {e}")

    # 2. Text search for preference patterns
    for pattern in preference_patterns:
        try:
            snippets = await crud.grep_messages(
                db,
                workspace_name=workspace_name,
                session_name=session_name,
                text=pattern,
                limit=10,
                context_window=0,
            )
            for matches, _ in snippets:
                for msg in matches:
                    if msg.peer_name == observed:
                        content_key = msg.content[:100].lower()
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            preferences.append(f"'{msg.content.strip()}'")
        except Exception as e:
            logger.warning(f"Error searching for pattern '{pattern}': {e}")

    # 3. Semantic search for broader coverage
    for query in semantic_queries:
        try:
            snippets = await crud.search_messages(
                db,
                workspace_name=workspace_name,
                session_name=session_name,
                query=query,
                limit=5,
                context_window=0,
            )
            for matches, _ in snippets:
                for msg in matches:
                    if msg.peer_name == observed:
                        content_lower = msg.content.lower()
                        # Check if this message contains preference-like content
                        if any(
                            p in content_lower
                            for p in instruction_patterns + preference_patterns
                        ):
                            content_key = msg.content[:100].lower()
                            if content_key not in seen_content:
                                seen_content.add(content_key)
                                if any(
                                    p in content_lower for p in instruction_patterns
                                ):
                                    instructions.append(f"'{msg.content.strip()}'")
                                else:
                                    preferences.append(f"'{msg.content.strip()}'")
        except Exception as e:
            logger.warning(f"Error in semantic search for '{query}': {e}")

    return {
        "instructions": instructions[:20],  # Cap at 20 each
        "preferences": preferences[:20],
    }


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

    def __init__(
        self,
        db: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str | None,
        current_messages: list[models.Message] | None,
        include_observation_ids: bool,
        history_token_limit: int,
    ):
        self.db = db
        self.workspace_name = workspace_name
        self.observer = observer
        self.observed = observed
        self.session_name = session_name
        self.current_messages = current_messages
        self.include_observation_ids = include_observation_ids
        self.history_token_limit = history_token_limit


async def _handle_create_observations(
    ctx: ToolContext, tool_input: dict[str, Any]
) -> str:
    """Handle create_observations tool."""
    observations = tool_input.get("observations", [])

    if not observations:
        return "ERROR: observations list is empty"

    # Determine message context based on whether we have current_messages
    if ctx.current_messages:
        # Deriver agent: require level field, use message IDs from batch
        for i, obs in enumerate(observations):
            if "content" not in obs:
                return f"ERROR: observation {i} missing 'content' field"
            if "level" not in obs:
                return f"ERROR: observation {i} missing 'level' field"
            if obs["level"] not in ["explicit", "deductive"]:
                return f"ERROR: observation {i} has invalid level '{obs['level']}', must be 'explicit' or 'deductive'"

        message_ids = [msg.id for msg in ctx.current_messages]
        message_created_at = str(ctx.current_messages[-1].created_at)
        obs_session_name = ctx.session_name or ctx.current_messages[0].session_name
    else:
        # Dialectic agent: force deductive, no source messages
        if not ctx.session_name:
            return "ERROR: Cannot create observations without a session context"

        for obs in observations:
            if "content" not in obs:
                return "ERROR: observation missing 'content' field"
            obs["level"] = "deductive"

        message_ids = []
        message_created_at = utc_now_iso()
        obs_session_name = ctx.session_name

    await create_observations(
        ctx.db,
        observations=observations,
        observer=ctx.observer,
        observed=ctx.observed,
        session_name=obs_session_name,
        workspace_name=ctx.workspace_name,
        message_ids=message_ids,
        message_created_at=message_created_at,
    )

    explicit_count = sum(1 for o in observations if o.get("level") == "explicit")
    deductive_count = sum(1 for o in observations if o.get("level") == "deductive")
    return f"Created {len(observations)} observations for {ctx.observed} by {ctx.observer} ({explicit_count} explicit, {deductive_count} deductive)"


async def _handle_update_peer_card(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle update_peer_card tool."""
    await update_peer_card(
        ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        content=tool_input["content"],
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
    top_k = min(tool_input.get("top_k", 5), 20)  # Cap at 20
    mem: Representation = await search_memory(
        ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        query=tool_input["query"],
        limit=top_k,
    )
    total_count = mem.len()
    if total_count == 0:
        return f"No observations found for query '{tool_input['query']}'"
    mem_str = mem.str_with_ids() if ctx.include_observation_ids else str(mem)
    return f"Found {total_count} observations for query '{tool_input['query']}':\n\n{mem_str}"


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
    limit = min(tool_input.get("limit", 10), 10)  # Cap at 10
    snippets = await search_messages(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        query=query,
        limit=limit,
    )
    if not snippets:
        return f"No messages found for query '{query}'"

    return _format_message_snippets(snippets, f"for query '{query}'")


async def _handle_grep_messages(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle grep_messages tool."""
    text = tool_input.get("text", "")
    if not text:
        return "ERROR: 'text' parameter is required"
    limit = min(tool_input.get("limit", 10), 15)  # Cap at 15
    context_window = min(tool_input.get("context_window", 2), 2)  # Cap context

    snippets = await grep_messages(
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

    messages = await get_messages_by_date_range(
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

    snippets = await search_messages_temporal(
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
    representation: Representation = await get_recent_observations(
        ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        limit=tool_input.get("limit", 10),
        session_name=ctx.session_name if session_only else None,
    )
    total_count = len(representation.explicit) + len(representation.deductive)
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
    representation = await get_most_derived_observations(
        ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        limit=tool_input.get("limit", 10),
    )
    total_count = len(representation.explicit) + len(representation.deductive)
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
    summary = await get_session_summary(
        ctx.db,
        workspace_name=ctx.workspace_name,
        session_name=ctx.session_name,
        summary_type=tool_input.get("summary_type", "short"),
    )
    if not summary:
        return "No session summary available yet"
    return f"Session summary ({summary['summary_type']}):\n{summary['content']}"


async def _handle_get_peer_card(ctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_peer_card tool."""
    _ = tool_input
    peer_card = await get_peer_card(
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

    deleted_count = await delete_observations(
        ctx.db,
        workspace_name=ctx.workspace_name,
        observer=ctx.observer,
        observed=ctx.observed,
        observation_ids=observation_ids,
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

    instructions = results["instructions"]
    preferences = results["preferences"]

    if not instructions and not preferences:
        return "No preferences or standing instructions found in conversation history."

    output_parts: list[str] = []

    if instructions:
        output_parts.append(
            f"**Standing Instructions Found ({len(instructions)}):**\n"
            + "\n".join(f"- {inst}" for inst in instructions)
        )

    if preferences:
        output_parts.append(
            f"**Preferences Found ({len(preferences)}):**\n"
            + "\n".join(f"- {pref}" for pref in preferences)
        )

    output_parts.append(
        "\n**Action Required:** Review these and add relevant ones to the peer card using `update_peer_card`. "
        + "Summarize standing instructions as clear rules (e.g., 'Always include cultural context when discussing social norms')."
    )

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
}


def create_tool_executor(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    current_messages: list[models.Message] | None = None,
    include_observation_ids: bool = False,
    history_token_limit: int = 8192,
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

    Returns:
        An async callable that executes tools with the captured context
    """
    ctx = ToolContext(
        db=db,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        session_name=session_name,
        current_messages=current_messages,
        include_observation_ids=include_observation_ids,
        history_token_limit=history_token_limit,
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
        logger.info(f"[tool call] {tool_name}")

        try:
            handler = _TOOL_HANDLERS.get(tool_name)
            if handler:
                return await handler(ctx, tool_input)
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
            return error_msg

    return execute_tool

"""
Tool definitions and executors for agentic dialectic.

Provides tools that the dialectic model can use to gather additional context:
- message_search: Search for relevant messages with surrounding context
- search_observations: Search theory-of-mind observations/facts
- get_messages_by_time: Retrieve messages from a time range
- get_session_info: Get session metadata and participants
- get_working_representation: Get theory-of-mind representation
- get_recent_conversation: Get recent conversation history
"""

import logging
from datetime import datetime
from typing import Any

from dateparser import parse as parse_date
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.crud.message import get_messages_id_range
from src.embedding_client import embedding_client

logger = logging.getLogger(__name__)

# Tool definitions in Anthropic format
MESSAGE_SEARCH_TOOL = {
    "name": "search_messages",
    "description": "Search for relevant messages in the session using semantic similarity. Returns the top 3 most relevant conversation segments with context (3 messages before and after each match). If multiple matches are nearby in the conversation, they are intelligently clustered to avoid repetition.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant messages",
            },
        },
        "required": ["query"],
    },
}

OBSERVATION_SEARCH_TOOL = {
    "name": "search_observations",
    "description": "Search the theory-of-mind observations (facts and conclusions) about the observed peer. These are derived insights extracted from conversations. Returns relevant observations with their level (explicit facts vs deductive conclusions), content, and when they were created. Use this to find specific knowledge that may not be immediately obvious from raw messages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant observations/facts",
            },
            "level": {
                "type": "string",
                "enum": ["explicit", "deductive", "all"],
                "description": "Filter by observation level: 'explicit' (direct facts), 'deductive' (inferred conclusions), or 'all' (default: 'all')",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 10)",
            },
        },
        "required": ["query"],
    },
}

TIME_RANGE_SEARCH_TOOL = {
    "name": "get_messages_by_time",
    "description": "Retrieve messages from a specific time period. Use this to answer questions about when events occurred, what was discussed during a time range, or how topics evolved over time. Returns messages with timestamps in chronological order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "time_range": {
                "type": "string",
                "description": "Time period to search. Examples: 'last week', 'last 3 days', 'January 2024', 'yesterday', 'last month'",
            },
            "peer_filter": {
                "type": "string",
                "description": "Optional: filter messages to/from a specific peer name",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to return (default: 20)",
            },
        },
        "required": ["time_range"],
    },
}

SESSION_INFO_TOOL = {
    "name": "get_session_info",
    "description": "Get information about the current session including active participants, when they joined, their observation settings, and session metadata. Use this to understand conversation context, multi-peer dynamics, or session-specific settings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "include_stats": {
                "type": "boolean",
                "description": "Whether to include statistics like message counts per peer (default: false)",
            },
        },
    },
}

WORKING_REPRESENTATION_TOOL = {
    "name": "get_working_representation",
    "description": "Retrieve the working representation (theory-of-mind model) for the observed peer. This includes both current session conclusions and historical facts from the global representation. Use this to get a comprehensive understanding of what is known about the observed peer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "include_semantic_query": {
                "type": "string",
                "description": "Optional: a query to semantically filter the most relevant observations",
            },
            "session_scoped": {
                "type": "boolean",
                "description": "Whether to scope the representation to the current session only (default: false)",
            },
        },
    },
}

CONVERSATION_HISTORY_TOOL = {
    "name": "get_recent_conversation",
    "description": "Get recent conversation history from the current session, including any summaries of earlier messages. Use this to understand the immediate context of the conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "message_limit": {
                "type": "integer",
                "description": "Maximum number of recent messages to return (default: 20)",
            },
            "include_summary": {
                "type": "boolean",
                "description": "Whether to include session summary of older messages (default: true)",
            },
        },
    },
}

# All available tools
AVAILABLE_TOOLS = [
    MESSAGE_SEARCH_TOOL,
    OBSERVATION_SEARCH_TOOL,
    TIME_RANGE_SEARCH_TOOL,
    SESSION_INFO_TOOL,
    WORKING_REPRESENTATION_TOOL,
    CONVERSATION_HISTORY_TOOL,
]


async def execute_message_search(
    db: AsyncSession,
    workspace_name: str,
    query: str,
    context_messages: int = 3,
) -> str:
    """
    Execute semantic search over messages in a session.

    Searches for the top 3 most relevant messages and intelligently clusters
    nearby results to avoid repetition. If multiple matches are close together
    in the conversation, they are returned as a single context window.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        query: Search query text
        context_messages: Number of messages before/after to include (default: 3)

    Returns:
        Formatted string with search results and context
    """
    try:
        # Generate embedding for the search query
        embedding = await embedding_client.embed(query)

        # Always search for top 3 most relevant messages
        stmt = (
            select(models.Message)
            .join(
                models.MessageEmbedding,
                models.Message.public_id == models.MessageEmbedding.message_id,
            )
            .where(models.Message.workspace_name == workspace_name)
            .order_by(models.MessageEmbedding.embedding.cosine_distance(embedding))
            .limit(3)
        )

        result = await db.execute(stmt)
        matching_messages = list(result.scalars().all())

        if not matching_messages:
            return "No matching messages found."

        # Sort matches by message ID to process in chronological order
        matching_messages.sort(key=lambda m: m.id)

        # Cluster nearby matches to avoid repetition
        # Two matches are in the same cluster if they're within (2 * context_messages) of each other
        cluster_distance = 2 * context_messages

        clusters: list[list[models.Message]] = []
        current_cluster: list[models.Message] = [matching_messages[0]]

        for msg in matching_messages[1:]:
            # Check if this message is close to the last message in the current cluster
            if msg.id - current_cluster[-1].id <= cluster_distance:
                # Add to current cluster
                current_cluster.append(msg)
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [msg]

        # Don't forget the last cluster
        clusters.append(current_cluster)

        # Build result blocks for each cluster
        formatted_results: list[str] = []
        match_ids = {msg.id for msg in matching_messages}

        for cluster_idx, cluster in enumerate(clusters, 1):
            # Calculate the unified context window for this cluster
            # Start from context_messages before the first match
            # End at context_messages after the last match
            start_id = max(1, cluster[0].id - context_messages)
            end_id = cluster[-1].id + context_messages + 1

            # Get all messages in this window
            context_msgs = await get_messages_id_range(
                db,
                workspace_name=workspace_name,
                session_name=cluster[0].session_name,
                start_id=start_id,
                end_id=end_id,
            )

            # Format the messages, marking all matches in this cluster
            formatted_context: list[str] = []
            for ctx_msg in context_msgs:
                marker = " <-- MATCH" if ctx_msg.id in match_ids else ""
                formatted_context.append(
                    f"  [{ctx_msg.peer_name}]: {ctx_msg.content}{marker}"
                )

            # If there are multiple clusters, label them
            if len(clusters) > 1:
                result_block = f"""Result {cluster_idx} ({len(cluster)} match{"es" if len(cluster) > 1 else ""}):
{chr(10).join(formatted_context)}
"""
            else:
                result_block = f"""Result:
{chr(10).join(formatted_context)}
"""
            formatted_results.append(result_block)

        return "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error executing message search: {e}")
        return f"Error searching messages: {str(e)}"


async def execute_observation_search(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    query: str,
    level: str = "all",
    top_k: int = 10,
) -> str:
    """
    Execute semantic search over theory-of-mind observations.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observer peer
        observed: Name of the observed peer
        query: Search query text
        level: Filter by observation level (explicit/deductive/all)
        top_k: Number of results to return

    Returns:
        Formatted string with observation results
    """
    try:
        # Get the collection for this observer/observed pair
        collection = await crud.get_collection(
            db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        if not collection:
            return f"No observations found for {observer} observing {observed}."

        # Generate embedding for the search query
        embedding = await embedding_client.embed(query)

        # Build the query with level filter if specified
        level_filter = None
        if level != "all":
            level_filter = {"level": level}

        # Search documents
        documents = await crud.query_documents(
            db,
            workspace_name=workspace_name,
            query=query,
            observer=observer,
            observed=observed,
            filters=level_filter,
            embedding=embedding,
            top_k=top_k,
        )

        if not documents:
            return "No matching observations found."

        # Format results
        formatted_results: list[str] = []
        for idx, doc in enumerate(documents, 1):
            level_label = doc.internal_metadata.get("level", "unknown").upper()
            created_at = doc.created_at.strftime("%Y-%m-%d %H:%M:%S")

            formatted_results.append(
                f"{idx}. [{level_label}] {doc.content}\n   (Created: {created_at})"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error executing observation search: {e}")
        return f"Error searching observations: {str(e)}"


async def execute_time_range_search(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    time_range: str,
    peer_filter: str | None = None,
    limit: int = 20,
) -> str:
    """
    Execute search for messages within a time range.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        time_range: Natural language time range (e.g., "last week")
        peer_filter: Optional peer name filter
        limit: Maximum number of messages to return

    Returns:
        Formatted string with messages
    """
    try:
        # Parse the time range using dateparser
        # dateparser handles phrases like "last week", "yesterday", etc.
        parsed_date = parse_date(
            time_range,
            settings={
                "RELATIVE_BASE": datetime.now(),
                "PREFER_DATES_FROM": "past",
            },
        )

        if not parsed_date:
            return f"Could not parse time range: '{time_range}'. Try phrases like 'last week', 'yesterday', '3 days ago', etc."

        # Get messages after this date
        stmt = (
            select(models.Message)
            .where(
                and_(
                    models.Message.workspace_name == workspace_name,
                    models.Message.session_name == session_name,
                    models.Message.created_at >= parsed_date,
                )
            )
            .order_by(models.Message.created_at.asc())
            .limit(limit)
        )

        # Add peer filter if specified
        if peer_filter:
            stmt = stmt.where(models.Message.peer_name == peer_filter)

        result = await db.execute(stmt)
        messages = list(result.scalars().all())

        if not messages:
            filter_text = f" from {peer_filter}" if peer_filter else ""
            return f"No messages found{filter_text} since {parsed_date.strftime('%Y-%m-%d %H:%M:%S')}."

        # Format results
        formatted_messages: list[str] = []
        for msg in messages:
            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
            formatted_messages.append(f"[{timestamp}] {msg.peer_name}: {msg.content}")

        header = f"Messages from {parsed_date.strftime('%Y-%m-%d %H:%M:%S')} onwards"
        if peer_filter:
            header += f" (filtered to {peer_filter})"
        header += ":\n\n"

        return header + "\n".join(formatted_messages)

    except Exception as e:
        logger.error(f"Error executing time range search: {e}")
        return f"Error searching by time range: {str(e)}"


async def execute_session_info(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    include_stats: bool = False,
) -> str:
    """
    Get session information including participants and metadata.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        include_stats: Whether to include message statistics

    Returns:
        Formatted string with session information
    """
    try:
        # Get session
        session = await crud.get_session(db, workspace_name, session_name)
        if not session:
            return f"Session '{session_name}' not found."

        # Get peers and their configurations
        peer_configs_stmt = await crud.get_session_peer_configuration(
            workspace_name, session_name
        )
        result = await db.execute(peer_configs_stmt)
        peer_configs_raw = result.all()

        # Transform the results into a more usable format
        peer_configs: list[dict[str, Any]] = []
        for peer_name, _peer_config, session_peer_config, is_active in peer_configs_raw:
            peer_configs.append(
                {
                    "peer_name": peer_name,
                    "is_active": is_active,
                    "configuration": session_peer_config,
                    "joined_at": session_peer_config.get("joined_at")
                    if session_peer_config
                    else None,
                    "left_at": session_peer_config.get("left_at")
                    if session_peer_config
                    else None,
                }
            )

        # Build output
        output = [f"Session: {session_name}"]
        output.append(f"Status: {'Active' if session.is_active else 'Inactive'}")
        output.append(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if session.metadata:
            output.append(f"Metadata: {session.metadata}")

        output.append(f"\nParticipants ({len(peer_configs)}):")

        for config in peer_configs:
            peer_name = config["peer_name"]
            is_active = config["is_active"]
            status = "Active" if is_active else "Left"

            peer_info = [f"  - {peer_name} ({status})"]

            joined_at = config.get("joined_at")
            if joined_at:
                peer_info.append(
                    f"    Joined: {joined_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            left_at = config.get("left_at")
            if left_at:
                peer_info.append(f"    Left: {left_at.strftime('%Y-%m-%d %H:%M:%S')}")

            config_data = config.get("configuration", {})
            if config_data:
                peer_info.append(f"    Config: {config_data}")

            output.extend(peer_info)

        # Add message statistics if requested
        if include_stats:
            stmt = (
                select(
                    models.Message.peer_name,
                    func.count(models.Message.public_id).label("message_count"),
                )
                .where(
                    and_(
                        models.Message.workspace_name == workspace_name,
                        models.Message.session_name == session_name,
                    )
                )
                .group_by(models.Message.peer_name)
            )

            result = await db.execute(stmt)
            stats = result.all()

            if stats:
                output.append("\nMessage Statistics:")
                for peer_name, count in stats:
                    output.append(f"  - {peer_name}: {count} messages")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return f"Error getting session info: {str(e)}"


async def execute_get_working_representation(
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    include_semantic_query: str | None = None,
    session_scoped: bool = False,
) -> str:
    """
    Get the working representation (theory-of-mind model) for an observed peer.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observer peer
        observed: Name of the observed peer
        session_name: Optional session name for scoping
        include_semantic_query: Optional query to filter observations
        session_scoped: Whether to scope to session only

    Returns:
        Formatted string with representation
    """
    try:
        representation = await crud.get_working_representation(
            workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name if session_scoped else None,
            include_semantic_query=include_semantic_query,
            semantic_search_top_k=settings.DIALECTIC.SEMANTIC_SEARCH_TOP_K,
            semantic_search_max_distance=settings.DIALECTIC.SEMANTIC_SEARCH_MAX_DISTANCE,
            include_most_derived=True,
        )

        return str(representation)

    except Exception as e:
        logger.error(f"Error getting working representation: {e}")
        return f"Error getting working representation: {str(e)}"


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None = None,
    observer: str | None = None,
    observed: str | None = None,
) -> str:
    """
    Execute a tool by name with the given parameters.

    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters from the model
        db: Database session
        workspace_name: Workspace context
        session_name: Session context (required for some tools)
        observer: Observer peer name (required for some tools)
        observed: Observed peer name (required for some tools)

    Returns:
        Tool execution result as a formatted string
    """
    if tool_name == "search_messages":
        query = parameters.get("query", "")
        return await execute_message_search(
            db=db,
            workspace_name=workspace_name,
            query=query,
        )

    elif tool_name == "search_observations":
        query = parameters.get("query", "")
        level = parameters.get("level", "all")
        top_k = parameters.get("top_k", 10)

        if not observer or not observed:
            return "Error: observer and observed are required for search_observations"

        return await execute_observation_search(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            query=query,
            level=level,
            top_k=top_k,
        )

    elif tool_name == "get_messages_by_time":
        time_range = parameters.get("time_range", "")
        peer_filter = parameters.get("peer_filter")
        limit = parameters.get("limit", 20)

        if not session_name:
            return "Error: session_name is required for get_messages_by_time"

        return await execute_time_range_search(
            db=db,
            workspace_name=workspace_name,
            session_name=session_name,
            time_range=time_range,
            peer_filter=peer_filter,
            limit=limit,
        )

    elif tool_name == "get_session_info":
        include_stats = parameters.get("include_stats", False)

        if not session_name:
            return "Error: session_name is required for get_session_info"

        return await execute_session_info(
            db=db,
            workspace_name=workspace_name,
            session_name=session_name,
            include_stats=include_stats,
        )

    elif tool_name == "get_working_representation":
        include_semantic_query = parameters.get("include_semantic_query")
        session_scoped = parameters.get("session_scoped", False)

        if not observer or not observed:
            return "Error: observer and observed are required for get_working_representation"

        return await execute_get_working_representation(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_semantic_query=include_semantic_query,
            session_scoped=session_scoped,
        )

    else:
        return f"Unknown tool: {tool_name}"

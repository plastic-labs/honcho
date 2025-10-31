"""
Tool definitions and executors for agentic dialectic.

Provides tools that the dialectic model can use to gather additional context:
- message_search: Search for relevant messages with surrounding context
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
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

# All available tools
AVAILABLE_TOOLS = [MESSAGE_SEARCH_TOOL]


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
                result_block = f"""Result {cluster_idx} ({len(cluster)} match{'es' if len(cluster) > 1 else ''}):
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


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    db: AsyncSession,
    workspace_name: str,
) -> str:
    """
    Execute a tool by name with the given parameters.

    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters from the model
        db: Database session
        workspace_name: Workspace context

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
    else:
        return f"Unknown tool: {tool_name}"

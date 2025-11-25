import logging
from collections.abc import Callable
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
        "description": "Search for observations in memory using semantic similarity. Use this to find relevant information about the peer when you need to recall specific details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
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
        "description": "Search for messages using semantic similarity and retrieve conversation snippets. Returns up to 5 matching messages, each with surrounding context (2 messages before and after). Nearby matches within the same session are merged into a single snippet to avoid repetition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to find relevant messages",
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
        "description": "Delete observations by their IDs. Use this to remove redundant, outdated, or low-quality observations during consolidation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of observation IDs to delete",
                },
            },
            "required": ["observation_ids"],
        },
    },
}

# Tools for the deriver agent (ingestion)
DERIVER_TOOLS: list[dict[str, Any]] = [
    TOOLS["search_memory"],
    TOOLS["search_messages"],
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
]

# Tools for the dreamer agent (consolidation)
DREAMER_TOOLS: list[dict[str, Any]] = [
    TOOLS["get_recent_observations"],
    TOOLS["get_most_derived_observations"],
    TOOLS["search_memory"],
    TOOLS["create_observations"],
    TOOLS["delete_observations"],
    TOOLS["update_peer_card"],
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

    Returns:
        List of messages in chronological order
    """
    if session_name:
        # Get messages from a specific session
        messages_stmt = await crud.get_messages(
            workspace_name=workspace_name,
            session_name=session_name,
            token_limit=8192,  # TODO config
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
) -> list[tuple[list[models.Message], list[models.Message]]]:
    """
    Search for messages using semantic similarity and return conversation snippets.

    Overlapping snippets within the same session are merged to avoid repetition.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        query: Search query text

    Returns:
        List of tuples: (matched_messages, context_messages)
        Each snippet may contain multiple matches if they were close together.
    """
    return await crud.search_messages(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        query=query,
        limit=5,
        context_window=2,
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


def create_tool_executor(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    current_messages: list[models.Message] | None = None,
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

    Returns:
        An async callable that executes tools with the captured context
    """

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
            # === WRITE TOOLS ===

            if tool_name == "create_observations":
                observations = tool_input.get("observations", [])

                if not observations:
                    return "ERROR: observations list is empty"

                # Determine message context based on whether we have current_messages
                if current_messages:
                    # Deriver agent: require level field, use message IDs from batch
                    for i, obs in enumerate(observations):
                        if "content" not in obs:
                            return f"ERROR: observation {i} missing 'content' field"
                        if "level" not in obs:
                            return f"ERROR: observation {i} missing 'level' field"
                        if obs["level"] not in ["explicit", "deductive"]:
                            return f"ERROR: observation {i} has invalid level '{obs['level']}', must be 'explicit' or 'deductive'"

                    message_ids = [msg.id for msg in current_messages]
                    message_created_at = str(current_messages[-1].created_at)
                    obs_session_name = session_name or current_messages[0].session_name
                else:
                    # Dialectic agent: force deductive, no source messages
                    if not session_name:
                        return "ERROR: Cannot create observations without a session context"

                    for obs in observations:
                        if "content" not in obs:
                            return "ERROR: observation missing 'content' field"
                        obs["level"] = "deductive"

                    message_ids = []
                    message_created_at = utc_now_iso()
                    obs_session_name = session_name

                await create_observations(
                    db,
                    observations=observations,
                    observer=observer,
                    observed=observed,
                    session_name=obs_session_name,
                    workspace_name=workspace_name,
                    message_ids=message_ids,
                    message_created_at=message_created_at,
                )

                explicit_count = sum(
                    1 for o in observations if o.get("level") == "explicit"
                )
                deductive_count = sum(
                    1 for o in observations if o.get("level") == "deductive"
                )
                return f"Created {len(observations)} observations for {observed} by {observer} ({explicit_count} explicit, {deductive_count} deductive)"

            elif tool_name == "update_peer_card":
                await update_peer_card(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    content=tool_input["content"],
                )
                return f"Updated peer card for {observed} by {observer}"

            # === READ TOOLS ===

            elif tool_name == "get_recent_history":
                history: list[models.Message] = await get_recent_history(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    observed=observed,
                )
                if not history:
                    return "No conversation history available"
                history_text = "\n".join(
                    [f"{m.peer_name}: {m.content}" for m in history]
                )
                scope = (
                    f"from session {session_name}"
                    if session_name
                    else f"from {observed} across sessions"
                )
                return f"Conversation history ({len(history)} messages {scope}):\n{history_text}"

            elif tool_name == "search_memory":
                mem: Representation = await search_memory(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    query=tool_input["query"],
                )
                total_count = mem.len()
                if total_count == 0:
                    return f"No observations found for query '{tool_input['query']}'"
                return f"Found {total_count} observations for query '{tool_input['query']}':\n\n{mem}"

            elif tool_name == "get_observation_context":
                messages = await get_observation_context(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    message_ids=tool_input["message_ids"],
                )
                if not messages:
                    return f"No messages found for IDs {tool_input['message_ids']}"
                messages_text = "\n".join(
                    [
                        format_new_turn_with_timestamp(
                            m.content, m.created_at, m.peer_name
                        )
                        for m in messages
                    ]
                )
                return (
                    f"Retrieved {len(messages)} messages with context:\n{messages_text}"
                )

            elif tool_name == "search_messages":
                snippets = await search_messages(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    query=tool_input["query"],
                )
                if not snippets:
                    return f"No messages found for query '{tool_input['query']}'"

                # Format each snippet with matched messages highlighted
                snippet_texts: list[str] = []
                total_matches = sum(len(matches) for matches, _ in snippets)
                for i, (matches, context) in enumerate(snippets, 1):
                    lines: list[str] = []
                    for msg in context:
                        lines.append(
                            format_new_turn_with_timestamp(
                                msg.content, msg.created_at, msg.peer_name
                            )
                        )
                    # Get session name from context (first message)
                    sess = context[0].session_name if context else "unknown"
                    snippet_texts.append(
                        f"--- Snippet {i} (session: {sess}, {len(matches)} match(es)) ---\n"
                        + "\n".join(lines)
                    )

                return (
                    f"Found {total_matches} matching messages in {len(snippets)} conversation snippets for query '{tool_input['query']}':\n\n"
                    + "\n\n".join(snippet_texts)
                )

            elif tool_name == "get_recent_observations":
                session_only = tool_input.get("session_only", False)
                representation: Representation = await get_recent_observations(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    limit=tool_input.get("limit", 10),
                    session_name=session_name if session_only else None,
                )
                total_count = len(representation.explicit) + len(
                    representation.deductive
                )
                if total_count == 0:
                    return "No recent observations found"
                scope = "this session" if session_only else "all sessions"
                return f"Found {total_count} recent observations from {scope}:\n\n{representation}"

            elif tool_name == "get_most_derived_observations":
                representation = await get_most_derived_observations(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    limit=tool_input.get("limit", 10),
                )
                total_count = len(representation.explicit) + len(
                    representation.deductive
                )
                if total_count == 0:
                    return "No established observations found"
                return f"Found {total_count} established (frequently reinforced) observations:\n\n{representation}"

            elif tool_name == "get_session_summary":
                if not session_name:
                    return "ERROR: No session available for summary"
                summary = await get_session_summary(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    summary_type=tool_input.get("summary_type", "short"),
                )
                if not summary:
                    return "No session summary available yet"
                return f"Session summary ({summary['summary_type']}):\n{summary['content']}"

            elif tool_name == "get_peer_card":
                peer_card = await get_peer_card(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                )
                if not peer_card:
                    return f"No peer card available for {observed}"
                return f"Peer card for {observed}:\n" + "\n".join(
                    f"- {fact}" for fact in peer_card
                )

            elif tool_name == "delete_observations":
                observation_ids = tool_input.get("observation_ids", [])
                if not observation_ids:
                    return "ERROR: observation_ids list is empty"

                deleted_count = await delete_observations(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    observation_ids=observation_ids,
                )
                return f"Deleted {deleted_count} observations"

            return f"Unknown tool: {tool_name}"

        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    return execute_tool

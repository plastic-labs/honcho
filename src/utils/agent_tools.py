import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.embedding_client import embedding_client
from src.utils.representation import Representation
from src.utils.types import DocumentLevel

logger = logging.getLogger(__name__)


AGENT_TOOLS: list[dict[str, Any]] = [
    {
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
    {
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
    {
        "name": "get_recent_history",
        "description": "Retrieve recent conversation history to get more context about the conversation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "token_limit": {
                    "type": "integer",
                    "description": "Maximum tokens of history to retrieve",
                    "default": 8192,
                },
            },
        },
    },
    {
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
    {
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
    {
        "name": "search_messages",
        "description": "Search for messages in the current session using semantic similarity. Use this to find 5 relevant messages based on content or topic.",
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
    session_name: str,
    token_limit: int = 8192,
) -> list[models.Message]:
    """
    Retrieve recent conversation history.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        token_limit: Maximum tokens to retrieve

    Returns:
        List of messages in chronological order
    """
    messages = await crud.get_messages(
        workspace_name=workspace_name,
        session_name=session_name,
        token_limit=token_limit,
        reverse=True,  # Get most recent first
    )
    result = await db.execute(messages)

    messages = result.scalars().all()
    # Return in chronological order
    return list(reversed(messages))


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
    session_name: str,
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
        session_name: Session identifier
        message_ids: List of message IDs to retrieve

    Returns:
        List of messages in chronological order, including the requested messages and surrounding context
    """
    if not message_ids:
        return []

    # Use a CTE to get seq_in_session values for target messages
    target_seqs_cte = (
        select(models.Message.seq_in_session)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.id.in_(message_ids))
        .cte("target_seqs")
    )

    # Query messages where seq_in_session is within ±1 of any target sequence
    # We use EXISTS with arithmetic to check if the message is adjacent to any target
    stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
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

    result = await db.execute(stmt)
    messages = list(result.scalars().all())

    return messages


async def search_messages(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    query: str,
) -> list[models.MessageEmbedding]:
    """
    Search for messages in the session using semantic similarity.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        query: Search query text

    Returns:
        List of message embeddings ordered by relevance (limited to 5 results)
    """
    return await crud.search_messages(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        query=query,
        limit=5,
    )


def create_agent_tool_executor(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    observer: str,
    observed: str,
    current_messages: list[models.Message],
) -> Callable[[str, dict[str, Any]], Any]:
    """
    Create a tool executor function with the necessary context for agent operations.

    This factory function captures the agent's context (database session, workspace,
    observer/observed peers, etc.) and returns an async callable that can be used
    as the tool_executor parameter in honcho_llm_call.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        session_name: Session identifier
        observer: The peer making observations
        observed: The peer being observed
        current_messages: List of current messages being processed

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
            if tool_name == "create_observations":
                observations = tool_input.get("observations", [])

                if not observations:
                    return "ERROR: observations list is empty"

                # Validate all observations have required fields and valid levels
                for i, obs in enumerate(observations):
                    if "content" not in obs:
                        return f"ERROR: observation {i} missing 'content' field"
                    if "level" not in obs:
                        return f"ERROR: observation {i} missing 'level' field"
                    if obs["level"] not in ["explicit", "deductive"]:
                        return f"ERROR: observation {i} has invalid level '{obs['level']}', must be 'explicit' or 'deductive'"

                # Extract all message IDs from current batch
                message_ids = [msg.id for msg in current_messages]
                message_created_at = str(current_messages[-1].created_at)

                await create_observations(
                    db,
                    observations=observations,
                    observer=observer,
                    observed=observed,
                    session_name=session_name,
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

            elif tool_name == "get_recent_history":
                history: list[models.Message] = await get_recent_history(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    token_limit=tool_input.get("token_limit", 8192),
                )
                history_text = "\n".join(
                    [f"{m.peer_name}: {m.content}" for m in history]
                )
                return (
                    f"Retrieved {len(history)} messages from history:\n{history_text}"
                )

            elif tool_name == "search_memory":
                representation = await search_memory(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    query=tool_input["query"],
                )
                total_count = len(representation.explicit) + len(
                    representation.deductive
                )
                if total_count == 0:
                    return f"No observations found for query '{tool_input['query']}'"
                return f"Found {total_count} observations for query '{tool_input['query']}':\n\n{representation}"

            elif tool_name == "get_observation_context":
                messages = await get_observation_context(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    message_ids=tool_input["message_ids"],
                )
                if not messages:
                    return f"ERROR: No messages found for message IDs {tool_input['message_ids']}"
                messages_text = "\n".join(
                    [
                        f"[{m.created_at.replace(microsecond=0)}] {m.peer_name}: {m.content}"
                        for m in messages
                    ]
                )
                return (
                    f"Retrieved {len(messages)} messages with context:\n{messages_text}"
                )

            elif tool_name == "search_messages":
                results = await search_messages(
                    db,
                    workspace_name=workspace_name,
                    session_name=session_name,
                    query=tool_input["query"],
                )
                if not results:
                    return f"No messages found for query '{tool_input['query']}'"
                results_text = "\n".join(
                    [f"- (ID: {r.message_id}) {r.content}" for r in results]
                )
                return f"Found {len(results)} messages for query '{tool_input['query']}':\n{results_text}"

            return f"Unknown tool: {tool_name}"

        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    return execute_tool

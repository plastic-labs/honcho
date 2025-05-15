import logging
from enum import Enum
from typing import Optional, Union, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.model_client import ModelClient, ModelProvider

from .. import models

logger = logging.getLogger(__name__)

# Export the public functions
__all__ = [
    "get_session_summaries",
    "get_messages_since_message",
    "get_messages_since_latest_summary",
    "create_summary",
    "save_summary_metamessage",
    "get_summarized_history",
    "should_create_summary",
    "MESSAGES_PER_SHORT_SUMMARY",
    "MESSAGES_PER_LONG_SUMMARY",
    "SummaryType",
]


# Configuration constants for summaries
MESSAGES_PER_SHORT_SUMMARY = 20  # How often to create short summaries
MESSAGES_PER_LONG_SUMMARY = 60  # How often to create long summaries


# The types of metamessages to use for summaries
class SummaryType(Enum):
    SHORT = "honcho_chat_summary_short"
    LONG = "honcho_chat_summary_long"


# Default model settings for summary generation
DEFAULT_PROVIDER = ModelProvider.GEMINI
DEFAULT_MODEL = "gemini-2.0-flash-lite"


async def get_session_summaries(
    db: AsyncSession,
    session_id: str,
    summary_type: SummaryType = SummaryType.SHORT,
    only_latest: bool = False,
) -> Union[list[models.Metamessage], Optional[models.Metamessage]]:
    """
    Get summaries for a given session.

    Args:
        db: Database session
        session_id: The session ID
        summary_type: Type of summary to retrieve ("short" or "long")
        only_latest: Whether to return only the latest summary

    Returns:
        If only_latest is True: The most recent summary metamessage, or None if none exists
        If only_latest is False: A list of all summary metamessages for the session
    """
    # Determine the metamessage type based on summary_type
    label = (
        SummaryType.SHORT.value
        if summary_type == SummaryType.SHORT
        else SummaryType.LONG.value
    )

    stmt = (
        select(models.Metamessage)
        .where(models.Metamessage.session_id == session_id)
        .where(models.Metamessage.label == label)
        .order_by(models.Metamessage.id.desc())
    )

    if only_latest:
        stmt = stmt.limit(1)
        result = await db.execute(stmt)
        # Always return a metamessage instance or None
        return result.scalar_one_or_none()
    else:
        result = await db.execute(stmt)
        return list(result.scalars().all())


async def get_messages_since_message(
    db: AsyncSession, session_id: str, message_id: Optional[str] = None
) -> list[models.Message]:
    """
    Get all messages since a specific message.

    Args:
        db: Database session
        session_id: The session ID
        message_id: The reference message ID

    Returns:
        List of messages after the reference message or all messages if message_id is None
    """
    # Base query for messages in this session
    query = (
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.id)
    )

    # If we have a reference message ID, filter to get newer messages
    if message_id:
        # First, get the ID of the message
        message_id_subquery = (
            select(models.Message.id)
            .where(models.Message.public_id == message_id)
            .scalar_subquery()
        )

        # Then filter to only get messages with higher IDs
        query = query.where(models.Message.id > message_id_subquery)

    # Execute query
    result = await db.execute(query)
    return list(result.scalars().all())


async def create_summary(
    messages: list[models.Message],
    previous_summary: Optional[str] = None,
    summary_type: SummaryType = SummaryType.SHORT,
) -> str:
    """
    Generate a summary of the provided messages using an LLM.

    Args:
        messages: List of messages to summarize
        previous_summary: Optional previous summary to provide context
        summary_type: Type of summary to create ("short" or "long")

    Returns:
        A summary of the conversation
    """
    # Combine messages into a conversation format
    conversation = "\n".join(
        [
            f"{'human' if msg.is_user else 'assistant'}: {msg.content}"
            for msg in messages
        ]
    )

    # Adjust system prompt based on summary type
    if summary_type == SummaryType.LONG:
        system_prompt = """You are a system that creates comprehensive summaries of conversations.
Focus on capturing:
1. Key facts and information shared
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed in detail
5. User's apparent emotional state and personality traits
6. Important themes and patterns across the conversation

It is very important that you clearly distinguish between the user's messages and the assistant's messages, and that only the user's literal words are attributed to them.

Provide a thorough and detailed summary that captures the essence of the conversation.
Your summary should serve as a comprehensive record of the important information in this conversation.

Return only the summary without any explanation or meta-commentary."""
    else:  # short summary
        system_prompt = """You are a system that summarizes parts of a conversation to create a concise and accurate summary.
Focus on capturing:
1. Key facts and information shared
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed
5. User's apparent emotional state

It is very important that you clearly distinguish between the user's messages and the assistant's messages, and that only the user's literal words are attributed to them.

Provide a concise, factual summary that captures the essence of the conversation.
Your summary should be detailed enough to serve as context for future messages,
but brief enough to be helpful.

Return only the summary without any explanation or meta-commentary."""

    # Include previous summary if available
    if previous_summary:
        user_prompt = f"""Here is a previous summary of the conversation:
{previous_summary}
Now please summarize these additional messages, incorporating the context from the previous summary.

Your summary should summarize the entire conversation in a self-contained way, such that someone could read it and understand the entire conversation.
{conversation}
Provide a {"comprehensive" if summary_type == SummaryType.LONG else "concise"} summary that captures both the previous context and the new information."""
    else:
        user_prompt = f"""Please summarize the following conversation segment:
{conversation}
Provide a {"comprehensive" if summary_type == SummaryType.LONG else "concise"} summary that captures the key points and context."""

    # Create a model client
    client = ModelClient(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)

    # Generate the summary
    llm_messages = [{"role": "user", "content": user_prompt}]

    try:
        summary = await client.generate(
            messages=llm_messages,
            system=system_prompt,
            max_tokens=1000
            if summary_type == SummaryType.SHORT
            else 2000,  # Allow longer responses for long summaries
            temperature=0.0,
            use_caching=True,
        )
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        # Fallback to a basic summary in case of error
        return f"Conversation with {len(messages)} messages about {messages[-1].content[:30]}..."


async def save_summary_metamessage(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    summary_content: str,
    message_count: int,
    summary_type: SummaryType = SummaryType.SHORT,
) -> models.Metamessage:
    """
    Save a summary as a metamessage.

    Args:
        db: Database session
        user_id: User ID
        session_id: Session ID
        message_id: The ID of the most recent message being summarized
        summary_content: The summary text to save
        message_count: Number of messages covered by this summary
        summary_type: Type of summary to save

    Returns:
        The created metamessage
    """
    # Get the label value from the enum
    label_value = summary_type.value

    # Create and save the metamessage
    metamessage = models.Metamessage(
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
        label=label_value,
        content=summary_content,
        h_metadata={"message_count": message_count, "summary_type": summary_type.name},
    )

    db.add(metamessage)
    await db.commit()

    logger.info(
        f"Saved {summary_type.name.lower()} summary metamessage for session {session_id} covering {message_count} messages"
    )
    return metamessage


async def get_full_history(
    db: AsyncSession, session_id: str
) -> tuple[str, list[models.Message]]:
    """
    Get all messages for a given session.
    """
    messages = await get_messages_since_message(db, session_id)
    return format_messages(messages), messages


async def get_summarized_history(
    db: AsyncSession, session_id: str, summary_type: SummaryType = SummaryType.SHORT
) -> tuple[str, list[models.Message], Optional[models.Metamessage]]:
    """
    Get a summarized version of the chat history by combining the latest summary
    with all messages since that summary.

    Args:
        db: Database session
        session_id: The session ID
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A tuple containing:
        - String formatted history text with summary and recent messages
        - List of messages since the latest summary
        - The latest summary metamessage, or None if no summary exists
    """
    # Get messages since the latest summary and the summary itself
    messages, latest_summary = await get_messages_since_latest_summary(
        db, session_id, summary_type
    )

    # Format messages
    messages_text = format_messages(messages)

    # We know latest_summary is either a Metamessage or None because of the type
    # narrowing in get_messages_since_latest_summary
    if latest_summary:
        # Combine summary with recent messages
        history_text = f"[CONVERSATION SUMMARY: {latest_summary.content}]\n\n[RECENT MESSAGES]\n{messages_text}"
    else:
        # No summary available, return just the messages
        history_text = messages_text
    return history_text, messages, latest_summary


async def get_messages_since_latest_summary(
    db: AsyncSession, session_id: str, summary_type: SummaryType = SummaryType.SHORT
) -> tuple[list[models.Message], Optional[models.Metamessage]]:
    """
    Get all messages since the latest summary for a session.

    This is a convenience method that combines:
    1. Getting the latest summary for the session
    2. Getting all messages since that summary

    Args:
        db: Database session
        session_id: The session ID
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A tuple containing:
        - List of messages since the latest summary (or all messages if no summary exists)
        - The latest summary metamessage, or None if no summary exists
    """
    # Get the latest summary (will be a Metamessage instance or None)
    # We include only_latest=True to get a single Metamessage (not a list)
    summary = await get_session_summaries(
        db, session_id, summary_type=summary_type, only_latest=True
    )

    # Type narrowing - summary is now either None or a Metamessage instance
    latest_summary = cast(Optional[models.Metamessage], summary)

    # Check if we have a valid summary with a message_id
    if latest_summary is not None:
        messages = await get_messages_since_message(
            db, session_id, latest_summary.message_id
        )
        return messages, latest_summary
    else:
        messages = await get_messages_since_message(db, session_id)
        return messages, None


async def should_create_summary(
    db: AsyncSession, session_id: str, summary_type: SummaryType = SummaryType.SHORT
) -> tuple[bool, list[models.Message], Optional[models.Metamessage]]:
    """
    Determine if a new summary should be created for this session.

    Args:
        db: Database session
        session_id: The session ID
        summary_type: Type of summary to check for ("short" or "long")

    Returns:
        Tuple containing:
        - Boolean indicating whether a summary should be created
        - List of messages to be included in the summary
        - The latest summary of the requested type, or None if no summary exists
    """
    messages, latest_summary = await get_messages_since_latest_summary(
        db, session_id, summary_type
    )
    threshold = (
        MESSAGES_PER_SHORT_SUMMARY
        if summary_type == SummaryType.SHORT
        else MESSAGES_PER_LONG_SUMMARY
    )
    should_create = len(messages) >= threshold
    return should_create, messages, latest_summary


def format_messages(messages: list[models.Message]) -> str:
    """
    Format a list of messages into a string.
    """
    if len(messages) == 0:
        return ""
    return "\n".join(
        [f"{'user' if msg.is_user else 'assistant'}: {msg.content}" for msg in messages]
    )

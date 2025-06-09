import logging
import re
from datetime import datetime
from enum import Enum
from typing import Optional, Union, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from inspect import cleandoc as c

from mirascope import llm
from mirascope.integrations.langfuse import with_langfuse


from .. import models

logger = logging.getLogger(__name__)


def parse_session_date_from_metadata(metadata: dict, fallback_to_created_at: bool = True) -> Optional[str]:
    """
    Parse session date from metadata that might contain formats like:
    "1:56 pm on 8 May, 2023" or other date strings.
    
    Returns formatted date string (YYYY-MM-DD) or None if not found/parseable.
    """
    # Check for session_date in metadata
    session_date_str = metadata.get("session_date")
    if not session_date_str:
        return None
    
    try:
        # Handle format like "1:56 pm on 8 May, 2023"
        # Extract the date part after "on"
        date_match = re.search(r'on\s+(\d{1,2}\s+\w+,?\s+\d{4})', session_date_str)
        if date_match:
            date_part = date_match.group(1)
            # Try to parse various date formats
            for fmt in ["%d %B, %Y", "%d %b, %Y", "%d %B %Y", "%d %b %Y"]:
                try:
                    parsed_date = datetime.strptime(date_part, fmt)
                    return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        
        # Try other common formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                parsed_date = datetime.strptime(session_date_str, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
    except Exception as e:
        logger.debug(f"Could not parse session date '{session_date_str}': {e}")
    
    return None

# Export the public functions
__all__ = [
    "get_session_summaries",
    "get_messages_since_message",
    "get_messages_since_latest_summary",
    "get_messages_since_latest_summary_before_message",
    "create_summary",
    "save_summary_metamessage",
    "get_summarized_history",
    "get_summarized_history_before_message",
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
    logger.debug(f"[get_session_summaries] Getting summaries for session_id: {session_id}, type: {summary_type.value}")
    
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
        summary = result.scalar_one_or_none()
        logger.debug(f"[get_session_summaries] Found latest summary: {summary.id if summary else None}")
        return summary
    else:
        result = await db.execute(stmt)
        summaries = list(result.scalars().all())
        logger.debug(f"[get_session_summaries] Found {len(summaries)} summaries")
        return summaries


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
    logger.debug(f"[get_messages_since_message] Getting messages for session_id: {session_id}, since message_id: {message_id}")
    
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
    messages = list(result.scalars().all())
    
    logger.debug(f"[get_messages_since_message] Found {len(messages)} messages for session_id: {session_id}")
    if messages:
        # Log first few messages to verify they belong to correct session
        for i, msg in enumerate(messages[:3]):
            logger.debug(f"[get_messages_since_message] Message {i}: session_id={msg.session_id}, is_user={msg.is_user}, content={msg.content[:30]}...")
    
    return messages


@with_langfuse()
@llm.call(
    provider="google",
    model="gemini-2.0-flash-lite",
    call_params={"max_tokens": 1000},
)
async def create_short_summary(
    messages: list[models.Message],
    previous_summary: Optional[str] = None,
):
    return c(
        f"""
        You are a system that summarizes parts of a conversation to create a concise and accurate summary.
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

        Return only the summary without any explanation or meta-commentary.

        <conversation>
        {format_messages(messages)}
        </conversation>

        <previous_summary>
        {previous_summary}
        </previous_summary>
        """
    )


@with_langfuse()
@llm.call(
    provider="google",
    model="gemini-2.0-flash-lite",
    call_params={"max_tokens": 2000},
)
async def create_long_summary(
    messages: list[models.Message],
    previous_summary: Optional[str] = None,
):
    return c(
        f"""
        You are a system that creates comprehensive summaries of conversations.
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

        Return only the summary without any explanation or meta-commentary.

        <conversation>
        {format_messages(messages)}
        </conversation>

        <previous_summary>  
        {previous_summary}
        </previous_summary>
        """
    )


async def create_summary(
    messages: list[models.Message],
    previous_summary: Optional[str] = None,
    summary_type: SummaryType = SummaryType.SHORT,
):
    if summary_type == SummaryType.SHORT:
        return await create_short_summary(messages, previous_summary)
    elif summary_type == SummaryType.LONG:
        return await create_long_summary(messages, previous_summary)


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
        app_id: App ID
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
    db: AsyncSession, session_id: str, summary_type: SummaryType = SummaryType.SHORT, fallback_to_created_at: bool = True
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
    logger.debug(f"[get_summarized_history] Getting history for session_id: {session_id}")
    
    # Get session creation date for temporal context
    session_stmt = select(models.Session).where(models.Session.public_id == session_id)
    session_result = await db.execute(session_stmt)
    session = session_result.scalar_one_or_none()
    
    if session:
        logger.debug(f"[get_summarized_history] Found session: {session.public_id}, created_at: {session.created_at}")
    else:
        logger.warning(f"[get_summarized_history] No session found for session_id: {session_id}")
    
    if session:
        # Try to get session date from metadata first
        metadata_date = parse_session_date_from_metadata(session.h_metadata)
        if metadata_date:
            session_date = metadata_date
        elif fallback_to_created_at:
            session_date = session.created_at.strftime("%Y-%m-%d")
        else:
            session_date = "unknown"
    else:
        session_date = "unknown"

    # Get messages since the latest summary and the summary itself
    messages, latest_summary = await get_messages_since_latest_summary(
        db, session_id, summary_type
    )
    
    logger.debug(f"[get_summarized_history] Retrieved {len(messages)} messages for session_id: {session_id}")
    if messages:
        logger.debug(f"[get_summarized_history] First message session_id: {messages[0].session_id}, is_user: {messages[0].is_user}, content: {messages[0].content[:50]}...")
        logger.debug(f"[get_summarized_history] Last message session_id: {messages[-1].session_id}, is_user: {messages[-1].is_user}, content: {messages[-1].content[:50]}...")

    # Format messages
    messages_text = format_messages(messages)

    # We know latest_summary is either a Metamessage or None because of the type
    # narrowing in get_messages_since_latest_summary
    if latest_summary:
        # Combine summary with recent messages, optionally including session date
        if session_date and session_date != "unknown":
            history_text = f"[SESSION DATE: {session_date}] [CONVERSATION SUMMARY: {latest_summary.content}]\n\n[RECENT MESSAGES]\n{messages_text}"
        else:
            history_text = f"[CONVERSATION SUMMARY: {latest_summary.content}]\n\n[RECENT MESSAGES]\n{messages_text}"
    else:
        # No summary available, optionally include session date
        if session_date and session_date != "unknown":
            history_text = f"[SESSION DATE: {session_date}]\n{messages_text}"
        else:
            history_text = messages_text
    return history_text, messages, latest_summary


async def get_summarized_history_before_message(
    db: AsyncSession, session_id: str, before_message_id: str, summary_type: SummaryType = SummaryType.SHORT, fallback_to_created_at: bool = True
) -> tuple[str, list[models.Message], Optional[models.Metamessage]]:
    """
    Get a summarized version of the chat history UP TO but NOT INCLUDING a specific message.
    This is used during deriver processing to get the context that existed before the message being processed.

    Args:
        db: Database session
        session_id: The session ID
        before_message_id: The message ID to stop before (exclude this message from history)
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A tuple containing:
        - String formatted history text with summary and recent messages
        - List of messages since the latest summary but before the specified message
        - The latest summary metamessage, or None if no summary exists
    """
    # Get session creation date for temporal context
    session_stmt = select(models.Session).where(models.Session.public_id == session_id)
    session_result = await db.execute(session_stmt)
    session = session_result.scalar_one_or_none()
    
    if session:
        # Try to get session date from metadata first
        metadata_date = parse_session_date_from_metadata(session.h_metadata)
        if metadata_date:
            session_date = metadata_date
        elif fallback_to_created_at:
            session_date = session.created_at.strftime("%Y-%m-%d")
        else:
            session_date = "unknown"
    else:
        session_date = "unknown"

    # Get messages since the latest summary and the summary itself, but stop before the specified message
    messages, latest_summary = await get_messages_since_latest_summary_before_message(
        db, session_id, before_message_id, summary_type
    )

    # Format messages
    messages_text = format_messages(messages)

    # We know latest_summary is either a Metamessage or None because of the type
    # narrowing in get_messages_since_latest_summary_before_message
    if latest_summary:
        # Combine summary with recent messages, optionally including session date
        if session_date and session_date != "unknown":
            history_text = f"[SESSION DATE: {session_date}] [CONVERSATION SUMMARY: {latest_summary.content}]\n\n[RECENT MESSAGES]\n{messages_text}"
        else:
            history_text = f"[CONVERSATION SUMMARY: {latest_summary.content}]\n\n[RECENT MESSAGES]\n{messages_text}"
    else:
        # No summary available, optionally include session date
        if session_date and session_date != "unknown":
            history_text = f"[SESSION DATE: {session_date}]\n{messages_text}"
        else:
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


async def get_messages_since_latest_summary_before_message(
    db: AsyncSession, session_id: str, before_message_id: str, summary_type: SummaryType = SummaryType.SHORT
) -> tuple[list[models.Message], Optional[models.Metamessage]]:
    """
    Get all messages since the latest summary for a session, but stop before a specific message.
    This is used during deriver processing to get the context that existed before the message being processed.

    Args:
        db: Database session
        session_id: The session ID
        before_message_id: The message ID to stop before (exclude this message from results)
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A tuple containing:
        - List of messages since the latest summary but before the specified message
        - The latest summary metamessage, or None if no summary exists
    """
    # Get the latest summary (will be a Metamessage instance or None)
    # We include only_latest=True to get a single Metamessage (not a list)
    summary = await get_session_summaries(
        db, session_id, summary_type=summary_type, only_latest=True
    )

    # Type narrowing - summary is now either None or a Metamessage instance
    latest_summary = cast(Optional[models.Metamessage], summary)

    # Get messages since the summary but before the specified message
    if latest_summary is not None:
        # Get messages since the summary
        messages_since_summary = await get_messages_since_message(
            db, session_id, latest_summary.message_id
        )
    else:
        # No summary exists, get all messages but we'll filter to before current message anyway
        messages_since_summary = await get_messages_since_message(db, session_id)

    # Filter out messages that come at or after the specified message
    # We need to get the database ID of the before_message_id to compare properly
    before_message_stmt = select(models.Message.id).where(models.Message.public_id == before_message_id)
    before_message_result = await db.execute(before_message_stmt)
    before_message_db_id = before_message_result.scalar_one_or_none()

    if before_message_db_id is not None:
        # Simply exclude the current message - keep all messages with lower IDs
        filtered_messages = [msg for msg in messages_since_summary if msg.id < before_message_db_id]
        return filtered_messages, latest_summary
    else:
        # If we can't find the before_message, return all messages (fallback behavior)
        logger.warning(f"Could not find message {before_message_id} to filter before, returning all messages")
        return messages_since_summary, latest_summary


# async def get_most_recent_summary_since_message()
# ``


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
    Format a list of messages into a string with timestamps.
    Groups messages by date when spanning multiple days.
    """
    if len(messages) == 0:
        return ""
    
    formatted_lines = []
    current_date = None
    
    for msg in messages:
        msg_date = msg.created_at.strftime('%Y-%m-%d')
        msg_time = msg.created_at.strftime('%H:%M:%S')
        
        # Add date header if we're on a new day
        if current_date != msg_date:
            if current_date is not None:  # Don't add header for first message
                formatted_lines.append(f"\n--- {msg_date} ---")
            current_date = msg_date
        
        # Format message with HMS timestamp
        role = 'user' if msg.is_user else 'assistant'
        formatted_lines.append(f"{msg_time} {role}: {msg.content}")
    
    return "\n".join(formatted_lines)

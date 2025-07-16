import datetime
import logging
import os
from enum import Enum
from typing import TypedDict

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.utils.clients import honcho_llm_call

from .. import crud, models

logger = logging.getLogger(__name__)


# TypedDict definitions for summary data
class Summary(TypedDict):
    """
    A summary object. Stored in session metadata and used in a session's get_context.

    Attributes:
        content: The summary text.
        message_count: The number of messages covered by this summary.
        summary_type: The type of summary (short or long).
        created_at: The timestamp of when the summary was created (ISO format string).
        message_id: The primary key ID of the message that triggered this summary.
        token_count: The number of tokens in the summary text.
    """

    content: str
    message_count: int
    summary_type: str
    created_at: str
    message_id: int
    token_count: int


# Export the public functions
__all__ = [
    "get_summary",
    "create_summary",
    "save_summary",
    "get_summarized_history",
    "should_create_summary",
    "SummaryType",
    "Summary",
]


# Configuration constants for summaries
MESSAGES_PER_SHORT_SUMMARY = settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
MESSAGES_PER_LONG_SUMMARY = settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY


# The types of summary to store in the session metadata
class SummaryType(Enum):
    SHORT = "honcho_chat_summary_short"
    LONG = "honcho_chat_summary_long"


# Mirascope functions for summaries
@honcho_llm_call(
    provider=settings.SUMMARY.PROVIDER,
    model=settings.SUMMARY.MODEL,
    max_tokens=settings.SUMMARY.MAX_TOKENS_SHORT,
)
async def create_short_summary(
    messages: list[models.Message],
    previous_summary: str | None = None,
):
    return f"""
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
{_format_messages(messages)}
</conversation>

<previous_summary>
{previous_summary or ""}
</previous_summary>
"""


@honcho_llm_call(
    provider=settings.SUMMARY.PROVIDER,
    model=settings.SUMMARY.MODEL,
    max_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
)
async def create_long_summary(
    messages: list[models.Message],
    previous_summary: str | None = None,
):
    return f"""
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
{_format_messages(messages)}
</conversation>

<previous_summary>  
{previous_summary or ""}
</previous_summary>
"""


async def summarize_if_needed(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    peer_name: str,
    message_id: int,
) -> None:
    """Create short/long summaries if thresholds met (baseline copy)."""

    if not session_name:
        return

    summary_start = os.times()[4]
    logger.debug("Checking if summaries should be created for session %s", session_name)

    # STEP 1: Short summary (every 10 messages)
    (
        should_create_short,
        short_messages,
        _,
    ) = await should_create_summary(
        db,
        workspace_name,
        session_name,
        peer_name,
        message_id,
        summary_type=SummaryType.SHORT,
    )

    if should_create_short:
        logger.debug("Short summary needed for %d messages", len(short_messages))

        # STEP 2: Check long summary need
        (
            should_create_long,
            long_messages,
            latest_long_summary,
        ) = await should_create_summary(
            db,
            workspace_name,
            session_name,
            peer_name,
            message_id,
            summary_type=SummaryType.LONG,
        )

        # STEP 3: Long summary first (if required)
        if should_create_long:
            logger.debug(
                "Creating new long summary covering %d messages", len(long_messages)
            )
            try:
                previous_long_text = (
                    latest_long_summary["content"] if latest_long_summary else None
                )

                new_long = await create_summary(
                    messages=long_messages,
                    previous_summary_text=previous_long_text,
                    summary_type=SummaryType.LONG,
                )

                await save_summary(
                    db,
                    new_long,
                    workspace_name,
                    session_name,
                )
                logger.debug("Long summary created and saved successfully")
            except Exception:
                logger.exception("Error creating long summary")
        else:
            logger.debug(
                "No long summary needed. Need %d messages since last long summary.",
                MESSAGES_PER_LONG_SUMMARY,
            )

        # STEP 4: Short summary creation
        logger.debug(
            "Creating new short summary covering %d messages", len(short_messages)
        )
        try:
            previous_long_text = (
                latest_long_summary["content"] if latest_long_summary else None
            )

            new_short = await create_summary(
                messages=short_messages,
                previous_summary_text=previous_long_text,
                summary_type=SummaryType.SHORT,
            )

            await save_summary(
                db,
                new_short,
                workspace_name,
                session_name,
            )
            logger.debug("Short summary created and saved successfully")
        except Exception:
            logger.exception("Error creating short summary")
    else:
        logger.debug(
            "No short summary needed. Need %d messages since last short summary.",
            MESSAGES_PER_SHORT_SUMMARY,
        )

    summary_time = os.times()[4] - summary_start
    logger.debug("Summary check completed in %.2fs", summary_time)


async def get_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    summary_type: SummaryType = SummaryType.SHORT,
) -> Summary | None:
    """
    Get summary for a given session or peer.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        summary_type: Type of summary to retrieve ("short" or "long")

    Returns:
        The summary data dictionary, or None if no summary exists
    """
    from src.exceptions import ResourceNotFoundException

    label = (
        SummaryType.SHORT.value
        if summary_type == SummaryType.SHORT
        else SummaryType.LONG.value
    )

    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None

    summaries: dict[str, Summary] = session.internal_metadata.get("summaries", {})
    if not summaries or label not in summaries:
        return None
    return summaries[label]


async def create_summary(
    messages: list[models.Message],
    previous_summary_text: str | None = None,
    summary_type: SummaryType = SummaryType.SHORT,
    max_tokens: int | None = None,
) -> Summary:
    """
    Generate a summary of the provided messages using an LLM.

    Args:
        messages: List of messages to summarize
        previous_summary_text: Optional previous summary to provide context
        summary_type: Type of summary to create ("short" or "long")
        max_tokens: Optional maximum number of tokens to generate. Supersedes summary_type.

    Returns:
        A summary of the conversation
    """
    try:
        if summary_type == SummaryType.SHORT:
            response = await create_short_summary(messages, previous_summary_text)
        else:
            response = await create_long_summary(messages, previous_summary_text)

        summary_text = str(response)
        calculated_max_tokens = max_tokens or (
            1000 if summary_type == SummaryType.SHORT else 2000
        )

        logger.info("Successfully generated summary for session")
        return Summary(
            content=summary_text,
            message_count=len(messages),
            summary_type=summary_type.value,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            message_id=messages[-1].id if messages else 0,
            token_count=calculated_max_tokens,
        )
    except Exception:
        logger.exception("Error generating summary")
        # Fallback to a basic summary in case of error
        return Summary(
            content=(
                f"Conversation with {len(messages)} messages about {messages[-1].content[:30]}..."
                if messages
                else ""
            ),
            message_count=0,
            summary_type=summary_type.value,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            message_id=messages[-1].id if messages else 0,
            token_count=50,
        )


async def save_summary(
    db: AsyncSession,
    summary: Summary,
    workspace_name: str,
    session_name: str,
) -> None:
    """
    Save a summary as metadata on a session.

    Args:
        db: Database session
        summary: The summary to save
        workspace_name: Workspace name
        session_name: Session name

    Returns:
        The updated session
    """
    from src.exceptions import ResourceNotFoundException

    # Get the label value from the enum
    label_value = summary["summary_type"]

    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, we can't save the summary
        logger.warning(
            f"Cannot save summary: session {session_name} not found in workspace {workspace_name}"
        )
        return

    # Use SQLAlchemy update() with PostgreSQL's || operator to properly merge JSONB
    # We need to merge the new summary into the existing summaries structure
    update_data = {}
    existing_summaries = session.internal_metadata.get("summaries", {})
    existing_summaries[label_value] = summary
    update_data["summaries"] = existing_summaries

    stmt = (
        update(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
        .values(
            internal_metadata=models.Session.internal_metadata.op("||")(update_data)
        )
    )

    await db.execute(stmt)
    await db.commit()

    logger.info(
        "Saved %s for session %s covering %s messages",
        summary["summary_type"],
        session_name,
        summary["message_count"],
    )


async def get_summarized_history(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str | None,
    cutoff: int | None = None,
    summary_type: SummaryType = SummaryType.SHORT,
) -> str:
    """
    Get a summarized version of the chat history by combining the latest summary
    with all messages since that summary.

    Note: history is exclusive of the cutoff message.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        peer_name: The peer name (None for session-level summaries)
        cutoff: (Optional) message ID to cutoff at
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A string formatted history text with summary and recent messages
    """
    # Get messages since the latest summary and the summary itself
    messages, latest_summary = await _get_latest_summary_and_messages_since(
        db, workspace_name, session_name, peer_name, cutoff, summary_type
    )

    # Format messages
    messages_text = _format_messages(messages)

    if latest_summary:
        # Combine summary with recent messages
        return f"[CONVERSATION SUMMARY: {latest_summary['content']}]\n\n[RECENT MESSAGES]\n{messages_text}"
    else:
        # No summary available, return just the messages
        return messages_text


async def _get_latest_summary_and_messages_since(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str | None,
    cutoff: int | None = None,
    summary_type: SummaryType = SummaryType.SHORT,
) -> tuple[list[models.Message], Summary | None]:
    """
    Get all messages since the latest summary for a session or peer.

    This is a convenience method that combines:
    1. Getting the latest summary for the session or peer
    2. Getting all messages since that summary

    Note that if the latest summary is not found, this will return all messages
    since the start of the session.

    Note: history is exclusive of the cutoff message.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        peer_name: The peer name (None for session-level summaries)
        cutoff: (Optional) message ID to cutoff at
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A tuple containing:
        - List of messages since the latest summary (or all messages if no summary exists)
        - The latest summary data, or None if no summary exists
    """
    # Get the latest summary
    summary = await get_summary(db, workspace_name, session_name, summary_type)

    # Check if we have a valid summary with a message_id
    if summary:
        messages = await crud.get_messages_id_range(
            db,
            workspace_name,
            session_name,
            peer_name,
            start_id=summary["message_id"],
            end_id=cutoff,
        )
        return messages, summary
    else:
        messages = await crud.get_messages_id_range(
            db, workspace_name, session_name, peer_name, end_id=cutoff
        )
        return messages, None


async def should_create_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str | None,
    message_id: int,
    summary_type: SummaryType = SummaryType.SHORT,
) -> tuple[bool, list[models.Message], Summary | None]:
    """
    Determine if a new summary should be created for this object (peer or session).

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        peer_name: The peer name (None for session-level summaries)
        message_id: The message ID to cutoff at
        summary_type: Type of summary to check for ("short" or "long")

    Returns:
        Tuple containing:
        - Boolean indicating whether a summary should be created
        - List of messages to be included in the summary
        - The latest summary of the requested type, or None if no summary exists
    """
    messages, latest_summary = await _get_latest_summary_and_messages_since(
        db,
        workspace_name,
        session_name,
        peer_name,
        cutoff=message_id,
        summary_type=summary_type,
    )
    threshold = (
        MESSAGES_PER_SHORT_SUMMARY
        if summary_type == SummaryType.SHORT
        else MESSAGES_PER_LONG_SUMMARY
    )
    should_create = len(messages) >= threshold
    logger.debug(
        "Should create summary: %s, messages: %s, threshold: %s",
        should_create,
        len(messages),
        threshold,
    )
    return should_create, messages, latest_summary


def _format_messages(messages: list[models.Message]) -> str:
    """
    Format a list of messages into a string by concatenating their content and
    prefixing each with the peer name.
    """
    if len(messages) == 0:
        return ""
    return "\n".join([f"{msg.peer_name}: {msg.content}" for msg in messages])

import asyncio
import datetime
import logging
import time
from enum import Enum
from typing import TypedDict

from mirascope import llm
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.exceptions import ResourceNotFoundException
from src.utils.clients import honcho_llm_call
from src.utils.logging import accumulate_metric

from .. import crud, models

logger = logging.getLogger(__name__)


# TypedDict definitions for summary data
class Summary(TypedDict):
    """
    A summary object. Stored in session metadata and used in a session's get_context.

    Attributes:
        content: The summary text.
        message_id: The primary key ID of the message that this summary covers up to.
        summary_type: The type of summary (short or long).
        created_at: The timestamp of when the summary was created (ISO format string).
        token_count: The number of tokens in the summary text.
    """

    content: str
    message_id: int
    summary_type: str
    created_at: str
    token_count: int


# Export the public functions
__all__ = [
    "get_summary",
    "get_both_summaries",
    "get_summarized_history",
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
    return_call_response=True,
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

It is very important that you clearly distinguish between each member of the conversation, and that only a user's literal words are attributed to them.

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
    return_call_response=True,
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

It is very important that you clearly distinguish between each member of the conversation, and that only a user's literal words are attributed to them.

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
    session_name: str,
    message_id: int,
    message_seq_in_session: int,
) -> None:
    """
    Create short/long summaries if thresholds met.

    This function checks for both short and long summary needs independently,
    without assuming any relationship between their thresholds.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        message_id: The message ID
    """

    logger.debug("Checking if summaries should be created for session %s", session_name)

    should_create_long: bool = message_seq_in_session % MESSAGES_PER_LONG_SUMMARY == 0
    should_create_short: bool = message_seq_in_session % MESSAGES_PER_SHORT_SUMMARY == 0

    # If both summaries need to be created, run them in parallel with separate database sessions
    if should_create_long and should_create_short:

        async def create_long_summary():
            from src.dependencies import tracked_db

            async with tracked_db("create_long_summary") as db_session:
                await _create_and_save_summary(
                    db_session,
                    workspace_name,
                    session_name,
                    message_id,
                    SummaryType.LONG,
                )

        async def create_short_summary():
            from src.dependencies import tracked_db

            async with tracked_db("create_short_summary") as db_session:
                await _create_and_save_summary(
                    db_session,
                    workspace_name,
                    session_name,
                    message_id,
                    SummaryType.SHORT,
                )

        await asyncio.gather(
            create_long_summary(),
            create_short_summary(),
            return_exceptions=True,
        )
        logger.debug("Both long and short summaries created and saved successfully")
    else:
        # If only one summary needs to be created, run them individually
        if should_create_long:
            await _create_and_save_summary(
                db,
                workspace_name,
                session_name,
                message_id,
                SummaryType.LONG,
            )
            logger.debug("Long summary created and saved successfully")
        elif should_create_short:
            await _create_and_save_summary(
                db,
                workspace_name,
                session_name,
                message_id,
                SummaryType.SHORT,
            )
            logger.debug("Short summary created and saved successfully")


async def _create_and_save_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: int,
    summary_type: SummaryType,
) -> None:
    """
    Create a new summary and save it to the database.
    1. Get the latest summary
    2. Get the messages since the latest summary
    3. Generate a new summary using the messages and the previous summary
    4. Save the new summary to the database
    """

    logger.info("Creating new %s summary", summary_type.name)
    # Time summarization step
    summary_start = time.perf_counter()

    latest_summary = await get_summary(db, workspace_name, session_name, summary_type)

    previous_summary_text = latest_summary["content"] if latest_summary else None

    messages = await crud.get_messages_id_range(
        db,
        workspace_name,
        session_name,
        start_id=latest_summary["message_id"] if latest_summary else 0,
        end_id=message_id,
    )

    new_summary = await _create_summary(
        messages=messages,
        previous_summary_text=previous_summary_text,
        summary_type=summary_type,
    )

    await _save_summary(
        db,
        new_summary,
        workspace_name,
        session_name,
    )

    summary_duration = (time.perf_counter() - summary_start) * 1000
    accumulate_metric(
        f"deriver_message_{message_id}",
        f"{summary_type.name}_summary_creation",
        summary_duration,
        "ms",
    )


async def _create_summary(
    messages: list[models.Message],
    previous_summary_text: str | None = None,
    summary_type: SummaryType = SummaryType.SHORT,
) -> Summary:
    """
    Generate a summary of the provided messages using an LLM.

    Args:
        messages_since_last: List of messages to summarize
        last_summary_text: Optional previous summary to provide context
        summary_type: Type of summary to create ("short" or "long")

    Returns:
        A full summary of the conversation up to the last message
    """

    try:
        response: llm.CallResponse
        if summary_type == SummaryType.SHORT:
            response = await create_short_summary(messages, previous_summary_text)
        else:
            response = await create_long_summary(messages, previous_summary_text)

        summary_text = response.content
        summary_tokens = (
            response.usage.output_tokens
            if response.usage
            else len(response.content) // 4
        )
    except Exception:
        logger.exception("Error generating summary!")
        # Fallback to a basic summary in case of error
        summary_text = (
            f"Conversation with {len(messages)} messages about {messages[-1].content[:30]}..."
            if messages
            else ""
        )
        summary_tokens = 50

    return Summary(
        content=summary_text,
        message_id=messages[-1].id if messages else 0,
        summary_type=summary_type.value,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        token_count=summary_tokens,
    )


async def _save_summary(
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
        "Saved %s for session %s covering up to message %s",
        summary["summary_type"],
        session_name,
        summary["message_id"],
    )


async def get_summarized_history(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
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
        cutoff: (Optional) message ID to cutoff at
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A string formatted history text with summary and recent messages
    """
    # Get messages since the latest summary and the summary itself
    summary = await get_summary(db, workspace_name, session_name, summary_type)

    # Check if we have a valid summary with a message_id
    if summary:
        messages = await crud.get_messages_id_range(
            db,
            workspace_name,
            session_name,
            start_id=summary["message_id"],
            end_id=cutoff,
        )
    else:
        messages = await crud.get_messages_id_range(
            db, workspace_name, session_name, end_id=cutoff
        )

    # Format messages
    messages_text = _format_messages(messages)

    if summary:
        # Combine summary with recent messages
        return f"""
<summary>
{summary["content"]}
</summary>
<recent_messages>
{messages_text}
</recent_messages>
"""

    # No summary available, return just the messages
    return messages_text


async def get_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    summary_type: SummaryType = SummaryType.SHORT,
) -> Summary | None:
    """
    Get summary for a given session.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        summary_type: Type of summary to retrieve ("short" or "long")

    Returns:
        The summary data dictionary, or None if no summary exists
    """
    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None

    summaries: dict[str, Summary] = session.internal_metadata.get("summaries", {})
    if not summaries or summary_type.value not in summaries:
        return None
    return summaries[summary_type.value]


async def get_both_summaries(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
) -> tuple[Summary | None, Summary | None]:
    """
    Get both short and long summaries for a given session.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name

    Returns:
        A tuple of the short and long summaries, or None if no summary exists
    """
    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None, None

    summaries: dict[str, Summary] = session.internal_metadata.get("summaries", {})
    return summaries.get(SummaryType.SHORT.value), summaries.get(SummaryType.LONG.value)


def _format_messages(messages: list[models.Message]) -> str:
    """
    Format a list of messages into a string by concatenating their content and
    prefixing each with the peer name.
    """
    if len(messages) == 0:
        return ""
    return "\n".join([f"{msg.peer_name}: {msg.content}" for msg in messages])

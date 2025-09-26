import asyncio
import logging
import time
from enum import Enum
from inspect import cleandoc as c

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypedDict

from src import schemas
from src.config import settings
from src.dependencies import tracked_db
from src.exceptions import ResourceNotFoundException
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.formatting import utc_now_iso
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


def to_schema_summary(s: Summary) -> schemas.Summary:
    return schemas.Summary(
        content=s["content"],
        message_id=s["message_id"],
        summary_type=s["summary_type"],
        created_at=s["created_at"],
        token_count=s["token_count"],
    )


# Export the public functions
__all__ = [
    "get_summary",
    "get_both_summaries",
    "get_summarized_history",
    "get_session_context",
    "get_session_context_formatted",
    "SummaryType",
    "Summary",
    "to_schema_summary",
]


# Configuration constants for summaries
MESSAGES_PER_SHORT_SUMMARY = settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
MESSAGES_PER_LONG_SUMMARY = settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY

SUMMARIES_KEY = "summaries"


# The types of summary to store in the session metadata
class SummaryType(Enum):
    SHORT = "honcho_chat_summary_short"
    LONG = "honcho_chat_summary_long"


async def create_short_summary(
    messages: list[models.Message],
    input_tokens: int,
    previous_summary: str | None = None,
) -> HonchoLLMCallResponse[str]:
    # input_tokens indicates how many tokens the message list + previous summary take up
    # we want to optimize short summaries to be smaller than the actual content being summarized
    # so we ask the agent to produce a word count roughly equal to either the input, or the max
    # size if the input is larger. the word/token ratio is roughly 4:3 so we multiply by 0.75.
    # LLMs *seem* to respond better to getting asked for a word count but should workshop this.
    output_words = int(min(input_tokens, settings.SUMMARY.MAX_TOKENS_SHORT) * 0.75)

    if previous_summary:
        previous_summary_text = previous_summary
    else:
        previous_summary_text = "There is no previous summary -- the messages are the beginning of the conversation."

    prompt = c(f"""
You are a system that summarizes parts of a conversation to create a concise and accurate summary. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a concise, factual summary that captures the essence of the conversation. Your summary should be detailed enough to serve as context for future messages, but brief enough to be helpful. Prefer a thorough chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{_format_messages(messages)}
</conversation>

Produce as thorough a summary as possible in {output_words} words or less.
""")

    return await honcho_llm_call(
        provider=settings.SUMMARY.PROVIDER,
        model=settings.SUMMARY.MODEL,
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_SHORT,
    )


async def create_long_summary(
    messages: list[models.Message],
    previous_summary: str | None = None,
) -> HonchoLLMCallResponse[str]:
    # the word/token ratio is roughly 4:3 so we multiply by 0.75.
    # LLMs *seem* to respond better to getting asked for a word count but should workshop this.
    output_words = int(settings.SUMMARY.MAX_TOKENS_LONG * 0.75)

    if previous_summary:
        previous_summary_text = previous_summary
    else:
        previous_summary_text = "There is no previous summary -- the messages are the beginning of the conversation."

    prompt = c(f"""
You are a system that creates thorough, comprehensive summaries of conversations. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed in detail
5. User's apparent emotional state and personality traits
6. Important themes and patterns across the conversation

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

Provide a thorough and detailed summary that captures the essence of the conversation. Your summary should serve as a comprehensive record of the important information in this conversation. Prefer an exhaustive chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{_format_messages(messages)}
</conversation>

Produce as thorough a summary as possible in {output_words} words or less.
""")

    return await honcho_llm_call(
        provider=settings.SUMMARY.PROVIDER,
        model=settings.SUMMARY.MODEL,
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
    )


async def summarize_if_needed(
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
            async with tracked_db("create_long_summary") as db_session:
                await _create_and_save_summary(
                    db_session,
                    workspace_name,
                    session_name,
                    message_id,
                    SummaryType.LONG,
                )
                logger.info(
                    "Saved long summary for session %s covering up to message %s (%s in session)",
                    session_name,
                    message_id,
                    message_seq_in_session,
                )

        async def create_short_summary():
            async with tracked_db("create_short_summary") as db_session:
                await _create_and_save_summary(
                    db_session,
                    workspace_name,
                    session_name,
                    message_id,
                    SummaryType.SHORT,
                )
                logger.info(
                    "Saved short summary for session %s covering up to message %s (%s in session)",
                    session_name,
                    message_id,
                    message_seq_in_session,
                )

        await asyncio.gather(
            create_long_summary(),
            create_short_summary(),
            return_exceptions=True,
        )
    else:
        async with tracked_db("create_summary") as db:
            # If only one summary needs to be created, run them individually
            if should_create_long:
                await _create_and_save_summary(
                    db,
                    workspace_name,
                    session_name,
                    message_id,
                    SummaryType.LONG,
                )
                logger.info(
                    "Saved long summary for session %s covering up to message %s (%s in session)",
                    session_name,
                    message_id,
                    message_seq_in_session,
                )
            elif should_create_short:
                await _create_and_save_summary(
                    db,
                    workspace_name,
                    session_name,
                    message_id,
                    SummaryType.SHORT,
                )
                logger.info(
                    "Saved short summary for session %s covering up to message %s (%s in session)",
                    session_name,
                    message_id,
                    message_seq_in_session,
                )


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

    messages_tokens = sum([message.token_count for message in messages])
    previous_summary_tokens = latest_summary["token_count"] if latest_summary else 0
    input_tokens = messages_tokens + previous_summary_tokens

    new_summary = await _create_summary(
        messages=messages,
        previous_summary_text=previous_summary_text,
        summary_type=summary_type,
        input_tokens=input_tokens,
    )

    await _save_summary(
        db,
        new_summary,
        workspace_name,
        session_name,
    )

    summary_duration = (time.perf_counter() - summary_start) * 1000
    accumulate_metric(
        f"summary_{workspace_name}_{message_id}",
        f"{summary_type.name}_summary_creation",
        summary_duration,
        "ms",
    )


async def _create_summary(
    messages: list[models.Message],
    previous_summary_text: str | None,
    summary_type: SummaryType,
    input_tokens: int,
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

    response: HonchoLLMCallResponse[str] | None = None
    try:
        if summary_type == SummaryType.SHORT:
            response = await create_short_summary(
                messages, input_tokens, previous_summary_text
            )
        else:
            response = await create_long_summary(messages, previous_summary_text)

        summary_text = response.content
        summary_tokens = response.output_tokens

        # Detect potential issues with the summary
        if not summary_text.strip():
            logger.error(
                "Generated summary is empty! This may indicate a token limit issue."
            )

        logger.info("Summary text: %s", summary_text)
        logger.info("Summary size: %s tokens", summary_tokens)
    except Exception:
        logger.exception("Error generating summary!")
        # Fallback to a basic summary in case of error
        summary_text = (
            f"Conversation with {len(messages)} messages about {messages[-1].content[:30]}..."
            if messages
            else ""
        )
        summary_tokens = 50

    accumulate_metric(
        f"summary_{messages[-1].workspace_name}_{messages[-1].id}",
        f"{summary_type.name}_summary_size",
        response.output_tokens if response else f"{summary_tokens} (est.)",
        "tokens",
    )

    return Summary(
        content=summary_text,
        message_id=messages[-1].id if messages else 0,
        summary_type=summary_type.value,
        created_at=utc_now_iso(),
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
    existing_summaries = session.internal_metadata.get(SUMMARIES_KEY, {})
    existing_summaries[label_value] = summary
    update_data[SUMMARIES_KEY] = existing_summaries

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

    summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
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

    summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
    return summaries.get(SummaryType.SHORT.value), summaries.get(SummaryType.LONG.value)


async def get_session_context(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int,
    *,
    cutoff: int | None = None,
    include_summary: bool = True,
) -> tuple[schemas.Summary | None, list[models.Message]]:
    """
    Get session context similar to the API endpoint but for internal use.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        token_limit: Maximum tokens for the context
        cutoff: Optional message ID to stop at (exclusive)
        include_summary: Whether to include summary if available

    Returns:
        Tuple of (summary, messages) where summary is a Summary pydantic model (or None)
        and messages is the list of message objects
    """
    if token_limit <= 0:
        return None, []

    summary = None
    messages_tokens = token_limit
    messages_start_id = 0

    if include_summary:
        # Allocate 40% of tokens to summary, 60% to messages
        summary_tokens_limit = int(token_limit * 0.4)

        latest_short_summary, latest_long_summary = await get_both_summaries(
            db, workspace_name, session_name
        )

        long_len = latest_long_summary["token_count"] if latest_long_summary else 0
        short_len = latest_short_summary["token_count"] if latest_short_summary else 0

        # Return the longest summary that fits within the token limit
        if (
            latest_long_summary
            and long_len <= summary_tokens_limit
            and long_len > short_len
        ):
            summary = schemas.Summary(
                content=latest_long_summary["content"],
                message_id=latest_long_summary["message_id"],
                summary_type=latest_long_summary["summary_type"],
                created_at=latest_long_summary["created_at"],
                token_count=latest_long_summary["token_count"],
            )
            messages_tokens = token_limit - latest_long_summary["token_count"]
            messages_start_id = latest_long_summary["message_id"]
        elif (
            latest_short_summary and short_len <= summary_tokens_limit and short_len > 0
        ):
            summary = schemas.Summary(
                content=latest_short_summary["content"],
                message_id=latest_short_summary["message_id"],
                summary_type=latest_short_summary["summary_type"],
                created_at=latest_short_summary["created_at"],
                token_count=latest_short_summary["token_count"],
            )
            messages_tokens = token_limit - latest_short_summary["token_count"]
            messages_start_id = latest_short_summary["message_id"]
        else:
            logger.warning(
                "No summary available for get_context call with token limit %s, returning empty string. long_summary_len: %s, short_summary_len: %s",
                token_limit,
                long_len,
                short_len,
            )

    # Get recent messages after summary
    messages = await crud.get_messages_id_range(
        db,
        workspace_name,
        session_name,
        start_id=messages_start_id,
        end_id=cutoff,
        token_limit=messages_tokens,
    )

    return summary, messages


async def get_session_context_formatted(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int,
    *,
    cutoff: int | None = None,
    include_summary: bool = True,
) -> str:
    """
    Get formatted session context as a string for internal use (e.g., deriver).

    This is a convenience wrapper around get_session_context that formats
    the output as a string.
    """
    if token_limit <= 0:
        return ""

    summary, messages = await get_session_context(
        db,
        workspace_name,
        session_name,
        token_limit,
        cutoff=cutoff,
        include_summary=include_summary,
    )

    # Format the messages
    messages_text = _format_messages(messages)
    summary_content = summary.content if summary else ""

    if summary_content and messages_text:
        return f"""<summary>
{summary_content}
</summary>

<recent_messages>
{messages_text}
</recent_messages>"""
    elif summary_content:
        return f"""<summary>
{summary_content}
</summary>"""
    elif messages_text:
        return messages_text
    else:
        return ""


def _format_messages(messages: list[models.Message]) -> str:
    """
    Format a list of messages into a string by concatenating their content and
    prefixing each with the peer name.
    """
    if len(messages) == 0:
        return ""
    return "\n".join([f"{msg.peer_name}: {msg.content}" for msg in messages])

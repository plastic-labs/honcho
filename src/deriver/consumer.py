import logging
import os

import sentry_sdk
from langfuse.decorators import observe
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings

from .. import crud
from ..utils import history
from .tom.embeddings import CollectionEmbeddingStore
from .tom.long_term import extract_facts_long_term

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=False)

TOM_METHOD = settings.DERIVER.TOM_METHOD
USER_REPRESENTATION_METHOD = settings.DERIVER.USER_REPRESENTATION_METHOD


async def process_item(db: AsyncSession, payload: dict):
    logger.debug(
        f"process_item received payload: {payload['message_id']} is_user={payload['is_user']}"
    )
    processing_args = [
        payload["content"],
        payload["app_id"],
        payload["user_id"],
        payload["session_id"],
        payload["message_id"],
        db,
    ]
    if payload["is_user"]:
        logger.debug(f"Processing user message: {payload['message_id']}")
        await process_user_message(*processing_args)
    else:
        logger.debug(f"Processing AI message: {payload['message_id']}")
        await process_ai_message(*processing_args)
    logger.debug(f"Finished processing message: {payload['message_id']}")
    await summarize_if_needed(
        db,
        payload["app_id"],
        payload["session_id"],
        payload["user_id"],
        payload["message_id"],
    )
    return


@sentry_sdk.trace
# @observe()
async def process_ai_message(
    content: str,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    db: AsyncSession,
):
    """
    Process an AI message. Make a prediction about what the user is going to say to it.
    """
    console.print(f"Processing AI message: {content}", style="bright_magenta")


@sentry_sdk.trace
@observe()
async def process_user_message(
    content: str,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    db: AsyncSession,
):
    """
    Process a user message by extracting facts and saving them to the vector store.
    This runs as a background process after a user message is logged.
    """
    console.print(f"Processing User Message: {content}", style="orange1")
    process_start = os.times()[4]  # Get current CPU time
    logger.debug(f"Starting fact extraction for user message: {message_id}")

    # Get chat history and append current message
    logger.debug(f"Retrieving chat history for session: {session_id}")
    (
        short_history_text,
        short_history_messages,
        latest_short_summary,
    ) = await history.get_summarized_history(
        db, session_id, summary_type=history.SummaryType.SHORT
    )
    chat_history_str = f"{short_history_text}\nhuman: {content}"

    # Extract facts from chat history
    logger.debug("Extracting facts from chat history")
    extract_start = os.times()[4]
    facts = await extract_facts_long_term(chat_history_str)
    extract_time = os.times()[4] - extract_start
    console.print(f"Extracted Facts: {facts}", style="bright_blue")
    logger.debug(f"Extracted {len(facts)} facts in {extract_time:.2f}s")

    # Save the facts to the collection
    logger.debug(f"Setting up embedding store for app: {app_id}, user: {user_id}")
    collection = await crud.get_or_create_user_protected_collection(
        db=db, app_id=app_id, user_id=user_id
    )
    embedding_store = CollectionEmbeddingStore(
        app_id=app_id,
        user_id=user_id,
        collection_id=collection.public_id,  # type: ignore
    )

    # Filter out facts that are duplicates of existing facts in the vector store
    logger.debug("Removing duplicate facts")
    dedup_start = os.times()[4]
    unique_facts = await embedding_store.remove_duplicates(facts)
    dedup_time = os.times()[4] - dedup_start
    logger.debug(
        f"Found {len(unique_facts)}/{len(facts)} unique facts in {dedup_time:.2f}s"
    )

    # Only save the unique facts
    if unique_facts:
        logger.debug(f"Saving {len(unique_facts)} unique facts to vector store")
        save_start = os.times()[4]
        await embedding_store.save_facts(unique_facts, message_id=message_id)
        save_time = os.times()[4] - save_start
        logger.debug(f"Facts saved in {save_time:.2f}s")
    else:
        logger.debug("No unique facts to save")

    console.print(f"Saved {len(unique_facts)} unique facts", style="bright_green")

    total_time = os.times()[4] - process_start
    logger.debug(f"Total processing time: {total_time:.2f}s")


async def summarize_if_needed(
    db: AsyncSession, app_id: str, session_id: str, user_id: str, message_id: str
):
    summary_start = os.times()[4]
    logger.debug("Checking if summaries should be created")

    # STEP 1: First check if we need a short summary (every 10 messages)
    (
        should_create_short,
        short_messages,
        latest_short_summary,
    ) = await history.should_create_summary(
        db, session_id, summary_type=history.SummaryType.SHORT
    )

    if should_create_short:
        logger.debug(f"Short summary needed for {len(short_messages)} messages")

        # STEP 2: If we need a short summary, check if we also need a long summary
        (
            should_create_long,
            long_messages,
            latest_long_summary,
        ) = await history.should_create_summary(
            db, session_id, summary_type=history.SummaryType.LONG
        )

        # STEP 3: If we need a long summary, create it first before creating the short summary
        if should_create_long:
            logger.debug(
                f"Creating new long summary covering {len(long_messages)} messages"
            )
            try:
                # Get previous long summary context if available
                previous_long_summary = (
                    latest_long_summary.content if latest_long_summary else None
                )

                # Create a new long summary
                long_summary_text = await history.create_summary(
                    messages=long_messages,
                    previous_summary=previous_long_summary,
                    summary_type=history.SummaryType.LONG,
                )
                # Save the long summary as a metamessage and capture the returned object
                latest_long_summary = await history.save_summary_metamessage(
                    db=db,
                    app_id=app_id,
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id,
                    summary_content=long_summary_text,
                    message_count=len(long_messages),
                    summary_type=history.SummaryType.LONG,
                )
                logger.debug("Long summary created and saved successfully")
            except Exception as e:
                logger.error(f"Error creating long summary: {str(e)}")
        else:
            logger.debug(
                f"No long summary needed. Need {history.MESSAGES_PER_LONG_SUMMARY} messages since last long summary."
            )

        # STEP 4: Now create the short summary, using the latest long summary for context if available
        logger.debug(
            f"Creating new short summary covering {len(short_messages)} messages"
        )
        try:
            previous_summary = (
                latest_long_summary.content if latest_long_summary else None
            )
            # Create a new short summary
            short_summary_text = await history.create_summary(
                messages=short_messages,
                previous_summary=previous_summary,
                summary_type=history.SummaryType.SHORT,
            )
            # Save the short summary as a metamessage
            await history.save_summary_metamessage(
                db=db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                summary_content=short_summary_text,
                message_count=len(short_messages),
                summary_type=history.SummaryType.SHORT,
            )
            logger.debug("Short summary created and saved successfully")
        except Exception as e:
            logger.error(f"Error creating short summary: {str(e)}")
    else:
        logger.debug(
            f"No short summary needed. Need {history.MESSAGES_PER_SHORT_SUMMARY} messages since last short summary."
        )

    summary_time = os.times()[4] - summary_start
    logger.debug(f"Summary check completed in {summary_time:.2f}s")

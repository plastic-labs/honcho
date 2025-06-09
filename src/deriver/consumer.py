import datetime
import json
import logging
import os

import sentry_sdk
from langfuse.decorators import observe
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud, schemas
from ..utils import history
from .fact_saver import (
    get_observation_saver_queue,
    initialize_observation_saver,
    shutdown_observation_saver,
)
from .surprise_reasoner import SurpriseReasoner
from .tom.embeddings import CollectionEmbeddingStore

# from .tom.long_term import extract_facts_long_term

# Removed dataset-specific datetime configuration - now using real-time only

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True
logger.info("Deriver using real-time datetime")

TOM_METHOD = os.getenv("TOM_METHOD", "single_prompt")
USER_REPRESENTATION_METHOD = os.getenv("USER_REPRESENTATION_METHOD", "long_term")


# FIXME see if this is SAFE
# async def add_metamessage(db, message_id, metamessage_type, content):
# metamessage = models.Metamessage(
#     message_id=message_id,
#     metamessage_type=metamessage_type,
#     content=content,
#     h_metadata={},
# )
# db.add(metamessage)


def get_current_datetime() -> str:
    """Get current datetime in UTC format."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def save_deriver_trace(
    db: AsyncSession,
    message_id: str,
    deriver_trace: dict,
    app_id: str,
    user_id: str,
    session_id: str,
):
    """Save the deriver trace as a metamessage attached to the user message."""
    try:
        # Convert trace to JSON string
        trace_content = json.dumps(deriver_trace, indent=2)

        # Create metamessage schema
        metamessage_data = schemas.MetamessageCreate(
            message_id=message_id,
            session_id=session_id,
            label="deriver_trace",
            content=trace_content,
            metadata={
                "trace_version": "1.0",
                "timestamp": deriver_trace.get("timestamp"),
            },
        )

        # Save to database
        await crud.create_metamessage(db, user_id, metamessage_data, app_id)
        logger.info(f"Saved deriver trace for message {message_id}")

    except Exception as e:
        logger.error(f"Failed to save deriver trace for message {message_id}: {e}")
        # Don't raise - trace saving should not block reasoning process


async def get_session_datetime(db: AsyncSession, session_id: str) -> str:
    """
    Get current datetime - simplified for real-time usage.

    Returns:
        Current datetime string 'YYYY-MM-DD HH:MM:SS'
    """
    return get_current_datetime()


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
    logger.debug(f"Processing AI message: {content}")


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
    logger.debug(f"Processing user message: {content}")
    process_start = os.times()[4]  # Get current CPU time
    logger.debug(f"Starting fact extraction for user message: {message_id}")

    # Initialize fact saver queue if not already running
    observation_saver = get_observation_saver_queue()
    if not observation_saver.is_running:
        await initialize_observation_saver(db)

    # Get session datetime for the "new turn" timestamp (matches your dataset)
    session_datetime = await get_session_datetime(db, session_id)
    logger.info(
        f"Session datetime result: '{session_datetime}' for session {session_id}"
    )

    # Create summary if needed BEFORE history retrieval to ensure consistent state
    await summarize_if_needed(db, app_id, session_id, user_id, message_id)

    # Get chat history UP TO but NOT INCLUDING the current message being processed
    (
        short_history_text,
        short_history_messages,
        latest_short_summary,
    ) = await history.get_summarized_history_before_message(
        db,
        session_id,
        message_id,
        summary_type=history.SummaryType.SHORT,
        fallback_to_created_at=True,
    )

    # Debug: Check if we just created a summary and messages are missing
    logger.info(f"History retrieved: {len(short_history_messages)} recent messages")
    if latest_short_summary:
        logger.info(
            f"Latest summary exists, created for message: {latest_short_summary.message_id}"
        )
    else:
        logger.info("No summary exists yet")

    # Use the properly formatted history that includes summary context
    formatted_history = short_history_text

    # instantiate embedding store from collection
    collection = await crud.get_or_create_user_protected_collection(
        db=db, app_id=app_id, user_id=user_id
    )
    embedding_store = CollectionEmbeddingStore(
        db=db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection.public_id,  # type: ignore
    )

    # Configure observation counts to keep context focused on most relevant information
    # These values balance having enough context while avoiding information overload
    embedding_store.set_observation_counts(
        abductive=2,  # Keep only the most relevant high-level insights
        inductive=4,  # Limit behavioral patterns to the most relevant ones
        deductive=6,  # Allow more specific observations but keep them manageable
    )

    # Retrieve semantically relevant observations with session context for the current message
    initial_context = (
        await embedding_store.get_relevant_observations_for_reasoning_with_context(
            current_message=content, conversation_context=formatted_history
        )
    )

    # Create reasoner instance
    reasoner = SurpriseReasoner(embedding_store=embedding_store)

    # Run consolidated reasoning that handles all three levels (deductive, inductive, abductive)
    logger.debug(
        "REASONING: Running unified insight derivation across all reasoning levels"
    )

    # The unified approach naturally handles both reactive reasoning (responding to new turns)
    # and proactive reasoning (generating abductive hypotheses when needed)
    final_observations, deriver_trace = await reasoner.recursive_reason_with_trace(
        context=initial_context,
        history=formatted_history,
        new_turn=content,
        message_id=message_id,
        session_id=session_id,
        current_time=session_datetime,
    )

    logger.debug("REASONING COMPLETION: Unified reasoning completed across all levels.")
    logger.info(f"Final observations:\n{json.dumps(final_observations, indent=2)}")

    # Save the deriver trace as a metamessage attached to the user message
    await save_deriver_trace(db, message_id, deriver_trace, app_id, user_id, session_id)

    # Log queue stats for monitoring
    stats = observation_saver.get_stats()
    logger.debug(f"Observation saver queue stats: {stats}")

    rsr_time = os.times()[4] - process_start
    logger.debug(f"Parallel reasoning completed in {rsr_time:.2f}s")


async def cleanup_deriver():
    """Cleanup function to gracefully shutdown deriver components."""
    await shutdown_observation_saver()
    logger.info("Deriver cleanup completed")


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
        logger.info(
            f"Creating short summary for {len(short_messages)} messages (current message: {message_id})"
        )
        if short_messages:
            logger.info(
                f"Summary will include messages from {short_messages[0].public_id} to {short_messages[-1].public_id}"
            )

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
                # Use the last message in the summarized messages, not the current message being processed
                last_summarized_message_id = (
                    long_messages[-1].public_id if long_messages else message_id
                )
                latest_long_summary = await history.save_summary_metamessage(
                    db=db,
                    app_id=app_id,
                    user_id=user_id,
                    session_id=session_id,
                    message_id=last_summarized_message_id,
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
            # Use the last message in the summarized messages, not the current message being processed
            last_summarized_message_id = (
                short_messages[-1].public_id if short_messages else message_id
            )
            await history.save_summary_metamessage(
                db=db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                message_id=last_summarized_message_id,
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

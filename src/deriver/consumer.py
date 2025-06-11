import datetime
import json
import logging
import os

import sentry_sdk
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud, schemas
from ..utils import history
from ..utils.logging import log_observations_tree, log_performance_metrics
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
# TODO: Re-enable when Mirascope-Langfuse compatibility issue is fixed  
# @observe()
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

    # Get current datetime for timestamping new observations
    current_datetime = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

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

    # Extract facts from chat history
    # logger.debug("Extracting facts from chat history")
    # extract_start = os.times()[4]
    # fact_extraction = await extract_facts_long_term(chat_history_str)
    # facts: list[str] = fact_extraction.facts or []
    # extract_time = os.times()[4] - extract_start
    # console.print(f"Extracted Facts: {fact_extraction.facts}", style="bright_blue")
    # logger.debug(f"Extracted {len(facts)} facts in {extract_time:.2f}s")
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

    # Convert ObservationContext to ReasoningResponse for compatibility
    initial_reasoning_context = reasoner._observation_context_to_reasoning_response(
        initial_context
    )

    # The unified approach naturally handles both reactive reasoning (responding to new turns)
    # and proactive reasoning (generating abductive hypotheses when needed)
    final_observations, deriver_trace = await reasoner.recursive_reason_with_trace(
        db=db,
        context=initial_reasoning_context,
        history=formatted_history,
        new_turn=content,
        message_id=message_id,
        session_id=session_id,
        current_time=current_datetime,
    )

    logger.debug("REASONING COMPLETION: Unified reasoning completed across all levels.")
    
    # Display final observations in a beautiful tree
    from src.utils.deriver import REASONING_LEVELS
    final_obs_dict = {
        level: getattr(final_observations, level, [])
        for level in REASONING_LEVELS
    }
    log_observations_tree(final_obs_dict, "🎯 FINAL OBSERVATIONS")
    
    # Display final reasoning metrics
    rsr_time = os.times()[4] - process_start
    total_observations = sum(len(obs_list) for obs_list in final_obs_dict.values())
    summary_metrics = {
        "total_processing_time": rsr_time * 1000,  # Convert to ms
        "total_iterations": deriver_trace.get("summary", {}).get("total_iterations", 0),
        "final_observation_count": total_observations,
        "reasoning_convergence": deriver_trace.get("summary", {}).get("convergence_reason", "unknown"),
    }
    log_performance_metrics(summary_metrics, "🏁 REASONING SUMMARY")

    # Save the deriver trace as a metamessage attached to the user message
    await save_deriver_trace(db, message_id, deriver_trace, app_id, user_id, session_id)


async def cleanup_deriver():
    """Cleanup function to gracefully shutdown deriver components."""
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
                long_summary_response = await history.create_summary(
                    messages=long_messages,
                    previous_summary=previous_long_summary,
                    summary_type=history.SummaryType.LONG,
                )
                long_summary_text = str(long_summary_response.content)
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
            short_summary_response = await history.create_summary(
                messages=short_messages,
                previous_summary=previous_summary,
                summary_type=history.SummaryType.SHORT,
            )
            short_summary_text = str(short_summary_response.content)
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

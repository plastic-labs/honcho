import logging
import os

import sentry_sdk
from langfuse.decorators import observe
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud
from ..utils import history
from .tom.embeddings import CollectionEmbeddingStore
from .tom.long_term import extract_facts_long_term

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

# Add a handler with DEBUG level for this specific logger
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Prevent propagation to avoid duplicate messages if root logger also shows DEBUG
    logger.propagate = False


console = Console(markup=False)

TOM_METHOD = os.getenv("TOM_METHOD", "single_prompt")
USER_REPRESENTATION_METHOD = os.getenv("USER_REPRESENTATION_METHOD", "long_term")


async def process_item(db: AsyncSession, payload: dict):
    logger.debug(
        "process_item received payload for message %s in session %s",
        payload["message_id"],
        payload["session_name"],
    )
    # TODO: validate these strings and whatnot
    processing_args = [
        payload["content"],
        payload["workspace_name"],
        payload["peer_name"],
        payload["session_name"],
        payload["message_id"],
        db,
    ]
    if payload["task_type"] == "representation":
        logger.debug(
            "Processing message %s in %s",
            payload["message_id"],
            payload["session_name"],
        )
        await process_message(*processing_args)
        logger.debug(
            "Finished processing message %s in %s %s",
            payload["message_id"],
            "session" if payload["session_name"] else "peer",
            payload["session_name"]
            if payload["session_name"]
            else payload["peer_name"],
        )
    await summarize_if_needed(
        db,
        payload["workspace_name"],
        payload["session_name"],
        payload["peer_name"],
        payload["message_id"],
    )
    return


@sentry_sdk.trace
@observe()
async def process_message(
    content: str,
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
    message_id: int,
    db: AsyncSession,
):
    """
    Process a user message by extracting facts and saving them to the vector store.
    This runs as a background process after a user message is logged.
    """
    console.print(f"Processing User Message: {content}", style="orange1")
    process_start = os.times()[4]  # Get current CPU time
    logger.debug(
        "Starting fact extraction for user message %s in %s %s",
        message_id,
        "session" if session_name else "peer",
        session_name if session_name else peer_name,
    )

    if session_name:
        # Get chat history and append current message
        logger.debug(
            "Retrieving chat history for %s %s",
            "session" if session_name else "peer",
            session_name if session_name else peer_name,
        )
        short_history_text = await history.get_summarized_history(
            db,
            workspace_name,
            session_name,
            peer_name,
            cutoff=message_id,
            summary_type=history.SummaryType.SHORT,
        )

        chat_history_str = f"{short_history_text}\nuser: {content}"
    else:
        chat_history_str = f"user: {content}"

    # Extract facts from chat history
    logger.debug("Extracting facts from chat history")
    extract_start = os.times()[4]
    facts = await extract_facts_long_term(chat_history_str)
    extract_time = os.times()[4] - extract_start
    console.print(f"Extracted Facts: {facts}", style="bright_blue")
    logger.debug(f"Extracted {len(facts)} facts in {extract_time:.2f}s")

    # Save the facts to the collection
    logger.debug(
        f"Setting up embedding store for workspace: {workspace_name}, peer: {peer_name}"
    )
    collection = await crud.get_or_create_peer_protected_collection(
        db, workspace_name, peer_name
    )
    embedding_store = CollectionEmbeddingStore(
        workspace_name=workspace_name,
        peer_name=peer_name,
        collection_name=collection.name,
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
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    peer_name: str,
    message_id: int,
):
    if not session_name:
        return

    summary_start = os.times()[4]
    logger.debug("Checking if summaries should be created for session %s", session_name)

    # STEP 1: First check if we need a short summary (every 10 messages)
    (
        should_create_short,
        short_messages,
        _,
    ) = await history.should_create_summary(
        db,
        workspace_name,
        session_name,
        peer_name,
        message_id,
        summary_type=history.SummaryType.SHORT,
    )

    if should_create_short:
        logger.debug(f"Short summary needed for {len(short_messages)} messages")

        # STEP 2: If we need a short summary, check if we also need a long summary
        (
            should_create_long,
            long_messages,
            latest_long_summary,
        ) = await history.should_create_summary(
            db,
            workspace_name,
            session_name,
            peer_name,
            message_id,
            summary_type=history.SummaryType.LONG,
        )

        # STEP 3: If we need a long summary, create it first before creating the short summary
        if should_create_long:
            logger.debug(
                f"Creating new long summary covering {len(long_messages)} messages"
            )
            try:
                # Get previous long summary context if available
                previous_long_summary_text = (
                    latest_long_summary["content"] if latest_long_summary else None
                )

                # Create a new long summary
                new_long_summary = await history.create_summary(
                    messages=long_messages,
                    previous_summary_text=previous_long_summary_text,
                    summary_type=history.SummaryType.LONG,
                )

                # Save the long summary
                await history.save_summary(
                    db,
                    new_long_summary,
                    workspace_name,
                    session_name,
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
            previous_long_summary_text = (
                latest_long_summary["content"] if latest_long_summary else None
            )

            # Create a new short summary
            new_short_summary = await history.create_summary(
                messages=short_messages,
                previous_summary_text=previous_long_summary_text,
                summary_type=history.SummaryType.SHORT,
            )

            # Save the short summary
            await history.save_summary(
                db,
                new_short_summary,
                workspace_name,
                session_name,
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

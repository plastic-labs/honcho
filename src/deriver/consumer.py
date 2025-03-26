import logging
import os

import sentry_sdk
from langfuse.decorators import observe
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud, models
from .tom.embeddings import CollectionEmbeddingStore
from .tom.long_term import extract_facts_long_term

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=False)

TOM_METHOD = os.getenv("TOM_METHOD", "single_prompt")
USER_REPRESENTATION_METHOD = os.getenv("USER_REPRESENTATION_METHOD", "long_term")


# FIXME see if this is SAFE
async def add_metamessage(db, message_id, metamessage_type, content):
    metamessage = models.Metamessage(
        message_id=message_id,
        metamessage_type=metamessage_type,
        content=content,
        h_metadata={},
    )
    db.add(metamessage)


async def get_chat_history(db, session_id, message_id, limit: int = 10) -> str:
    subquery = (
        select(models.Message.id)
        .where(models.Message.public_id == message_id)
        .scalar_subquery()
    )
    messages_stmt = (
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.id.desc())
        .where(models.Message.id < subquery)
        .limit(limit)
    )

    result = await db.execute(messages_stmt)
    messages = result.scalars().all()[::-1]

    chat_history_str = "\n".join(
        [f"human: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
    )
    return chat_history_str


async def process_item(db: AsyncSession, payload: dict):
    logger.debug(f"process_item received payload: {payload['message_id']} is_user={payload['is_user']}")
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
    return


@sentry_sdk.trace
@observe()
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
    chat_history_str = await get_chat_history(db, session_id, message_id)
    chat_history_str = f"{chat_history_str}\nhuman: {content}"

    # Extract facts from chat history
    logger.debug("Extracting facts from chat history")
    extract_start = os.times()[4]
    facts = await extract_facts_long_term(chat_history_str)
    extract_time = os.times()[4] - extract_start
    console.print(f"Extracted Facts: {facts}", style="bright_blue")
    logger.debug(f"Extracted {len(facts)} facts in {extract_time:.2f}s")
    
    # Save the facts to the collection
    logger.debug(f"Setting up embedding store for app: {app_id}, user: {user_id}")
    collection = await crud.get_collection_by_name(db, app_id, user_id, "honcho")
    embedding_store = CollectionEmbeddingStore(
        db=db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection.public_id # type: ignore
    )
    
    # Filter out facts that are duplicates of existing facts in the vector store
    logger.debug("Removing duplicate facts")
    dedup_start = os.times()[4]
    unique_facts = await embedding_store.remove_duplicates(facts)
    dedup_time = os.times()[4] - dedup_start
    logger.debug(f"Found {len(unique_facts)}/{len(facts)} unique facts in {dedup_time:.2f}s")
    
    # Only save the unique facts
    if unique_facts:
        logger.debug(f"Saving {len(unique_facts)} unique facts to vector store")
        save_start = os.times()[4]
        await embedding_store.save_facts(unique_facts)
        save_time = os.times()[4] - save_start
        logger.debug(f"Facts saved in {save_time:.2f}s")
    else:
        logger.debug("No unique facts to save")
    
    console.print(f"Saved {len(unique_facts)} unique facts", style="bright_green")
    
    total_time = os.times()[4] - process_start
    logger.debug(f"Total processing time: {total_time:.2f}s")

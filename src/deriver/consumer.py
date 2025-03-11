import logging
import os
import re

import sentry_sdk
from langfuse.decorators import langfuse_context, observe
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, crud
from .tom import get_tom_inference, get_user_representation
from .tom.long_term import extract_facts_long_term
from .tom.embeddings import CollectionEmbeddingStore


logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=False)

TOM_METHOD = os.getenv("TOM_METHOD", "single_prompt")
USER_REPRESENTATION_METHOD = os.getenv("USER_REPRESENTATION_METHOD", "long_term")


def parse_xml_content(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags in a string.
    
    Args:
        text: The text containing XML-like tags
        tag: The tag name to extract content from
        
    Returns:
        The content between the opening and closing tags, or an empty string if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


# FIXME see if this is SAFE
async def add_metamessage(db, message_id, metamessage_type, content):
    metamessage = models.Metamessage(
        message_id=message_id,
        metamessage_type=metamessage_type,
        content=content,
        h_metadata={},
    )
    db.add(metamessage)


async def get_chat_history(db, session_id, message_id) -> str:
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
        .limit(10)
    )

    result = await db.execute(messages_stmt)
    messages = result.scalars().all()[::-1]

    chat_history_str = "\n".join(
        [f"human: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
    )
    return chat_history_str


async def process_item(db: AsyncSession, payload: dict):
    processing_args = [
        payload["content"],
        payload["app_id"],
        payload["user_id"],
        payload["session_id"],
        payload["message_id"],
        db,
    ]
    if payload["is_user"]:
        await process_user_message(*processing_args)
    else:
        await process_ai_message(*processing_args)
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

    # Get chat history and append current message
    chat_history_str = await get_chat_history(db, session_id, message_id)
    chat_history_str = f"{chat_history_str}\nhuman: {content}"

    # Extract facts from chat history
    facts = await extract_facts_long_term(chat_history_str)
    console.print(f"Extracted Facts: {facts}", style="bright_blue")
    
    # Save the facts to the collection
    collection = await crud.get_collection_by_name(db, app_id, user_id, "honcho")
    embedding_store = CollectionEmbeddingStore(
        db=db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection.public_id # type: ignore
    )
    
    # Filter out facts that are duplicates of existing facts in the vector store
    unique_facts = await embedding_store.remove_duplicates(facts)
    # Only save the unique facts
    await embedding_store.save_facts(unique_facts)
    console.print(f"Saved {len(unique_facts)} unique facts", style="bright_green")

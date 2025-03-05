import logging
import os
import re

import sentry_sdk
from langfuse.decorators import langfuse_context, observe
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models
from ..exceptions import ResourceNotFoundException, ValidationException
from .tom import get_tom_inference, get_user_representation

# Configure logging
logger = logging.getLogger(__name__)

# Turn off SQLAlchemy Echo logging
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=False)

TOM_METHOD = os.getenv("TOM_METHOD", "single_prompt")
USER_REPRESENTATION_METHOD = os.getenv("USER_REPRESENTATION_METHOD", "single_prompt")


# FIXME see if this is SAFE
async def add_metamessage(db, message_id, metamessage_type, content):
    metamessage = models.Metamessage(
        message_id=message_id,
        metamessage_type=metamessage_type,
        content=content,
        h_metadata={},
    )
    db.add(metamessage)


def parse_xml_content(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


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
    """
    Process a queue item based on whether it's a user or AI message.
    
    Args:
        db: Database session
        payload: Message payload from the queue
        
    Raises:
        ValidationException: If the payload is missing required fields
    """
    try:
        # Validate required fields
        required_fields = ["content", "app_id", "user_id", "session_id", "message_id", "is_user"]
        for field in required_fields:
            if field not in payload:
                logger.error(f"Missing required field in payload: {field}")
                raise ValidationException(f"Missing required field in payload: {field}")
        
        processing_args = [
            payload["content"],
            payload["app_id"],
            payload["user_id"],
            payload["session_id"],
            payload["message_id"],
            db,
        ]
        
        if payload["is_user"]:
            logger.info(f"Processing user message: {payload['message_id']}")
            await process_user_message(*processing_args)
        else:
            logger.info(f"Processing AI message: {payload['message_id']}")
            await process_ai_message(*processing_args)
            
    except Exception as e:
        logger.error(f"Error processing message {payload.get('message_id', 'unknown')}: {str(e)}")
        if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
            sentry_sdk.capture_exception(e)
        raise


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
    Process a user message by:
    - Getting TOM inference
    - Getting user representation
    """
    console.print(f"Processing User Message: {content}", style="orange1")

    # Get chat history and append current message
    chat_history_str = await get_chat_history(db, session_id, message_id)
    chat_history_str = f"{chat_history_str}\nhuman: {content}"

    # Get TOM inference, parse and save it
    tom_inference_response = await get_tom_inference(
        chat_history_str, session_id, method=TOM_METHOD
    )
    tom_inference = parse_xml_content(tom_inference_response, "prediction")
    await add_metamessage(
        db,
        message_id,
        "tom_inference",
        tom_inference,
    )
    await db.commit()

    # Fetch the latest user representation
    user_representation_stmt = (
        select(models.Metamessage)
        .join(
            models.Message,
            models.Message.public_id == models.Metamessage.message_id,
        )
        .join(
            models.Session,
            models.Message.session_id == models.Session.public_id,
        )
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Metamessage.metamessage_type == "user_representation")
        .order_by(models.Metamessage.id.desc())  # get the most recent
        .limit(1)
    )

    response = await db.execute(user_representation_stmt)
    existing_representation = response.scalar_one_or_none()

    existing_representation_content = (
        existing_representation.content if existing_representation else "None"
    )
    logger.info(f"User {user_id}: Existing Representation retrieved")
    logger.debug(f"User {user_id}: Existing Representation: {existing_representation_content}")

    langfuse_context.update_current_trace(
        session_id=session_id,
        user_id=user_id,
        release=os.getenv("SENTRY_RELEASE"),
        metadata={"environment": os.getenv("SENTRY_ENVIRONMENT")},
    )

    # Call user_representation
    user_representation_response = await get_user_representation(
        chat_history=chat_history_str,
        session_id=session_id,
        user_representation=existing_representation_content,
        tom_inference=tom_inference,
        method=USER_REPRESENTATION_METHOD,
    )

    # parse the user_representation response
    user_representation_response = parse_xml_content(
        user_representation_response, "representation"
    )

    # Store the user_representation response as a metamessage
    await add_metamessage(
        db,
        message_id,
        "user_representation",
        user_representation_response,
    )
    await db.commit()

    console.print(
        f"User Representation:\n{user_representation_response}",
        style="bright_green",
    )

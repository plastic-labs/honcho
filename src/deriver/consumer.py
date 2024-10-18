import logging
import re

from rich import print as rprint
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models
from .voe import tom_inference, user_representation

# Turn off SQLAlchemy Echo logging
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


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
    rprint(f"[green]Processing AI message: {content}[/green]")

    subquery = (
        select(models.Message.created_at)
        .where(models.Message.public_id == message_id)
        .scalar_subquery()
    )
    messages_stmt = (
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.desc())
        .where(models.Message.created_at < subquery)
        .limit(10)
    )

    result = await db.execute(messages_stmt)
    messages = result.scalars().all()[::-1]

    chat_history_str = "\n".join(
        [f"human: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
    )
    # append current message to chat history
    chat_history_str = f"{chat_history_str}\nai: {content}"

    tom_inference_response = await tom_inference(
        chat_history_str, session_id=session_id
    )

    prediction = parse_xml_content(tom_inference_response, "prediction")

    await add_metamessage(
        db,
        message_id,
        "tom_inference",
        prediction,
    )

    await db.commit()

    rprint("[blue]Tom Inference:")
    content_lines = str(prediction)
    rprint(f"[blue]{content_lines}")


async def process_user_message(
    content: str,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    db: AsyncSession,
):
    """
    Process a user message. If there are revised user predictions to run VoE against, run it. Otherwise pass.
    """
    rprint(f"[orange1]Processing User Message: {content}")
    subquery = (
        select(models.Message.created_at)
        .where(models.Message.public_id == message_id)
        .scalar_subquery()
    )

    messages_stmt = (
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .where(models.Message.is_user == False)
        .order_by(models.Message.created_at.desc())
        .where(models.Message.created_at < subquery)
        .limit(1)
    )

    response = await db.execute(messages_stmt)
    ai_message = response.scalar_one_or_none()

    if ai_message and ai_message.content:
        rprint(f"[orange1]AI Message: {ai_message.content}")

        # Fetch the tom_inference metamessage
        tom_inference_stmt = (
            select(models.Metamessage)
            .where(models.Metamessage.message_id == ai_message.public_id)
            .where(models.Metamessage.metamessage_type == "tom_inference")
            .order_by(models.Metamessage.created_at.asc())
            .limit(1)
        )
        response = await db.execute(tom_inference_stmt)
        tom_inference_metamessage = response.scalar_one_or_none()

        if tom_inference_metamessage and tom_inference_metamessage.content:
            rprint(f"[orange1]Tom Inference: {tom_inference_metamessage.content}")

            # Fetch the existing user representation
            user_representation_stmt = (
                select(models.Metamessage)
                .where(models.Metamessage.message_id == ai_message.public_id)
                .where(models.Metamessage.metamessage_type == "user_representation")
                .order_by(models.Metamessage.created_at.desc())
                .limit(1)
            )
            response = await db.execute(user_representation_stmt)
            existing_representation = response.scalar_one_or_none()

            existing_representation_content = (
                existing_representation.content if existing_representation else "None"
            )

            # Call user_representation
            user_representation_response = await user_representation(
                chat_history=f"{ai_message.content}\nhuman: {content}",
                session_id=session_id,
                user_representation=existing_representation_content,
                tom_inference=tom_inference_metamessage.content,
            )

            # Store the user_representation response as a metamessage
            await add_metamessage(
                db,
                message_id,
                "user_representation",
                user_representation_response,
            )

            # parse the user_representation response
            user_representation_response = parse_xml_content(
                user_representation_response, "representation"
            )

            rprint("[bright_magenta]User Representation:")
            rprint(f"[bright_magenta]{user_representation_response}")

        else:
            raise Exception("\033[91mTom Inference NOT READY YET")
    else:
        rprint("[red]No AI message before this user message[/red]")
        return

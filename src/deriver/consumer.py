import logging
import re
import uuid
from typing import List, Union

from dotenv import load_dotenv
from rich import print as rprint
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud, models, schemas
from ..db import SessionLocal
from .voe import (
    user_prediction_thought,
    user_prediction_thought_revision,
    voe_thought,
    voe_derive_facts,
    check_voe_list,
)

load_dotenv()

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
    collection: models.Collection
    collection = await crud.get_collection_by_name(
        db, payload["app_id"], payload["user_id"], "honcho"
    )
    if collection is None:
        collection_create = schemas.CollectionCreate(name="honcho", metadata={})
        collection = await crud.create_collection(
            db,
            collection=collection_create,
            app_id=payload["app_id"],
            user_id=payload["user_id"],
        )
    collection_id = collection.id
    processing_args = [
        payload["content"],
        payload["app_id"],
        payload["user_id"],
        payload["session_id"],
        collection_id,
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
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    collection_id: uuid.UUID,
    message_id: uuid.UUID,
    db: AsyncSession,
):
    """
    Process an AI message. Make a prediction about what the user is going to say to it.
    """
    rprint(f"[green]Processing AI message: {content}[/green]")

    subquery = (
        select(models.Message.created_at)
        .where(models.Message.id == message_id)
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

    # user prediction thought
    user_prediction_thought_response = await user_prediction_thought(chat_history_str)
    prediction = parse_xml_content(user_prediction_thought_response, "prediction")
    additional_data = parse_xml_content(user_prediction_thought_response, "additional-data")
    if additional_data:
        additional_data = [item.split('. ', 1)[1] for item in additional_data.split('\n') if item.strip()]
    else:
        additional_data = []
        
    ## query the collection to build the context
    response = await crud.query_documents(
        db,
        app_id=app_id,
        user_id=user_id,
        collection_id=collection_id,
        query="\n".join(additional_data),
        top_k=15,
    )
    additional_data_list = [document.content for document in response]

    context_str = "\n".join(additional_data_list)

    # user prediction thought revision given the context
    user_prediction_thought_revision_response = await user_prediction_thought_revision(
        user_prediction_thought=user_prediction_thought_response,
        retrieved_context=context_str,
        chat_history=chat_history_str,
    )
    revision = parse_xml_content(user_prediction_thought_revision_response, "revision")

    if not revision:
        rprint("[blue]Model predicted no changes to the user prediction thought")
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought",
            prediction,
        )
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought_revision",
            prediction,
        )
    else:
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought",
            prediction,
        )
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought_revision",
            revision,
        )

    await db.commit()


    rprint("[blue]User Prediction Thought:")
    content_lines = str(prediction)
    rprint(f"[blue]{content_lines}")

    rprint("[deep_pink1]User Prediction Thought Revision Response:")
    content_lines = str(revision)
    rprint(f"[deep_pink1]{content_lines}")


async def process_user_message(
    content: str,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    collection_id: uuid.UUID,
    message_id: uuid.UUID,
    db: AsyncSession,
):
    """
    Process a user message. If there are revised user predictions to run VoE against, run it. Otherwise pass.
    """
    rprint(f"[orange1]Processing User Message: {content}")
    subquery = (
        select(models.Message.created_at)
        .where(models.Message.id == message_id)
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
        metamessages_stmt = (
            select(models.Metamessage)
            .where(models.Metamessage.message_id == ai_message.id)
            .where(
                models.Metamessage.metamessage_type
                == "user_prediction_thought_revision"
            )
            .order_by(models.Metamessage.created_at.asc())
            .limit(1)
        )
        response = await db.execute(metamessages_stmt)
        metamessage = response.scalar_one_or_none()

        if metamessage and metamessage.content:
            rprint(f"[orange1]Metamessage: {metamessage.content}")

            # VoE thought
            voe_thought_response = await voe_thought(
                user_prediction_thought_revision=metamessage.content,
                actual=content
            )
            voe_thought_parsed = parse_xml_content(voe_thought_response, "assessment")

            # VoE derive facts
            voe_derive_facts_response = await voe_derive_facts(
                ai_message=ai_message.content,
                user_prediction_thought_revision=metamessage.content,
                actual=content,
                voe_thought=voe_thought_parsed,
            )
            voe_derive_facts_parsed = parse_xml_content(voe_derive_facts_response, "facts")

            rprint("[orange1]Voe Thought:")
            content_lines = str(voe_thought_parsed)
            rprint(f"[orange1]{content_lines}")

            rprint("[orange1]Voe Derive Facts Response:")
            content_lines = str(voe_derive_facts_parsed)
            rprint(f"[orange1]{content_lines}")

            if voe_derive_facts_parsed != "None":
                facts = re.findall(r"\d+\.\s([^\n]+)", voe_derive_facts_parsed)
                rprint("[orange1]The Facts Themselves:")
                rprint(facts)
            else:
                facts = []
            new_facts = await check_dups(app_id, user_id, collection_id, facts)

            for fact in new_facts:
                create_document = schemas.DocumentCreate(content=fact)
                async with SessionLocal() as db:
                    doc = await crud.create_document(
                        db,
                        document=create_document,
                        app_id=app_id,
                        user_id=user_id,
                        collection_id=collection_id,
                    )
                    rprint(f"[orange1]Returned Document: {doc.content}")
        else:
            raise Exception("\033[91mUser Thought Prediction Revision NOT READY YET")
    else:
        rprint("[red]No AI message before this user message[/red]")
        return


async def check_dups(
    app_id: uuid.UUID, user_id: uuid.UUID, collection_id: uuid.UUID, facts: List[str]
):
    """Check that we're not storing duplicate facts"""

    result = None
    new_facts = []
    async with SessionLocal() as db:
        result = await crud.query_documents(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            query="\n".join(facts),
            top_k=15,
        )
    existing_facts = [document.content for document in result]
    if len(existing_facts) == 0:  # we just never had any facts
        rprint(f"[light_steel_blue]We have no existing facts.\n New facts: {facts}")
    else:
        checked_list = await check_voe_list(existing_facts, facts)   # this returns a numbered list or "None"
        if checked_list != "None":
            new_facts_list = checked_list.split('\n')
            for fact in new_facts_list:
                if fact.strip():  # Check if the fact is not empty
                    fact_content = fact.split('. ', 1)[1] if '. ' in fact else fact
                    new_facts.append(fact_content.strip())
                    rprint(f"[light_steel_blue]New Fact: {fact_content.strip()}")
        else:
            rprint("[light_steel_blue]No new facts found")

    return new_facts
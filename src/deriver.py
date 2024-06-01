import asyncio
import logging
import os
import re
import uuid
from typing import List

import sentry_sdk
import uvloop
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, models, schemas
from .db import SessionLocal
from .voe import (
    CheckVoeList,
    UserPredictionThought,
    UserPredictionThoughtRevision,
    VoeDeriveFacts,
    VoeThought,
)

load_dotenv()

SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
    )


# Turn of SQLAlchemy Echo logging
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
    print(f"\033[92mProcessing AI message: {content}")

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
    user_prediction_thought = UserPredictionThought(chat_history=chat_history_str)
    user_prediction_thought_response = await user_prediction_thought.call_async()

    ## query the collection to build the context
    additional_data = re.findall(
        r"\d+\.\s([^\n]+)", user_prediction_thought_response.content
    )
    additional_data_set = set()
    additional_data_list = []
    for d in additional_data:
        response = await crud.query_documents(
            db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            query=d,
            top_k=3,
        )
        for document in response:
            additional_data_set.add(document)
        additional_data_list = list(additional_data_set)

    context_str = "\n".join([document.content for document in additional_data_list])

    # user prediction thought revision given the context
    user_prediction_thought_revision = UserPredictionThoughtRevision(
        user_prediction_thought=user_prediction_thought_response.content,
        retrieved_context=context_str,
        chat_history=chat_history_str,
    )
    user_prediction_thought_revision_response = (
        await user_prediction_thought_revision.call_async()
    )

    if user_prediction_thought_revision_response.content == "None":
        print(f"\033[94mModel predicted no changes to the user prediction thought")
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought",
            user_prediction_thought_response.content,
        )
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought_revision",
            user_prediction_thought_response.content,
        )
    else:
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought",
            user_prediction_thought_response.content,
        )
        await add_metamessage(
            db,
            message_id,
            "user_prediction_thought_revision",
            user_prediction_thought_revision_response.content,
        )

    await db.commit()

    # debugging
    print(f"\033[94m=================")
    print(f"\033[94mUser Prediction Thought Prompt:")
    content_lines = str(user_prediction_thought).split("\n")
    for line in content_lines:
        print(f"\033[94m{line}")
    print(f"\033[94mUser Prediction Thought:")
    content_lines = str(user_prediction_thought_response.content).split("\n")
    for line in content_lines:
        print(f"\033[94m{line}")
    print(f"\033[94m=================\033[0m")

    print(f"\033[95m=================")
    print(f"\033[95mUser Prediction Thought Revision:")
    content_lines = str(user_prediction_thought_revision).split("\n")
    for line in content_lines:
        print(f"\033[95m{line}")
    print(f"\033[95mUser Prediction Thought Revision Response:")
    content_lines = str(user_prediction_thought_revision_response.content).split("\n")
    for line in content_lines:
        print(f"\033[95m{line}")
    print(f"\033[95m=================\033[0m")


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
    print(f"\033[93mProcessing User Message: {content}")
    subquery = (
        select(models.Message.created_at)
        .where(models.Message.id == message_id)
        .scalar_subquery()
    )

    messages_stmt = (
        select(models.Message)
        .where(models.Message.created_at < subquery)
        .where(models.Message.session_id == session_id)
        .where(models.Message.is_user == False)
        .limit(1)
    )

    response = await db.execute(messages_stmt)
    ai_message = response.scalar_one_or_none()

    if ai_message and ai_message.content:
        print(f"\033[93mAI Message: {ai_message.content}")
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
            print(f"\033[93mMetamessage: {metamessage.content}")

            # VoE thought
            voe_thought = VoeThought(
                user_prediction_thought_revision=metamessage.content, actual=content
            )
            voe_thought_response = await voe_thought.call_async()

            # VoE derive facts
            voe_derive_facts = VoeDeriveFacts(
                ai_message=ai_message.content,
                user_prediction_thought_revision=metamessage.content,
                actual=content,
                voe_thought=voe_thought_response.content,
            )
            voe_derive_facts_response = await voe_derive_facts.call_async()

            # debugging
            print(f"\033[93m=================")
            print(f"\033[93mVoe Thought Prompt:")
            content_lines = str(voe_thought).split("\n")
            for line in content_lines:
                print(f"\033[93m{line}")
            print(f"\033[93mVoe Thought:")
            content_lines = str(voe_thought_response.content).split("\n")
            for line in content_lines:
                print(f"\033[93m{line}")
            print(f"\033[93m=================\033[0m")

            print(f"\033[93m=================")
            print(f"\033[93mVoe Derive Facts Prompt:")
            content_lines = str(voe_derive_facts).split("\n")
            for line in content_lines:
                print(f"\033[93m{line}")
            print(f"\033[93mVoe Derive Facts Response:")
            content_lines = str(voe_derive_facts_response.content).split("\n")
            for line in content_lines:
                print(f"\033[93m{line}")
            print(f"\033[93m=================\033[0m")

            facts = re.findall(r"\d+\.\s([^\n]+)", voe_derive_facts_response.content)
            print(f"\033[93m=================")
            print(f"\033[93mThe Facts Themselves:")
            print(facts)
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
                    print(f"\033[93mReturned Document: {doc.content}")
        else:
            raise Exception(f"\033[91mUser Thought Prediction Revision NOT READY YET")
    else:
        print(f"\033[91mNo AI message before this user message")
        return


async def check_dups(
    app_id: uuid.UUID, user_id: uuid.UUID, collection_id: uuid.UUID, facts: List[str]
):
    """Check that we're not storing duplicate facts"""

    check_duplication = CheckVoeList(existing_facts=[], new_fact="")
    result = None
    new_facts = []
    global_existing_facts = []  # for debugging
    for fact in facts:
        async with SessionLocal() as db:
            result = await crud.query_documents(
                db=db,
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                query=fact,
                top_k=5,
            )
        existing_facts = [document.content for document in result]
        if len(existing_facts) == 0:
            new_facts.append(fact)
            print(f"New Fact: {fact}")
            continue

        global_existing_facts.extend(existing_facts)  # for debugging

        check_duplication.existing_facts = existing_facts
        check_duplication.new_fact = fact
        response = await check_duplication.call_async()
        print("==================")
        print("Dedupe Responses")
        print(response.content)
        print("==================")
        if response.content == "true":
            new_facts.append(fact)
            print(f"New Fact: {fact}")
            continue

    print("===================")
    print(f"Existing Facts: {global_existing_facts}")
    print(f"Net New Facts {new_facts}")
    print("===================")
    return new_facts


async def dequeue(semaphore: asyncio.Semaphore, queue_empty_flag: asyncio.Event):
    async with semaphore, SessionLocal() as db:
        try:
            result = await db.execute(
                select(models.QueueItem)
                .order_by(models.QueueItem.created_at)
                .where(models.QueueItem.processed == False)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            item = result.scalar_one_or_none()

            if item:
                print("========")
                print("Processing item")
                print("========")
                await process_item(db, payload=item.payload)
                item.processed = True
                await db.commit()
            else:
                # No items to process, set the queue_empty_flag
                queue_empty_flag.set()

        except Exception as e:
            print("==========")
            print("Exception")
            print(e)
            print("==========")
            await db.rollback()


async def polling_loop(semaphore: asyncio.Semaphore, queue_empty_flag: asyncio.Event):
    while True:
        if queue_empty_flag.is_set():
            await asyncio.sleep(1)  # Sleep briefly if the queue is empty
            queue_empty_flag.clear()  # Reset the flag
            continue
        if semaphore.locked():
            await asyncio.sleep(2)  # Sleep briefly if the semaphore is fully locked
            continue
        task = asyncio.create_task(dequeue(semaphore, queue_empty_flag))
        await asyncio.sleep(0)  # Yield control to allow tasks to run


async def main():
    semaphore = asyncio.Semaphore(1)  # Limit to 5 concurrent dequeuing operations
    queue_empty_flag = asyncio.Event()  # Event to signal when the queue is empty
    await polling_loop(semaphore, queue_empty_flag)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())

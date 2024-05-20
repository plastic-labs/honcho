import asyncio
import os
import re
import uuid
from typing import List

import sentry_sdk
import uvloop
from dotenv import load_dotenv
# from realtime.connection import Socket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, models, schemas
from .db import SessionLocal
from .voe import (
    UserPredictionThought, 
    UserPredictionThoughtRevision, 
    VoeThought, 
    VoeDeriveFacts, 
    CheckVoeList
)

load_dotenv()

SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
    )


# SUPABASE_ID = os.getenv("SUPABASE_ID")
# SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

async def process_item(db: AsyncSession, payload: dict):

    collection: models.Collection
    # async with SessionLocal() as db:
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
    if payload["is_user"]:
        await process_user_message(
            payload["content"],
            payload["app_id"],
            payload["user_id"],
            payload["session_id"],
            collection_id,
            payload["message_id"],
            db,
        )
    else:
        await process_ai_message(
            payload["content"],
            payload["app_id"],
            payload["user_id"],
            payload["session_id"],
            collection_id,
            payload["message_id"],
            db,
        )
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
    Process an AI message. If there's enough of a conversation history to run user prediction, run it. Otherwise pass.
    """
    messages_stmt = await crud.get_messages(
        db=db, app_id=app_id, user_id=user_id, session_id=session_id, reverse=True
    )
    messages_stmt = messages_stmt.limit(10)
    response = await db.execute(messages_stmt)
    messages = response.scalars().all()
    # messages = messages[::-1]
    contents = [m.content for m in messages]
    print("===================")
    print("Contents")
    print(contents)
    print("===================")

    # there needs to be at least one user and one ai message
    if len(contents) > 2:

        chat_history_str = "\n".join(
            [f"user: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
        )
        # user prediction thought
        user_prediction_thought = UserPredictionThought(chat_history=chat_history_str)
        user_prediction_thought_response = await user_prediction_thought.call_async()

        ## query the collection to build the context
        additional_data = re.findall(r"\d+\.\s([^\n]+)", user_prediction_thought_response.content)
        additional_data_list = []
        for d in additional_data:
            response = await crud.query_documents(db, app_id=app_id, user_id=user_id, collection_id=collection_id, query=d, top_k=3)
            additional_data_list.extend([document.content for document in response])

        context_str = "\n".join(additional_data_list)

        # user prediction thought revision given the context
        user_prediction_thought_revision = UserPredictionThoughtRevision(
            user_prediction_thought=user_prediction_thought_response.content, 
            retrieved_context=context_str, 
            chat_history=chat_history_str
        )
        user_prediction_thought_revision_response = await user_prediction_thought_revision.call_async()

        upt_metamessage = models.Metamessage(
            message_id=message_id,
            metamessage_type="user_prediction_thought",
            content=user_prediction_thought_response.content,
            h_metadata={},
        )

        uptr_metamessage = models.Metamessage(
            message_id=message_id,
            metamessage_type="user_prediction_thought_revision",
            content=user_prediction_thought_revision_response.content,
            h_metadata={},
        )

        db.add(upt_metamessage)
        db.add(uptr_metamessage)
        await db.commit()

    else:
        pass


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
    Process a user message. If there's enough of a conversation history to run VoE, run it. Otherwise pass.
    """  
    messages_stmt = await crud.get_messages(
        db=db, app_id=app_id, user_id=user_id, session_id=session_id, reverse=True
    )
    messages_stmt = messages_stmt.limit(10)
    response = await db.execute(messages_stmt)
    messages = response.scalars().all()
    contents = [m.content for m in messages]
    print("===================")
    print("Contents")
    print(contents)
    print("===================")

    # get the most recent user thought prediction revision
    metamessages_stmt = await crud.get_metamessages(
        db=db, 
        app_id=app_id, 
        user_id=user_id, 
        session_id=session_id, 
        message_id=message_id, 
        metamessage_type="user_prediction_thought_revision", 
        reverse=True
    )
    metamessages_stmt = metamessages_stmt.limit(1)
    response = await db.execute(metamessages_stmt)
    metamessage = response.scalars().all()

    if metamessage:
        # VoE thought
        voe_thought = VoeThought(
            user_prediction_thought_revision=metamessage.content, 
            actual=content
        )
        voe_thought_response = await voe_thought.call_async()

        # VoE derive facts
        most_recent_ai_message = next((m.content for m in messages if not m.is_user), "")
        voe_derive_facts = VoeDeriveFacts(
            ai_message=most_recent_ai_message,
            user_prediction_thought_revision=metamessage.content, 
            actual=content, 
            voe_thought=voe_thought_response.content
        )
        voe_derive_facts_response = await voe_derive_facts.call_async()

        # check dups
        facts = re.findall(r"\d+\.\s([^\n]+)", voe_derive_facts_response.content)
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
                print(f"Returned Document: {doc}")

    else:
        pass


async def check_dups(
    app_id: uuid.UUID, user_id: uuid.UUID, collection_id: uuid.UUID, facts: List[str]
):
    """Check that we're not storing duplicate facts"""

    check_duplication = CheckVoeList(existing_facts=[], facts=[])
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
        check_duplication.fact = fact
        response = await check_duplication.call_async()
        print("==================")
        print("Dedupe Responses")
        print(response)
        print(response.content)
        print("==================")
        if response.content == "True":
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
                .where(models.QueueItem.processed == False)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            item = result.scalar_one_or_none()

            if item:
                print("========")
                print("Processing")
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
            print("========")
            print("Queue is empty flag")
            print("========")
            await asyncio.sleep(5)  # Sleep briefly if the queue is empty
            queue_empty_flag.clear()  # Reset the flag
            continue
        if semaphore.locked():
            print("========")
            print("Semaphore Locked")
            print("========")
            await asyncio.sleep(2)  # Sleep briefly if the semaphore is fully locked
            continue
        task = asyncio.create_task(dequeue(semaphore, queue_empty_flag))
        task.add_done_callback(lambda t: print(f"Task done: {t}"))
        await asyncio.sleep(0)  # Yield control to allow tasks to run
        # tasks = []
        # for _ in range(5):
        #     tasks.append(task)
        # await asyncio.gather(*tasks)
        # await dequeue()
        # await asyncio.sleep(5)


async def main():
    semaphore = asyncio.Semaphore(1)  # Limit to 5 concurrent dequeuing operations
    queue_empty_flag = asyncio.Event()  # Event to signal when the queue is empty
    await polling_loop(semaphore, queue_empty_flag)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())
    # URL = f"wss://{SUPABASE_ID}.supabase.co/realtime/v1/websocket?apikey={SUPABASE_API_KEY}&vsn=1.0.0"
    # URL = f"ws://127.0.0.1:54321/realtime/v1/websocket?apikey={SUPABASE_API_KEY}"  # For local Supabase
    # listen_to_websocket(URL)
    # s = Socket(URL)
    # s.connect()

    # channel = s.set_channel("realtime:public:messages")
    # channel.join().on("INSERT", lambda payload: asyncio.create_task(callback(payload)))
    # s.listen()

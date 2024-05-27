import asyncio
import logging
import os
import re
import uuid
from typing import List
from datetime import datetime

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
    Process an AI message. If there's enough of a conversation history to run user prediction, run it. Otherwise pass.
    """
    messages_stmt = await crud.get_messages(
        db=db, app_id=app_id, user_id=user_id, session_id=session_id, reverse=False
    )
    messages_stmt = messages_stmt.limit(10)
    response = await db.execute(messages_stmt)
    messages = response.scalars().all()
    # messages = messages[::-1]
    contents = [m.content for m in messages]

    # there needs to be at least one user and one ai message
    if len(contents) > 2:
        print("\033[91m===================")
        print("\033[91mProcessing AI message:")
        content_lines = content.split('\n')
        for line in content_lines:
            print(f"\033[91m{line}")
        print("\033[91m===================\033[0m")
        chat_history_str = "\n".join(
            [
                f"user: {m.content}" if m.is_user else f"ai: {m.content}"
                for m in messages
            ]
        )
        # user prediction thought
        user_prediction_thought = UserPredictionThought(chat_history=chat_history_str)
        user_prediction_thought_response = await user_prediction_thought.call_async()

        ## query the collection to build the context
        additional_data = re.findall(
            r"\d+\.\s([^\n]+)", user_prediction_thought_response.content
        )
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
            additional_data_list.extend([document.content for document in response])

        context_str = "\n".join(additional_data_list)

        # user prediction thought revision given the context
        user_prediction_thought_revision = UserPredictionThoughtRevision(
            user_prediction_thought=user_prediction_thought_response.content,
            retrieved_context=context_str,
            chat_history=chat_history_str,
        )
        user_prediction_thought_revision_response = (
            await user_prediction_thought_revision.call_async()
        )

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

        print("\033[94m==================")
        print("\033[94mUser Prediction Thought")
        content_lines = user_prediction_thought_response.content.split('\n')
        for line in content_lines:
            print(f"\033[94m{line}")
        print("\033[94m==================\033[0m")

        print("\033[92m==================")
        print("\033[92mUser Prediction Thought Revision")
        content_lines = user_prediction_thought_revision_response.content.split('\n')
        for line in content_lines:
            print(f"\033[92m{line}")
        print("\033[92m==================\033[0m")

        db.add(upt_metamessage)
        db.add(uptr_metamessage)
        await db.commit()

    else:
        print(f"\033[91m===================")
        print(f"\033[91mAI Message Processed -- Not enough conversation history for User Prediction")
        print(f"\033[91m===================\033[0m")


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
    messages_stmt = await crud.get_messages(
        db=db, app_id=app_id, user_id=user_id, session_id=session_id, reverse=True
    ) # could use a filter for "is_user" = False, tried it but don't know how to get it to work
    # there's also the case where the AI response gets written to honcho as this one's executing
    # so the most recent AI message isn't the one before this user message
    # need to ensure the AI message we query comes before this user message
    messages_stmt = messages_stmt.limit(10)
    response = await db.execute(messages_stmt)
    messages = response.scalars().all()
    contents = [m.content for m in messages]
    print(f"\033[93mMessages Queried: {contents}")

    if len(contents) > 2:
        print("\033[93m===================")
        print("\033[93mProcessing user message:")
        content_lines = content.split('\n')
        for line in content_lines:
            print(f"\033[93m{line}")
        print("\033[93m===================\033[0m")
        # get the most recent ai message with user thought predictions associated with it
        ai_message = schemas.Message(
            content="",
            is_user=False,
            session_id=session_id,
            id=uuid.uuid4(),
            h_metadata={},
            metadata={},
            created_at=datetime.now(),
        )
        for i, message in enumerate(messages):
            if message.id == message_id:
                # check that the next one in the list is the AI message we're looking for
                if i + 1 < len(messages) and not messages[i + 1].is_user:
                    ai_message = messages[i + 1]
                    print(f"Most Recent AI Message: {ai_message.content}")
                    break
                else:
                    print(len(messages))
                    print(messages[i + 1])
                    print(f"\033[93mMessage IDs Matched, but no AI message found before this user message")
                    return
        if ai_message.content:    
            # get the most recent user thought prediction (TODO: Should be querying this one exactly as well)
            metamessages_stmt = await crud.get_metamessages(
                db=db,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                message_id=ai_message.id,
                metamessage_type="user_prediction_thought_revision",
                reverse=True,
            )
            metamessages_stmt = metamessages_stmt.limit(10)
            mm_response = await db.execute(metamessages_stmt)
            metamessages = mm_response.scalars().all()
            mm_contents = [m.content for m in metamessages]
            print("\033[94m===================")
            print(f"\033[94mMetamessages: {mm_contents}")
            print("\033[94m===================\033[0m")
        else:
            print(f"\033[93mNo AI message found before this user message")
            return

        if len(mm_contents) > 0:
            metamessage = mm_contents[0]
            print("\033[94m==================")
            print("\033[94mMost Recent User Prediction Thought Revision")
            content_lines = metamessage.split('\n')
            for line in content_lines:
                print(f"\033[94m{line}")
            print("\033[94m==================\033[0m")
            # VoE thought
            voe_thought = VoeThought(
                user_prediction_thought_revision=metamessage, actual=content
            )
            voe_thought_response = await voe_thought.call_async()

            voe_derive_facts = VoeDeriveFacts(
                ai_message=ai_message.content,
                user_prediction_thought_revision=metamessage,
                actual=content,
                voe_thought=voe_thought_response.content,
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
                    print(f"Returned Document: {doc.content}")

        else:
            print(f"\033[93m===================")
            print(f"\033[93mAttempted to process User message -- No User Prediction Thought Revisions to run VoE against")
            print(f"\033[93m===================\033[0m")
            return
    else:
        print("\033[93m===================")
        print("\033[93mAttempted to process User message -- Not enough conversation history to run VoE")
        print("\033[93m===================\033[0m")
        return




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
        check_duplication.facts = [fact]
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
            # print("========")
            print("Queue is empty flag")
            # print("========")
            await asyncio.sleep(5)  # Sleep briefly if the queue is empty
            queue_empty_flag.clear()  # Reset the flag
            continue
        if semaphore.locked():
            # print("========")
            # print("Semaphore Locked")
            # print("========")
            await asyncio.sleep(2)  # Sleep briefly if the semaphore is fully locked
            continue
        task = asyncio.create_task(dequeue(semaphore, queue_empty_flag))
        # task.add_done_callback(lambda t: print(f"Task done: {t}"))
        await asyncio.sleep(0)  # Yield control to allow tasks to run


async def main():
    semaphore = asyncio.Semaphore(1)  # Limit to 5 concurrent dequeuing operations
    queue_empty_flag = asyncio.Event()  # Event to signal when the queue is empty
    await polling_loop(semaphore, queue_empty_flag)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())

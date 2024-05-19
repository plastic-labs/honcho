import asyncio
import os
import re
import uuid
from typing import List

import sentry_sdk
import uvloop
from dotenv import load_dotenv
from mirascope.openai import OpenAICall, OpenAICallParams

# from realtime.connection import Socket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, models, schemas
from .db import SessionLocal

load_dotenv()

SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
    )


# SUPABASE_ID = os.getenv("SUPABASE_ID")
# SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# user prediction thought + additional data to improve prediction
class UserPredictionThought(OpenAICall):
    prompt_template = """
    USER:
    Generate a "thought" that makes a theory of mind prediction about what the user will say based on the way the conversation has been going. Also list other pieces of data that would help improve your prediction.

    Conversation:
    ```
    {chat_history}
    ```

    Generate the additional pieces of data as a numbered list.

    ASSISTANT:
    Thought:
    """

    chat_history: str
    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")

# user prediction thought revision given context
class UserPredictionThoughtRevision(OpenAICall):
    prompt_template = """
    USER:
    You are tasked with revising theory of mind "thoughts" about what the user is going to say. Here is the thought generated previously:

    Thought: ```
    {user_prediction_thought}
    ```

    Based on this thought, the following personal data has been retrieved:

    Personal Data: ```
    {retrieved_vectors}
    ```

    And here's the conversation history that was used to generate the original thought:

    History: ```
    {chat_history}
    ```
    
    Given the thought, conversation history, and personal data, revise the thought. If there are no changes to be made, output "None".

    ASSISTANT:
    thought revision:
    """
    user_prediction_thought_revision: str
    retrieved_vectors = str
    chat_history: str
    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")

# VoE thought
class VoeThought(OpenAICall):
    prompt_template = """
    USER:
    Below is a "thought" about what the user was going to say, and then what the user actually said. Generate a theory of mind prediction about the user based on the difference between the "thought" and actual response.

    Thought: ```
    {user_prediction_thought_revision}
    ```

    Actual: ```
    {actual}
    ```

    Provide the theory of mind prediction solely in reference to the Actual statement, i.e. do not generate something that negates the thought. Do not speculate anything about the user.
    """
    user_prediction_thought_revision: str
    actual: str
    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")

# VoE derive facts
class VoeDeriveFacts(OpenAICall):
    prompt_template = """
    USER:
    Below is the most recent AI message we sent to a user, a "thought" about what the user was going to say to that, what the user actually responded with, and a theory of mind prediction about the user's response. Derive a fact (or list of facts) about the user based on the difference between the original thought and their actual response plus the theory of mind prediction about that response.

    Most recent AI message: ```
    {ai_message}
    ```

    Thought about what they were going to say: ```
    {user_prediction_thought_revision}
    ```

    Actual response: ```
    {actual}
    ```

    Theory of mind prediction about that response: ```
    {voe_thought}
    ```

    Provide the fact(s) solely in reference to the Actual response and theory of mind prediction about that response; i.e. do not derive a fact that negates the thought about what they were going to say. Do not speculate anything about the user. Each fact must contain enough specificity to stand alone. If there are many facts, list them out. Your response should be a numbered list with each item on a new line, for example: `\n\n1. foo\n\n2. bar\n\n3. baz`. If there's nothing to derive (i.e. the statements are sufficiently similar), print "None".
    """
    user_prediction_thought_revision: str
    actual: str
    voe_thought: str
    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")

# check dups
class CheckVoeList(OpenAICall):
    prompt_template = """
    USER:
    Please compare the following two lists and keep only unique items:

    Old: ```
    {existing_facts}
    ```

    New: ```
    {facts}
    ```
    
    Remove redundant information from the new list and output the remaining facts. Your response should be a numbered list with each fact on a new line, for example: `\n\n1. foo\n\n2. bar\n\n3. baz`. If there's nothing to remove (i.e. the statements are sufficiently different), print "None".
    """
    existing_facts: str
    facts: str
    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")


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
    await process_user_message(
        payload["content"],
        payload["app_id"],
        payload["user_id"],
        payload["session_id"],
        collection_id,
        payload["message_id"],
        db,
    )
    return


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
    Process a user message thought the VoE pipeline
    """
    messages_stmt = await crud.get_messages(
        db=db, app_id=app_id, user_id=user_id, session_id=session_id, reverse=True
    )
    messages_stmt = messages_stmt.limit(10)
    response = await db.execute(messages_stmt)
    messages = response.scalars().all()
    messages = messages[::-1]
    contents = [m.content for m in messages]
    print("===================")
    print("Contents")
    print(contents)
    print("===================")

    chat_history_str = "\n".join(
        [f"user: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
    )

    ############

    # user prediction thought
    user_prediction_thought = UserPredictionThought(chat_history=chat_history_str)
    user_prediction_thought_response = await user_prediction_thought.call_async()

    ## query the dialectic api to build the context
    additional_data = re.findall(r"\d+\.\s([^\n]+)", user_prediction_thought_response.content)

    for datum in additional_data:
        

    # user prediction thought revision given the context

    # VoE thought

    # VoE derive facts

    # check dups

    ###############

    facts_response = await DeriveFacts(
        chat_history=chat_history_str, user_input=content
    ).call_async()
    facts = re.findall(r"\d+\.\s([^\n]+)", facts_response.content)

    print("===================")
    print(f"DERIVED FACTS: {facts}")
    print("===================")
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
        # doc = crud.create_document(content=fact)
    # for fact in new_facts:
    #     session.create_metamessage(
    #         message=user_message, metamessage_type="fact", content=fact
    #     )
    # print(f"Created fact: {fact}")


async def check_dups(
    app_id: uuid.UUID, user_id: uuid.UUID, collection_id: uuid.UUID, facts: List[str]
):
    """Check that we're not storing duplicate facts"""

    check_duplication = CheckDups(existing_facts=[], fact="")
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

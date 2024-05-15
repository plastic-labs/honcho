import asyncio
import os
import re
import uuid
from typing import List

import sentry_sdk
from dotenv import load_dotenv
from mirascope.openai import OpenAICall, OpenAICallParams
from realtime.connection import Socket
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from . import crud, models, schemas
from .db import SessionLocal

load_dotenv()

SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "False").lower() == "true"
if SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        enable_tracing=True,
    )


SUPABASE_ID = os.getenv("SUPABASE_ID")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")


class DeriveFacts(OpenAICall):
    prompt_template = """
    You are tasked with deriving discrete facts about the user based on their input. The goal is to only extract absolute facts from the message, do not make inferences beyond the text provided.

    chat history: ```{chat_history}```
    user input: ```{user_input}```

    Output the facts as a numbered list.
    """

    chat_history: str
    user_input: str

    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")


class CheckDups(OpenAICall):
    prompt_template = """
    Your job is to determine if the new fact exists in the old:

    Old: ```{existing_facts}```

    New: ```{fact}```
    
    If the new fact is sufficiently represented in the old list, return False. Otherwise, if the fact is indeed new, return True.
    """
    existing_facts: List[str]
    fact: str

    call_params = OpenAICallParams(model="gpt-4o-2024-05-13")


async def callback(payload):
    # print(payload["record"]["is_user"])
    # print(type(payload["record"]["is_user"]))
    if payload["record"]["is_user"]:  # Check if the message is from a user
        session_id = payload["record"]["session_id"]
        message_id = payload["record"]["id"]
        content = payload["record"]["content"]

        # Example of querying for a user_id based on session_id, adjust according to your schema
        session: models.Session
        user_id: uuid.UUID
        app_id: uuid.UUID
        async with SessionLocal() as db:
            stmt = (
                select(models.Session)
                .join(models.Session.messages)
                .where(models.Message.id == message_id)
                .where(models.Session.id == session_id)
                .options(selectinload(models.Session.user))
            )
            result = await db.execute(stmt)
            session = result.scalars().one()
            user = session.user
            user_id = user.id
            app_id = user.app_id
        collection: models.Collection
        async with SessionLocal() as db:
            collection = await crud.get_collection_by_name(
                db, app_id, user_id, "honcho"
            )
            if collection is None:
                collection_create = schemas.CollectionCreate(name="honcho", metadata={})
                collection = await crud.create_collection(
                    db,
                    collection=collection_create,
                    app_id=app_id,
                    user_id=user_id,
                )
        collection_id = collection.id
        await process_user_message(
            content, app_id, user_id, session_id, collection_id, message_id
        )
    return


async def process_user_message(
    content: str,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    collection_id: uuid.UUID,
    message_id: uuid.UUID,
):
    """
    Process a user message and derive facts from it (check for duplicates before writing to the collection).
    """
    async with SessionLocal() as db:
        messages_stmt = await crud.get_messages(
            db=db, app_id=app_id, user_id=user_id, session_id=session_id, reverse=True
        )
        messages_stmt = messages_stmt.limit(10)
        response = await db.execute(messages_stmt)
        messages = response.scalars().all()
        messages = messages[::-1]
        # contents = [m.content for m in messages]
        # print(contents)

    chat_history_str = "\n".join(
        [f"user: {m.content}" if m.is_user else f"ai: {m.content}" for m in messages]
    )
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
        if response.content == "True":
            new_facts.append(fact)
            print(f"New Fact: {fact}")
            continue

    print("===================")
    print(f"Existing Facts: {global_existing_facts}")
    print(f"Net New Facts {new_facts}")
    print("===================")
    return new_facts


if __name__ == "__main__":
    # URL = f"wss://{SUPABASE_ID}.supabase.co/realtime/v1/websocket?apikey={SUPABASE_API_KEY}&vsn=1.0.0"
    URL = f"ws://127.0.0.1:54321/realtime/v1/websocket?apikey={SUPABASE_API_KEY}"  # For local Supabase
    # listen_to_websocket(URL)
    s = Socket(URL)
    s.connect()

    channel = s.set_channel("realtime:public:messages")
    channel.join().on("INSERT", lambda payload: asyncio.create_task(callback(payload)))
    s.listen()

import asyncio
import os
import uuid
from typing import List

import sentry_sdk
from dotenv import load_dotenv
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    load_prompt,
)
from langchain_openai import ChatOpenAI
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

llm = ChatOpenAI(model_name="gpt-4")
output_parser = NumberedListOutputParser()

SYSTEM_DERIVE_FACTS = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/derive_facts.yaml")
)
SYSTEM_CHECK_DUPS = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/check_dup_facts.yaml")
)

system_check_dups: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
    prompt=SYSTEM_CHECK_DUPS
)

system_derive_facts: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
    prompt=SYSTEM_DERIVE_FACTS
)


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

    facts = await derive_facts(messages, content)
    print("===================")
    print(f"DERIVED FACTS: {facts}")
    print("===================")
    new_facts = await check_dups(app_id, user_id, collection_id, facts)

    print("===================")
    print(f"CHECKED FOR DUPLICATES: {new_facts}")
    print("===================")

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


async def derive_facts(chat_history, input: str) -> List[str]:
    """Derive facts from the user input"""

    fact_derivation = ChatPromptTemplate.from_messages([system_derive_facts])
    chain = fact_derivation | llm
    response = await chain.ainvoke(
        {
            "chat_history": [
                (
                    "user: " + message.content
                    if message.is_user
                    else "ai: " + message.content
                )
                for message in chat_history
            ],
            "user_input": input,
        }
    )
    facts = output_parser.parse(response.content)

    return facts


async def check_dups(
    app_id: uuid.UUID, user_id: uuid.UUID, collection_id: uuid.UUID, facts: List[str]
):
    """Check that we're not storing duplicate facts"""

    check_duplication = ChatPromptTemplate.from_messages([system_check_dups])
    query = " ".join(facts)
    result = None
    async with SessionLocal() as db:
        result = await crud.query_documents(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            query=query,
            top_k=10,
        )
    # result = collection.query(query=query, top_k=10)
    existing_facts = [document.content for document in result]
    print("===================")
    print(f"Existing Facts {existing_facts}")
    print("===================")
    if len(existing_facts) == 0:
        return facts
    chain = check_duplication | llm
    response = await chain.ainvoke({"existing_facts": existing_facts, "facts": facts})
    new_facts = output_parser.parse(response.content)
    print("===================")
    print(f"New Facts {facts}")
    print("===================")
    return new_facts


if __name__ == "__main__":
    URL = f"wss://{SUPABASE_ID}.supabase.co/realtime/v1/websocket?apikey={SUPABASE_API_KEY}&vsn=1.0.0"
    # URL = f"ws://127.0.0.1:54321/realtime/v1/websocket?apikey={SUPABASE_API_KEY}"  # For local Supabase
    s = Socket(URL)
    s.connect()

    channel = s.set_channel("realtime:public:messages")
    channel.join().on("INSERT", lambda payload: asyncio.create_task(callback(payload)))
    s.listen()

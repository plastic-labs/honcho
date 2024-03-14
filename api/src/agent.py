import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    load_prompt,
)
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, schemas

load_dotenv()

# from supabase import Client

SYSTEM_DIALECTIC = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/dialectic.yaml")
)
system_dialectic: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
    prompt=SYSTEM_DIALECTIC
)

llm: ChatOpenAI = ChatOpenAI(model_name="gpt-4")


async def chat(
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    query: str,
    db: AsyncSession,
):
    collection = await crud.get_collection_by_name(db, app_id, user_id, "honcho")
    retrieved_facts = None
    if collection is None:
        collection_create = schemas.CollectionCreate(name="honcho", metadata={})
        collection = await crud.create_collection(
            db,
            collection=collection_create,
            app_id=app_id,
            user_id=user_id,
        )
    else:
        retrieved_documents = await crud.query_documents(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection.id,
            query=query,
            top_k=1,
        )
        if len(retrieved_documents) > 0:
            retrieved_facts = retrieved_documents[0].content

    dialectic_prompt = ChatPromptTemplate.from_messages([system_dialectic])
    chain = dialectic_prompt | llm
    response = await chain.ainvoke(
        {
            "agent_input": query,
            "retrieved_facts": retrieved_facts if retrieved_facts else "None",
        }
    )

    return schemas.AgentChat(content=response.content)


async def hydrate():
    pass


async def insight():
    pass

import os
import uuid

from dotenv import load_dotenv
from mirascope.base import BaseConfig
from mirascope.openai import OpenAICall, OpenAICallParams, azure_client_wrapper
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, schemas

load_dotenv()


class Dialectic(OpenAICall):
    prompt_template = """
    You are tasked with responding to the query based on the context provided. 
    ---
    query: {agent_input}
    context: {retrieved_facts}
    ---
    Provide a brief, matter-of-fact, and appropriate response to the query based on the context provided. If the context provided doesn't aid in addressing the query, return None. 
    """
    agent_input: str
    retrieved_facts: str

    configuration = BaseConfig(
        client_wrappers=[
            azure_client_wrapper(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        ]
    )
    call_params = OpenAICallParams(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"), temperature=1.2, top_p=0.5
    )
    # call_params = OpenAICallParams(model="gpt-4o-2024-05-13")


async def prep_inference(
    db: AsyncSession,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
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

    chain = Dialectic(
        agent_input=query,
        retrieved_facts=retrieved_facts if retrieved_facts else "None",
    )
    return chain


async def chat(
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
    db: AsyncSession,
):
    chain = await prep_inference(db, app_id, user_id, query)
    response = await chain.call_async()

    return schemas.AgentChat(content=response.content)


async def stream(
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
    db: AsyncSession,
):
    chain = await prep_inference(db, app_id, user_id, query)
    return chain.stream_async()

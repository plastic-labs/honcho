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
    conversation_history: {chat_history}
    ---
    Provide a brief, matter-of-fact, and appropriate response to the query based on the context provided. If the context provided doesn't aid in addressing the query, return None. 
    """
    agent_input: str
    retrieved_facts: str
    chat_history: list[str]

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


async def chat_history(
    db: AsyncSession, app_id: uuid.UUID, user_id: uuid.UUID, session_id: uuid.UUID
) -> list[str]:
    stmt = await crud.get_messages(db, app_id, user_id, session_id)
    results = await db.execute(stmt)
    messages = results.scalars()
    history = []
    for message in messages:
        if message.is_user:
            history.append(f"user:{message.content}")
        else:
            history.append(f"assistant:{message.content}")
    return history


async def prep_inference(
    db: AsyncSession,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
) -> None | list[str]:
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
            top_k=3,
        )
        if len(retrieved_documents) > 0:
            retrieved_facts = [d.content for d in retrieved_documents]

    return retrieved_facts


# async def chat(
#     app_id: uuid.UUID,
#     user_id: uuid.UUID,
#     query: str,
#     db: AsyncSession,
#     stream: bool = False,
# ):
#     retrieved_facts = await prep_inference(db, app_id, user_id, query)
#     facts = "None"
#     if retrieved_facts is not None:
#         facts = "\n".join(retrieved_facts)
#     chain = Dialectic(
#         agent_input=query,
#         retrieved_facts=facts,
#     )
#
#     if stream:
#         return chain.stream_async()
#     response = chain.call()
#     return schemas.AgentChat(content=response.content)


async def chat(
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    queries: schemas.AgentQuery,
    db: AsyncSession,
    stream: bool = False,
):
    if isinstance(queries.queries, str):
        questions = [queries.queries]
    else:
        questions = queries.queries

    all_facts = set()
    for query in questions:
        retrieved_facts = await prep_inference(db, app_id, user_id, query)
        if retrieved_facts is not None:
            all_facts.update(retrieved_facts)
    facts = "None"
    if len(all_facts) > 0:
        facts = "\n".join(all_facts)
    query = "\n".join(questions)
    history = await chat_history(db, app_id, user_id, session_id)
    chain = Dialectic(
        agent_input=query,
        retrieved_facts=facts,
        chat_history=history,
    )
    if stream:
        return chain.stream_async()
    response = chain.call()
    return schemas.AgentChat(content=response.content)


# async def stream(
#     app_id: uuid.UUID,
#     user_id: uuid.UUID,
#     query: str,
#     db: AsyncSession,
# ):
#     retrieved_facts = await prep_inference(db, app_id, user_id, query)
#     chain = Dialectic(
#         agent_input=query,
#         retrieved_facts=retrieved_facts if retrieved_facts else "None",
#     )
#     return chain.stream_async()

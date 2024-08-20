import os
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from anthropic import Anthropic, AsyncAnthropic

from . import crud, schemas

load_dotenv()

# Initialize the Anthropic client
anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def prep_inference(
    db: AsyncSession,
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
) -> str:
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

    prompt = f"""
    You are tasked with responding to the query based on the context provided. 
    ---
    query: {query}
    context: {retrieved_facts if retrieved_facts else "None"}
    ---
    Provide a brief, matter-of-fact, and appropriate response to the query based on the context provided. If the context provided doesn't aid in addressing the query, return None. 
    """
    return prompt

async def chat(
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
    db: AsyncSession,
) -> schemas.AgentChat:
    prompt = await prep_inference(db, app_id, user_id, query)
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return schemas.AgentChat(content=response.content[0].text)

async def stream(
    app_id: uuid.UUID,
    user_id: uuid.UUID,
    query: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    prompt = await prep_inference(db, app_id, user_id, query)
    stream = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    async for chunk in stream:
        if chunk.type == "content_block_delta":
            yield chunk.delta.text
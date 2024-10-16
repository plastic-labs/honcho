import asyncio
from collections.abc import Iterable

from dotenv import load_dotenv
from mirascope.openai import OpenAICall, OpenAICallParams

from src import crud, schemas
from src.db import SessionLocal

load_dotenv()


class AsyncSet:
    def __init__(self):
        self._set: set[str] = set()
        self._lock = asyncio.Lock()

    async def add(self, item: str):
        async with self._lock:
            self._set.add(item)

    async def update(self, items: Iterable[str]):
        async with self._lock:
            self._set.update(items)

    def get_set(self) -> set[str]:
        return self._set.copy()


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

    call_params = OpenAICallParams(model="gpt-4o", temperature=1.2, top_p=0.5)


async def chat_history(app_id: str, user_id: str, session_id: str) -> list[str]:
    async with SessionLocal() as db:
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
    app_id: str,
    user_id: str,
    query: str,
    collection_name: str,
) -> None | list[str]:
    async with SessionLocal() as db:
        collection = await crud.get_collection_by_name(
            db, app_id, user_id, collection_name
        )
        retrieved_facts = None
        if collection:
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


async def generate_facts(
    app_id: str,
    user_id: str,
    fact_set: AsyncSet,
    collection_name: str,
    questions: list[str],
):
    async def fetch_facts(query):
        retrieved_facts = await prep_inference(app_id, user_id, query, collection_name)
        if retrieved_facts is not None:
            await fact_set.update(retrieved_facts)

    await asyncio.gather(*[fetch_facts(query) for query in questions])


async def fact_generator(
    app_id: str,
    user_id: str,
    collections: list[str],
    questions: list[str],
):
    fact_set = AsyncSet()
    fact_tasks = [
        generate_facts(app_id, user_id, fact_set, col, questions) for col in collections
    ]
    await asyncio.gather(*fact_tasks)
    fact_set_copy = fact_set.get_set()
    facts = "None"
    if fact_set_copy and len(fact_set_copy) > 0:
        facts = "\n".join(fact_set_copy)
    return facts


async def chat(
    app_id: str,
    user_id: str,
    session_id: str,
    query: schemas.AgentQuery,
    stream: bool = False,
):
    questions = [query.queries] if isinstance(query.queries, str) else query.queries

    final_query = "\n".join(questions) if len(questions) > 1 else questions[0]

    collections = (
        [query.collections] if isinstance(query.collections, str) else query.collections
    )

    # Run fact generation and chat history retrieval concurrently
    fact_task = fact_generator(app_id, user_id, collections, questions)
    history_task = chat_history(app_id, user_id, session_id)

    # Wait for both tasks to complete
    facts, history = await asyncio.gather(fact_task, history_task)

    chain = Dialectic(
        agent_input=final_query,
        retrieved_facts=facts,
        chat_history=history,
    )

    if stream:
        return chain.stream_async()
    response = chain.call()
    return schemas.AgentChat(content=response.content)

import asyncio
import os
from collections.abc import Iterable

from anthropic import Anthropic
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
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


class Dialectic:
    def __init__(
        self, agent_input: str, user_representation: str, chat_history: list[str]
    ):
        self.agent_input = agent_input
        self.user_representation = user_representation
        self.chat_history = chat_history
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def call(self):
        prompt = f"""
        You are tasked with responding to the query based on the context provided. 
        <query>{self.agent_input}</query>
        <context>{self.user_representation}</context>
        <conversation_history>{self.chat_history}</conversation_history>
        Provide a brief, matter-of-fact, and appropriate response to the query based on the context provided. If the context provided doesn't aid in addressing the query, return only the word "None". 
        """

        response = self.client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="claude-3-5-sonnet-20240620",
            max_tokens=300,
        )
        return response.content

    def stream(self):
        prompt = f"""
        You are tasked with responding to the query based on the context provided. 
        <query>{self.agent_input}</query>
        <context>{self.user_representation}</context>
        <conversation_history>{self.chat_history}</conversation_history>
        Provide a brief, matter-of-fact, and appropriate response to the query based on the context provided. If the context provided doesn't aid in addressing the query, return only the word "None". 
        """

        yield from self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=300,
            stream=True,
        )


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


async def get_latest_user_representation(
    db: AsyncSession, app_id: str, user_id: str, session_id: str
) -> str:
    stmt = (
        select(models.Metamessage)
        .join(models.Message, models.Message.public_id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.public_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Metamessage.metamessage_type == "user_representation")
        .limit(1)
    )
    result = await db.execute(stmt)
    representation = result.scalar_one_or_none()
    return (
        representation.content
        if representation
        else "No user representation available."
    )


async def chat(
    app_id: str,
    user_id: str,
    session_id: str,
    query: schemas.AgentQuery,
    stream: bool = False,
):
    questions = [query.queries] if isinstance(query.queries, str) else query.queries
    final_query = "\n".join(questions) if len(questions) > 1 else questions[0]

    async with SessionLocal() as db:
        # Run user representation retrieval and chat history retrieval concurrently
        user_rep_task = get_latest_user_representation(db, app_id, user_id, session_id)
        history_task = chat_history(app_id, user_id, session_id)

        # Wait for both tasks to complete
        user_representation, history = await asyncio.gather(user_rep_task, history_task)

    chain = Dialectic(
        agent_input=final_query,
        user_representation=user_representation,
        chat_history=history,
    )

    if stream:
        return chain.stream()
    response = chain.call()
    return schemas.AgentChat(content=response[0].text)

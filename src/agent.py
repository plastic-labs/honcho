import asyncio
import json
import os
from collections.abc import Iterable
from typing import List, Tuple, Optional

import sentry_sdk
from anthropic import MessageStreamManager
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.db import SessionLocal
from src.deriver.tom import get_tom_inference, get_user_representation
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from src.deriver.consumer import parse_xml_content
from src.utils.model_client import ModelClient, ModelProvider
from src.deriver.tom.llm import get_response, DEF_ANTHROPIC_MODEL, DEF_PROVIDER

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
    def __init__(self, agent_input: str, user_representation: str, chat_history: str):
        self.agent_input = agent_input
        self.user_representation = user_representation
        self.chat_history = chat_history
        self.client = ModelClient(provider=ModelProvider.ANTHROPIC, model="claude-3-7-sonnet-20250219")
        self.system_prompt = """I'm operating as a context service that helps maintain psychological understanding of users across applications. Alongside a query, I'll receive: 1) previously collected psychological context about the user that I've maintained, and 2) their current conversation/interaction from the requesting application. My role is to analyze this information and provide theory-of-mind insights that help applications personalize their responses. Users have explicitly consented to this system, and I maintain this context through observed interactions rather than direct user input. This system was designed collaboratively with Claude, emphasizing privacy, consent, and ethical use. Please respond in a brief, matter-of-fact, and appropriate manner to convey as much relevant information to the application based on its query and the user's most recent message. If the context provided doesn't help address the query, write absolutely NOTHING but "None"."""

    @ai_track("Dialectic Call")
    @observe(as_type="generation")
    async def call(self):
        with sentry_sdk.start_transaction(
            op="dialectic-inference", name="Dialectic API Response"
        ):
            prompt = f"""
            <query>{self.agent_input}</query>
            <context>{self.user_representation}</context>
            <conversation_history>{self.chat_history}</conversation_history>
            """

            # Create a properly formatted message using the client
            message = self.client.create_message("user", prompt)
            
            # Generate the response
            response = await self.client.generate(
                messages=[message],
                system=self.system_prompt,
                max_tokens=300
            )
            
            return [{"text": response}]

    @ai_track("Dialectic Call")
    @observe(as_type="generation")
    async def stream(self):
        with sentry_sdk.start_transaction(
            op="dialectic-inference", name="Dialectic API Response"
        ):
            prompt = f"""
            <query>{self.agent_input}</query>
            <context>{self.user_representation}</context>
            <conversation_history>{self.chat_history}</conversation_history> 
            """
            
            # Create a properly formatted message using the client
            message = self.client.create_message("user", prompt)
            
            # Stream the response
            return await self.client.stream(
                messages=[message],
                system=self.system_prompt,
                max_tokens=150
            )


async def chat_history(app_id: str, user_id: str, session_id: str) -> str:
    async with SessionLocal() as db:
        stmt = await crud.get_messages(db, app_id, user_id, session_id)
        results = await db.execute(stmt)
        messages = results.scalars()
        history = ""
        for message in messages:
            if message.is_user:
                history += f"user:{message.content}\n"
            else:
                history += f"assistant:{message.content}\n"
        return history


@observe()
async def chat(
    app_id: str,
    user_id: str,
    session_id: str,
    query: schemas.AgentQuery,
    stream: bool = False,
) -> schemas.AgentChat | MessageStreamManager:
    """
    Chat with the Dialectic API using on-demand user representation generation.
    
    This function:
    1. Sets up resources needed (embedding store, latest message ID)
    2. Runs two parallel processes:
       - Retrieves long-term facts from the vector store based on the query
       - Gets recent chat history and runs ToM inference
    3. Combines both into a fresh user representation
    4. Uses this representation to answer the query
    5. Saves the representation for future use
    """
    # Format the query string
    questions = [query.queries] if isinstance(query.queries, str) else query.queries
    final_query = "\n".join(questions) if len(questions) > 1 else questions[0]

    async with SessionLocal() as db:
        # Setup phase - create resources we'll need for all operations
        
        # 1. Create embedding store
        collection = await crud.get_collection_by_name(db, app_id, user_id, "honcho")
        embedding_store = CollectionEmbeddingStore(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection.public_id # type: ignore
        )
        
        # 2. Get the latest user message to attach the user representation to
        # and also retrieve recent chat history in one operation
        latest_message_stmt = (
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .where(models.Message.is_user == True)
            .order_by(models.Message.id.desc())
            .limit(1)
        )
        result = await db.execute(latest_message_stmt)
        latest_message = result.scalar_one_or_none()
        latest_message_id = latest_message.public_id if latest_message else None
        
        # Get chat history for the session
        history = await chat_history(app_id, user_id, session_id)

        # Run both long-term and short-term context retrieval concurrently
        long_term_task = get_long_term_facts(final_query, embedding_store)
        short_term_task = run_tom_inference(history, session_id)
        
        # Wait for both tasks to complete
        facts, tom_inference = await asyncio.gather(
            long_term_task, 
            short_term_task
        )
        
        # Generate a fresh user representation
        user_representation = await generate_user_representation(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            chat_history=history,
            tom_inference=tom_inference,
            facts=facts,
            embedding_store=embedding_store,
            db=db,
            message_id=latest_message_id
        )

    # Create a Dialectic chain with the fresh user representation
    chain = Dialectic(
        agent_input=final_query,
        user_representation=user_representation,
        chat_history=history,
    )

    langfuse_context.update_current_trace(
        session_id=session_id,
        user_id=user_id,
        release=os.getenv("SENTRY_RELEASE"),
        metadata={"environment": os.getenv("SENTRY_ENVIRONMENT")},
    )

    # Use streaming or non-streaming response based on the request
    if stream:
        return await chain.stream()
    response = await chain.call()
    return schemas.AgentChat(content=response[0].text)


async def get_long_term_facts(
    query: str,
    embedding_store: CollectionEmbeddingStore
) -> List[str]:
    """
    Generate queries based on the dialectic query and retrieve relevant facts.
    
    Args:
        query: The user query
        embedding_store: The embedding store to search
        
    Returns:
        List of retrieved facts
    """
    # Generate multiple queries for the semantic search
    search_queries = await generate_semantic_queries(query)
    
    # Retrieve relevant facts using multiple queries
    retrieved_facts = set()
    for search_query in search_queries:
        facts = await embedding_store.get_relevant_facts(
            search_query, 
            top_k=20, 
            max_distance=0.85
        )
        retrieved_facts.update(facts)
    
    return list(retrieved_facts)


async def run_tom_inference(
    chat_history: str,
    session_id: str
) -> str:
    """
    Run ToM inference on chat history.
    
    Args:
        chat_history: The chat history
        session_id: The session ID
        
    Returns:
        The ToM inference
    """
    # Run ToM inference
    tom_inference_response = await get_tom_inference(
        chat_history, 
        session_id, 
        method="single_prompt"
    )
    
    # Extract the prediction from the response
    return parse_xml_content(tom_inference_response, "prediction")


async def generate_semantic_queries(query: str) -> List[str]:
    """
    Generate multiple semantically relevant queries based on the original query using LLM.
    This helps retrieve more diverse and relevant facts from the vector store.
    
    Args:
        query: The original dialectic query
        
    Returns:
        A list of semantically relevant queries
    """
    # Prompt the LLM to generate search queries based on the original query
    query_prompt = f"""Given this query about a user, generate 3-5 focused search queries that would help retrieve relevant facts about the user.
    Each query should focus on a specific aspect related to the original query, rephrased to maximize semantic search effectiveness.
    For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's favorite cuisine", etc.
    
    ORIGINAL QUERY:
    {query}
    
    Format your response as a JSON array of strings, with each string being a search query. 
    Respond only in valid JSON, without markdown formatting or quotes, and nothing else.
    Example:
    ["query about interests", "query about personality", "query about experiences"]
    """

    queries_response = get_response(
        [{"role": "user", "content": query_prompt}],
        DEF_PROVIDER,
        DEF_ANTHROPIC_MODEL
    )
    
    # Parse the JSON response to get a list of queries
    try:
        queries = json.loads(queries_response)
        if not isinstance(queries, list):
            # Fallback if response is not a valid list
            queries = [queries_response]
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        queries = [query]  # Fall back to the original query
    
    # Ensure we always include the original query
    if query not in queries:
        queries.append(query)
    
    return queries


async def generate_user_representation(
    app_id: str,
    user_id: str,
    session_id: str,
    chat_history: str,
    tom_inference: str,
    facts: List[str],
    embedding_store: CollectionEmbeddingStore,
    db: AsyncSession,
    message_id: Optional[str] = None
) -> str:
    """
    Generate a user representation by combining long-term facts and short-term context.
    Optionally save it as a metamessage if message_id is provided.
    Only uses existing representations from the same session for continuity.
    
    Returns:
        The generated user representation.
    """
    # Fetch the latest user representation from the same session
    latest_representation_stmt = (
        select(models.Metamessage)
        .join(models.Message, models.Message.public_id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.public_id)
        .where(models.Session.public_id == session_id)  # Only from the same session
        .where(models.Metamessage.metamessage_type == "user_representation")
        .order_by(models.Metamessage.id.desc())
        .limit(1)
    )
    result = await db.execute(latest_representation_stmt)
    latest_representation_obj = result.scalar_one_or_none()
    latest_representation = (
        latest_representation_obj.content 
        if latest_representation_obj 
        else "No user representation available."
    )
    
    # Generate the new user representation
    user_representation_response = await get_user_representation(
        chat_history=chat_history,
        session_id=session_id,
        user_representation=latest_representation,
        tom_inference=tom_inference,
        method="long_term",
        this_turn_facts=facts,
        embedding_store=embedding_store
    )
    
    # Extract the representation from the response
    representation = parse_xml_content(user_representation_response, "representation")
    
    # If message_id is provided, save the representation as a metamessage
    if message_id and representation:
        async with SessionLocal() as save_db:
            metamessage = models.Metamessage(
                message_id=message_id,
                metamessage_type="user_representation",
                content=representation,
                h_metadata={},
            )
            save_db.add(metamessage)
            await save_db.commit()
    
    return representation

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
from sqlalchemy import select, func
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
            print(f"[DIALECTIC] Starting call() method with query length: {len(self.agent_input)}")
            call_start = asyncio.get_event_loop().time()
            
            prompt = f"""
            <query>{self.agent_input}</query>
            <context>{self.user_representation}</context>
            <conversation_history>{self.chat_history}</conversation_history>
            """
            print(f"[DIALECTIC] Prompt constructed with context length: {len(self.user_representation)} chars")

            # Create a properly formatted message using the client
            message = self.client.create_message("user", prompt)
            
            # Generate the response
            print(f"[DIALECTIC] Calling model for generation")
            model_start = asyncio.get_event_loop().time()
            response = await self.client.generate(
                messages=[message],
                system=self.system_prompt,
                max_tokens=300
            )
            model_time = asyncio.get_event_loop().time() - model_start
            print(f"[DIALECTIC] Model response received in {model_time:.2f}s: {len(response)} chars")
            
            total_time = asyncio.get_event_loop().time() - call_start
            print(f"[DIALECTIC] call() completed in {total_time:.2f}s")
            return [{"text": response}]

    @ai_track("Dialectic Call")
    @observe(as_type="generation")
    async def stream(self):
        with sentry_sdk.start_transaction(
            op="dialectic-inference", name="Dialectic API Response"
        ):
            print(f"[DIALECTIC] Starting stream() method with query length: {len(self.agent_input)}")
            stream_start = asyncio.get_event_loop().time()
            
            prompt = f"""
            <query>{self.agent_input}</query>
            <context>{self.user_representation}</context>
            <conversation_history>{self.chat_history}</conversation_history> 
            """
            print(f"[DIALECTIC] Prompt constructed with context length: {len(self.user_representation)} chars")
            
            # Create a properly formatted message using the client
            message = self.client.create_message("user", prompt)
            
            # Stream the response
            print(f"[DIALECTIC] Calling model for streaming")
            model_start = asyncio.get_event_loop().time()
            stream = await self.client.stream(
                messages=[message],
                system=self.system_prompt,
                max_tokens=150
            )
            stream_setup_time = asyncio.get_event_loop().time() - model_start
            print(f"[DIALECTIC] Stream started in {stream_setup_time:.2f}s")
            
            total_time = asyncio.get_event_loop().time() - stream_start
            print(f"[DIALECTIC] stream() setup completed in {total_time:.2f}s")
            return stream


async def get_chat_history(app_id: str, user_id: str, session_id: str) -> str:
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
    
    print(f"[AGENT] Received query: {final_query} for session {session_id}")
    print(f"[AGENT] Starting on-demand user representation generation")
    
    start_time = asyncio.get_event_loop().time()

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
        print(f"[AGENT] Created embedding store with collection_id: {collection.public_id if collection else None}")
        
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
        print(f"[AGENT] Latest user message ID: {latest_message_id}")
        
        # Get chat history for the session
        history = await get_chat_history(app_id, user_id, session_id)

        # Run both long-term and short-term context retrieval concurrently
        print(f"[AGENT] Starting parallel tasks for context retrieval")
        long_term_task = get_long_term_facts(final_query, embedding_store)
        short_term_task = run_tom_inference(history, session_id)
        
        # Wait for both tasks to complete
        facts, tom_inference = await asyncio.gather(
            long_term_task, 
            short_term_task
        )
        print(f"[AGENT] Retrieved {len(facts)} facts from long-term memory")
        print(f"[AGENT] TOM inference completed with {len(tom_inference)} characters")
        
        # Generate a fresh user representation
        print(f"[AGENT] Generating user representation")
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
        print(f"[AGENT] User representation generated: {len(user_representation)} characters")

    # Create a Dialectic chain with the fresh user representation
    chain = Dialectic(
        agent_input=final_query,
        user_representation=user_representation,
        chat_history=history,
    )
    
    generation_time = asyncio.get_event_loop().time() - start_time
    print(f"[AGENT] User representation generation completed in {generation_time:.2f}s")

    langfuse_context.update_current_trace(
        session_id=session_id,
        user_id=user_id,
        release=os.getenv("SENTRY_RELEASE"),
        metadata={"environment": os.getenv("SENTRY_ENVIRONMENT")},
    )

    # Use streaming or non-streaming response based on the request
    print(f"[AGENT] Calling Dialectic with streaming={stream}")
    query_start_time = asyncio.get_event_loop().time()
    if stream:
        response_stream = await chain.stream()
        print(f"[AGENT] Dialectic stream started after {asyncio.get_event_loop().time() - query_start_time:.2f}s")
        return response_stream
    
    response = await chain.call()
    query_time = asyncio.get_event_loop().time() - query_start_time
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"[AGENT] Dialectic response received in {query_time:.2f}s (total: {total_time:.2f}s)")
    return schemas.AgentChat(content=response[0]["text"])


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
    print(f"[FACTS] Starting fact retrieval for query: {query}")
    fact_start_time = asyncio.get_event_loop().time()
    
    # Generate multiple queries for the semantic search
    print(f"[FACTS] Generating semantic queries")
    search_queries = await generate_semantic_queries(query)
    print(f"[FACTS] Generated {len(search_queries)} semantic queries: {search_queries}")
    
    # Retrieve relevant facts using multiple queries
    retrieved_facts = set()
    for i, search_query in enumerate(search_queries):
        print(f"[FACTS] Searching for query {i+1}/{len(search_queries)}: {search_query}")
        query_start = asyncio.get_event_loop().time()
        facts = await embedding_store.get_relevant_facts(
            search_query, 
            top_k=20, 
            max_distance=0.85
        )
        query_time = asyncio.get_event_loop().time() - query_start
        print(f"[FACTS] Query {i+1} retrieved {len(facts)} facts in {query_time:.2f}s")
        retrieved_facts.update(facts)
    
    total_time = asyncio.get_event_loop().time() - fact_start_time
    print(f"[FACTS] Total fact retrieval completed in {total_time:.2f}s with {len(retrieved_facts)} unique facts")
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
    print(f"[TOM] Running ToM inference for session {session_id}")
    tom_start_time = asyncio.get_event_loop().time()
    
    # Get chat history length to determine if this is a new conversation
    async with SessionLocal() as db:
        # Count messages in this session
        messages_count_stmt = (
            select(func.count())
            .select_from(models.Message)
            .where(models.Message.session_id == session_id)
        )
        result = await db.execute(messages_count_stmt)
        message_count = result.scalar_one()
        
        print(f"[TOM] Session has {message_count} messages in the database")
        
        if message_count <= 1:
            print(f"[TOM] This is a new conversation, using empty user representation")
            # For new conversations, use an empty string as user_representation
            # to avoid inheriting from previous sessions
            tom_inference_response = await get_tom_inference(
                chat_history, 
                session_id, 
                method="single_prompt",
                user_representation=""
            )
        else:
            print(f"[TOM] This is an existing conversation, using normal inference flow")
            tom_inference_response = await get_tom_inference(
                chat_history, 
                session_id, 
                method="single_prompt"
            )
    
    # Extract the prediction from the response
    prediction = parse_xml_content(tom_inference_response, "prediction")
    tom_time = asyncio.get_event_loop().time() - tom_start_time
    
    print(f"[TOM] ToM inference completed in {tom_time:.2f}s")
    print(f"[TOM] Prediction length: {len(prediction)} characters")
    
    return prediction


async def generate_semantic_queries(query: str) -> List[str]:
    """
    Generate multiple semantically relevant queries based on the original query using LLM.
    This helps retrieve more diverse and relevant facts from the vector store.
    
    Args:
        query: The original dialectic query
        
    Returns:
        A list of semantically relevant queries
    """
    print(f"[SEMANTIC] Generating semantic queries from: {query}")
    query_start = asyncio.get_event_loop().time()
    
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

    print(f"[SEMANTIC] Calling LLM for query generation")
    llm_start = asyncio.get_event_loop().time()
    # Note: get_response is async, so we need to await it
    queries_response = await get_response(
        [{"role": "user", "content": query_prompt}],
        DEF_PROVIDER,
        DEF_ANTHROPIC_MODEL
    )
    llm_time = asyncio.get_event_loop().time() - llm_start
    print(f"[SEMANTIC] LLM response received in {llm_time:.2f}s: {queries_response[:100]}...")
    
    # Parse the JSON response to get a list of queries
    try:
        queries = json.loads(queries_response)
        if not isinstance(queries, list):
            # Fallback if response is not a valid list
            print(f"[SEMANTIC] LLM response not a list, using as single query")
            queries = [queries_response]
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        print(f"[SEMANTIC] Failed to parse JSON response, using raw response as query")
        queries = [query]  # Fall back to the original query
    
    # Ensure we always include the original query
    if query not in queries:
        print(f"[SEMANTIC] Adding original query to results")
        queries.append(query)
    
    total_time = asyncio.get_event_loop().time() - query_start
    print(f"[SEMANTIC] Generated {len(queries)} queries in {total_time:.2f}s")
    
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
    print(f"[REPRESENTATION] Starting user representation generation")
    rep_start_time = asyncio.get_event_loop().time()
    
    # Fetch the latest user representation from the same session
    print(f"[REPRESENTATION] Fetching latest representation for session {session_id}")
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
    print(f"[REPRESENTATION] Found previous representation: {len(latest_representation)} characters")
    print(f"[REPRESENTATION] Using {len(facts)} facts for representation")
    
    # Generate the new user representation
    print(f"[REPRESENTATION] Calling get_user_representation")
    gen_start_time = asyncio.get_event_loop().time()
    user_representation_response = await get_user_representation(
        chat_history=chat_history,
        session_id=session_id,
        user_representation=latest_representation,
        tom_inference=tom_inference,
        method="long_term",
        this_turn_facts=facts,
        embedding_store=embedding_store
    )
    gen_time = asyncio.get_event_loop().time() - gen_start_time
    print(f"[REPRESENTATION] get_user_representation completed in {gen_time:.2f}s")
    
    # Extract the representation from the response
    representation = parse_xml_content(user_representation_response, "representation")
    print(f"[REPRESENTATION] Extracted representation: {len(representation)} characters")
    
    # If message_id is provided, save the representation as a metamessage
    if message_id and representation:
        print(f"[REPRESENTATION] Saving representation to message_id: {message_id}")
        save_start = asyncio.get_event_loop().time()
        async with SessionLocal() as save_db:
            metamessage = models.Metamessage(
                message_id=message_id,
                metamessage_type="user_representation",
                content=representation,
                h_metadata={},
            )
            save_db.add(metamessage)
            await save_db.commit()
        save_time = asyncio.get_event_loop().time() - save_start
        print(f"[REPRESENTATION] Representation saved in {save_time:.2f}s")
    
    total_time = asyncio.get_event_loop().time() - rep_start_time
    print(f"[REPRESENTATION] Total representation generation completed in {total_time:.2f}s")
    return representation

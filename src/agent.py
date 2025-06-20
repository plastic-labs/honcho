import asyncio
import json
import logging
import os
from typing import Any, Optional

import sentry_sdk
from anthropic import MessageStreamManager
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.dependencies import tracked_db
from src.deriver.tom import get_tom_inference
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from src.deriver.tom.long_term import get_user_representation_long_term
from src.utils import history, parse_xml_content
from src.utils.model_client import ModelClient, ModelProvider

# Configure logging
logger = logging.getLogger(__name__)

DEF_DIALECTIC_PROVIDER = ModelProvider.ANTHROPIC
DEF_DIALECTIC_MODEL = "claude-3-7-sonnet-20250219"

DEF_QUERY_GENERATION_PROVIDER = ModelProvider.GROQ
DEF_QUERY_GENERATION_MODEL = "llama-3.1-8b-instant"
QUERY_GENERATION_SYSTEM = """Given this query about a user, generate 3 focused search queries that would help retrieve relevant facts about the user.
    Each query should focus on a specific aspect related to the original query, rephrased to maximize semantic search effectiveness.
    For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's favorite cuisine", etc.
    
    Format your response as a JSON array of strings, with each string being a search query. 
    Respond only in valid JSON, without markdown formatting or quotes, and nothing else.
    Example:
    ["query about interests", "query about personality", "query about experiences"]"""

load_dotenv()


class Dialectic:
    def __init__(self, agent_input: str, user_representation: str, chat_history: str):
        self.agent_input = agent_input
        self.user_representation = user_representation
        self.chat_history = chat_history
        self.client = ModelClient(
            provider=DEF_DIALECTIC_PROVIDER, model=DEF_DIALECTIC_MODEL
        )
        self.system_prompt = """You are operating as a context service that helps maintain psychological understanding of users across applications. Alongside a query, you'll receive: 1) previously collected psychological context about the user that I've maintained, 2) a series of long-term facts about the user, and 3) their current conversation/interaction from the requesting application. Your goal is to analyze this information and provide theory-of-mind insights that help applications personalize their responses.  Please respond in a brief, matter-of-fact, and appropriate manner to convey as much relevant information to the application based on its query and the user's most recent message. You are encouraged to provide any context from the provided resources that helps provide a more complete or nuanced understanding of the user, as long as it is somewhat relevant to the query. If the context provided doesn't help address the query, write absolutely NOTHING but "None"."""

    @ai_track("Dialectic Call")
    @observe()
    async def call(self):
        with sentry_sdk.start_transaction(
            op="dialectic-inference", name="Dialectic API Response"
        ):
            logger.debug(
                f"Starting call() method with query length: {len(self.agent_input)}"
            )
            call_start = asyncio.get_event_loop().time()

            prompt = f"""
            <query>{self.agent_input}</query>
            <context>{self.user_representation}</context>
            <conversation_history>{self.chat_history}</conversation_history>
            """
            logger.debug(
                f"Prompt constructed with context length: {len(self.user_representation)} chars"
            )

            # Create a properly formatted message
            message: dict[str, Any] = {"role": "user", "content": prompt}

            # Generate the response
            logger.debug("Calling model for generation")
            model_start = asyncio.get_event_loop().time()
            response = await self.client.generate(
                messages=[message], system=self.system_prompt, max_tokens=1000
            )
            model_time = asyncio.get_event_loop().time() - model_start
            logger.debug(
                f"Model response received in {model_time:.2f}s: {len(response)} chars"
            )

            total_time = asyncio.get_event_loop().time() - call_start
            logger.debug(f"call() completed in {total_time:.2f}s")
            return [{"text": response}]

    @ai_track("Dialectic Call")
    @observe()
    async def stream(self):
        with sentry_sdk.start_transaction(
            op="dialectic-inference", name="Dialectic API Response"
        ):
            logger.debug(
                f"Starting stream() method with query length: {len(self.agent_input)}"
            )
            stream_start = asyncio.get_event_loop().time()

            prompt = f"""
            <query>{self.agent_input}</query>
            <context>{self.user_representation}</context>
            <conversation_history>{self.chat_history}</conversation_history> 
            """
            logger.debug(
                f"Prompt constructed with context length: {len(self.user_representation)} chars"
            )

            # Create a properly formatted message
            message: dict[str, Any] = {"role": "user", "content": prompt}

            # Stream the response
            logger.debug("Calling model for streaming")
            model_start = asyncio.get_event_loop().time()
            stream = await self.client.stream(
                messages=[message], system=self.system_prompt, max_tokens=1000
            )

            stream_setup_time = asyncio.get_event_loop().time() - model_start
            logger.debug(f"Stream started in {stream_setup_time:.2f}s")

            total_time = asyncio.get_event_loop().time() - stream_start
            logger.debug(f"stream() setup completed in {total_time:.2f}s")
            return stream


@observe()
async def chat(
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
    queries: str | list[str],
    stream: bool = False,
) -> schemas.DialecticResponse | MessageStreamManager:
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

    Args:
        workspace_name: The workspace name
        peer_name: The peer name
        session_name: The session name. If None, this queries the global representation.
        queries: The queries to ask the Dialectic API
        stream: Whether to stream the response

    Returns:
        Either a string or a stream of messages from the LLM provider, depending on the stream flag
    """
    # Format the query string
    questions = [queries] if isinstance(queries, str) else queries
    final_query = "\n".join(questions) if len(questions) > 1 else questions[0]

    logger.debug(f"Received query: {final_query} for session {session_name}")
    logger.debug("Starting on-demand user representation generation")

    start_time = asyncio.get_event_loop().time()

    # Setup phase - create resources we'll need for all operations

    # 1. Fetch latest peer message & chat history
    async with tracked_db("chat.load_history") as db_history:
        stmt = (
            select(models.Message)
            .where(models.Message.workspace_name == workspace_name)
            .where(models.Message.peer_name == peer_name)
            .order_by(models.Message.id.desc())
            .limit(1)
        )
        if session_name:
            stmt = stmt.where(models.Message.session_name == session_name)
        latest_messages = await db_history.execute(stmt)
        latest_message = latest_messages.scalar_one_or_none()
        latest_message_id = latest_message.public_id if latest_message else None
        if session_name:
            chat_history = await history.get_summarized_history(
                db_history,
                workspace_name,
                session_name,
                peer_name,
                summary_type=history.SummaryType.SHORT,
            )
            if not chat_history:
                logger.warning(f"No chat history found for session {session_name}")
                chat_history = (
                    f"someone asked this about the user's message: {final_query}"
                )
            logger.debug(
                f"Workspace: {workspace_name}, Peer: {peer_name}, Session: {session_name}"
            )
        else:
            chat_history = ""
        logger.debug("Retrieved chat history: %s lines", len(chat_history.split("\n")))

    # Run short-term inference and long-term facts in parallel
    async def fetch_long_term():
        async with tracked_db("chat.get_collection") as db_embed:
            name = "global_representation" if session_name is None else ""
            collection = await crud.get_or_create_collection(
                db_embed, workspace_name, peer_name, collection_name=name
            )
            collection_name = collection.name  # Extract the ID while session is active
        facts = await get_long_term_facts(
            final_query, workspace_name, peer_name, collection_name
        )
        return facts

    long_term_task = asyncio.create_task(fetch_long_term())
    short_term_task = asyncio.create_task(run_tom_inference(chat_history))

    facts, tom_inference = await asyncio.gather(long_term_task, short_term_task)
    logger.debug(f"Retrieved {len(facts)} facts from long-term memory")
    logger.debug(f"TOM inference completed with {len(tom_inference)} characters")

    # Generate a fresh user representation
    logger.debug("Generating user representation")
    async with tracked_db("chat.generate_user_representation") as db_rep:
        user_representation = await generate_user_representation(
            workspace_name,
            peer_name,
            session_name,
            chat_history=chat_history,
            tom_inference=tom_inference,
            facts=facts,
            db=db_rep,
            message_id=latest_message_id,
            with_inference=False,
        )
    logger.debug(
        f"User representation generated: {len(user_representation)} characters"
    )

    # Create a Dialectic chain with the fresh user representation
    chain = Dialectic(
        agent_input=final_query,
        user_representation=user_representation,
        chat_history=chat_history,
    )

    generation_time = asyncio.get_event_loop().time() - start_time
    logger.debug(f"User representation generation completed in {generation_time:.2f}s")

    langfuse_context.update_current_trace(
        session_id=session_name,
        user_id=peer_name,
        release=os.getenv("SENTRY_RELEASE"),
        metadata={"environment": os.getenv("SENTRY_ENVIRONMENT")},
    )

    # Use streaming or non-streaming response based on the request
    logger.debug(f"Calling Dialectic with streaming={stream}")
    query_start_time = asyncio.get_event_loop().time()
    if stream:
        response_stream = await chain.stream()
        logger.debug(
            f"Dialectic stream started after {asyncio.get_event_loop().time() - query_start_time:.2f}s"
        )
        return response_stream

    response = await chain.call()
    query_time = asyncio.get_event_loop().time() - query_start_time
    total_time = asyncio.get_event_loop().time() - start_time
    logger.debug(
        f"Dialectic response received in {query_time:.2f}s (total: {total_time:.2f}s)"
    )
    return schemas.DialecticResponse(content=response[0]["text"])


async def get_long_term_facts(
    query: str,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
) -> list[str]:
    """
    Generate queries based on the dialectic query and retrieve relevant facts.

    Args:
        query: The user query
        workspace_name: The workspace name
        peer_name: The peer name
        collection_name: The collection name

    Returns:
        List of retrieved facts
    """
    logger.debug(f"Starting fact retrieval for query: {query}")
    fact_start_time = asyncio.get_event_loop().time()

    # Generate multiple queries for the semantic search
    logger.debug("Generating semantic queries")
    search_queries = await generate_semantic_queries(query)
    logger.debug(f"Generated {len(search_queries)} semantic queries: {search_queries}")

    # Create a list of coroutines, one for each query
    async def execute_query(i: int, search_query: str) -> list[str]:
        logger.debug(f"Starting query {i + 1}/{len(search_queries)}: {search_query}")
        query_start = asyncio.get_event_loop().time()
        query_embedding_store = CollectionEmbeddingStore(
            workspace_name=workspace_name,
            peer_name=peer_name,
            collection_name=collection_name,
        )
        facts = await query_embedding_store.get_relevant_facts(
            search_query, top_k=10, max_distance=0.85
        )
        query_time = asyncio.get_event_loop().time() - query_start
        logger.debug(f"Query {i + 1} retrieved {len(facts)} facts in {query_time:.2f}s")
        return facts

    # Execute all queries in parallel
    query_tasks = [
        execute_query(i, search_query) for i, search_query in enumerate(search_queries)
    ]
    all_facts_lists = await asyncio.gather(*query_tasks)

    # Combine all facts into a single set to remove duplicates
    retrieved_facts = set()
    for facts in all_facts_lists:
        retrieved_facts.update(facts)

    total_time = asyncio.get_event_loop().time() - fact_start_time
    logger.debug(
        f"Total fact retrieval completed in {total_time:.2f}s with {len(retrieved_facts)} unique facts"
    )
    return list(retrieved_facts)


async def run_tom_inference(chat_history: str) -> str:
    """
    Run ToM inference on chat history.

    Args:
        chat_history: The chat history

    Returns:
        The ToM inference
    """
    # Run ToM inference
    logger.debug("Running ToM inference")
    tom_start_time = asyncio.get_event_loop().time()

    # Get chat history length to determine if this is a new conversation
    tom_inference_response = await get_tom_inference(
        chat_history, user_representation="", method="single_prompt"
    )

    # Extract the prediction from the response
    tom_time = asyncio.get_event_loop().time() - tom_start_time

    logger.debug(f"ToM inference completed in {tom_time:.2f}s")
    prediction = parse_xml_content(tom_inference_response, "prediction")
    logger.debug(f"Prediction length: {len(prediction)} characters")

    return prediction


async def generate_semantic_queries(query: str) -> list[str]:
    """
    Generate multiple semantically relevant queries based on the original query using LLM.
    This helps retrieve more diverse and relevant facts from the vector store.

    Args:
        query: The original dialectic query

    Returns:
        A list of semantically relevant queries
    """
    logger.debug(f"Generating semantic queries from: {query}")
    query_start = asyncio.get_event_loop().time()

    logger.debug("Calling LLM for query generation")
    llm_start = asyncio.get_event_loop().time()

    # Create a new model client
    client = ModelClient(
        provider=DEF_QUERY_GENERATION_PROVIDER, model=DEF_QUERY_GENERATION_MODEL
    )

    # Prepare the messages for Anthropic
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

    # Generate the response
    try:
        result = await client.generate(
            messages=messages,
            system=QUERY_GENERATION_SYSTEM,
            max_tokens=1000,
            use_caching=True,  # Likely not caching because the system prompt is under 1000 tokens
        )
        llm_time = asyncio.get_event_loop().time() - llm_start
        logger.debug(f"LLM response received in {llm_time:.2f}s: {result[:100]}...")

        # Parse the JSON response to get a list of queries
        try:
            queries = json.loads(result)
            if not isinstance(queries, list):
                # Fallback if response is not a valid list
                logger.debug("LLM response not a list, using as single query")
                queries = [result]
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            logger.debug("Failed to parse JSON response, using raw response as query")
            queries = [query]  # Fall back to the original query

        # Ensure we always include the original query
        if query not in queries:
            logger.debug("Adding original query to results")
            queries.append(query)

        total_time = asyncio.get_event_loop().time() - query_start
        logger.debug(f"Generated {len(queries)} queries in {total_time:.2f}s")

        return queries
    except Exception as e:
        logger.error(f"Error during API call: {str(e)}")
        raise


async def generate_user_representation(
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
    chat_history: str,
    tom_inference: str,
    facts: list[str],
    db: AsyncSession,
    message_id: Optional[str] = None,
    with_inference: bool = False,
) -> str:
    """
    Generate a user representation by combining long-term facts and short-term context.
    Save it to peer metadata if no session is provided (global-level), or save it to
    session-peers table metadata if a session is provided (local-level).
    If session-level, uses existing representations from the same session for continuity.

    Returns:
        The generated user representation.
    """
    logger.debug("Starting user representation generation")
    rep_start_time = asyncio.get_event_loop().time()

    if with_inference:
        latest_representation = await crud.get_working_representation(
            db, workspace_name, peer_name, session_name
        )

        logger.debug(
            f"Found previous representation: {len(latest_representation)} characters"
        )
        logger.debug(f"Using {len(facts)} facts for representation")

        # Generate the new user representation
        logger.debug("Calling get_user_representation")
        gen_start_time = asyncio.get_event_loop().time()
        user_representation_response = await get_user_representation_long_term(
            chat_history=chat_history,
            facts=facts,
            user_representation=latest_representation,
            tom_inference=tom_inference,
        )
        gen_time = asyncio.get_event_loop().time() - gen_start_time
        logger.debug(f"get_user_representation completed in {gen_time:.2f}s")

        # Extract the representation from the response
        representation = parse_xml_content(
            user_representation_response, "representation"
        )
        logger.debug(f"Extracted representation: {len(representation)} characters")
    else:
        representation = f"""
PREDICTION ABOUT THE USER'S CURRENT MENTAL STATE:
{tom_inference}

RELEVANT LONG-TERM FACTS ABOUT THE USER:
{facts}
"""
    logger.debug(f"Representation: {representation}")
    # If message_id is provided, save the representation as metadata
    if not representation:
        logger.debug("Empty representation, skipping save")
    elif not message_id:
        logger.debug("No message_id, skipping save")
    else:
        logger.debug(f"Saving representation to message_id: {message_id}")
        save_start = asyncio.get_event_loop().time()
        try:
            await crud.set_working_representation(
                db,
                representation,
                workspace_name,
                peer_name,
                session_name,
            )
            save_time = asyncio.get_event_loop().time() - save_start
            logger.debug(f"Representation saved in {save_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during save DB operation: {str(e)}")
            await db.rollback()

    total_time = asyncio.get_event_loop().time() - rep_start_time
    logger.debug(f"Total representation generation completed in {total_time:.2f}s")
    return representation

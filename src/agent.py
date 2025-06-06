import asyncio
import json
import logging
import os
from collections.abc import Iterable
from typing import Any, Optional
from inspect import cleandoc as c


from dotenv import load_dotenv
from sentry_sdk.ai.monitoring import ai_track
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.dependencies import tracked_db
from src.deriver.tom import get_tom_inference
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from src.deriver.tom.long_term import get_user_representation_long_term
from src.utils import history, parse_xml_content

from mirascope import llm, prompt_template
from mirascope.integrations.langfuse import with_langfuse

# Configure logging
logger = logging.getLogger(__name__)

USER_REPRESENTATION_METAMESSAGE_TYPE = "honcho_user_representation"


load_dotenv()


@prompt_template()
def dialectic_prompt(query: str, user_representation: str, chat_history: str) -> str:
    return c(
        f"""
        You are operating as a context service that helps maintain psychological understanding of users across applications. Alongside a query, you'll receive:

        1. previously collected psychological context about the user that I've maintained, 
        2. a series of long-term facts about the user, and 
        3. their current conversation/interaction from the requesting application. 

        Your goal is to analyze this information and provide theory-of-mind insights that help applications personalize their responses.  Please respond in a brief, matter-of-fact, and appropriate manner to convey as much relevant information to the application based on its query and the user's most recent message. You are encouraged to provide any context from the provided resources that helps provide a more complete or nuanced understanding of the user, as long as it is somewhat relevant to the query. If the context provided doesn't help address the query, write absolutely NOTHING but "None".

        <query>{query}</query>
        <context>{user_representation}</context>
        <conversation_history>{chat_history}</conversation_history>        
        """
    )


@ai_track("Dialectic Call")
@with_langfuse()
@llm.call(provider="anthropic", model="claude-3-7-sonnet-20250219")
async def dialectic_call(query: str, user_representation: str, chat_history: str):
    return dialectic_prompt(query, user_representation, chat_history)


@ai_track("Dialectic Stream")
@with_langfuse()
@llm.call(provider="anthropic", model="claude-3-7-sonnet-20250219", stream=True)
async def dialectic_stream(query: str, user_representation: str, chat_history: str):
    return dialectic_prompt(query, user_representation, chat_history)


async def chat(
    app_id: str,
    user_id: str,
    session_id: str,
    queries: str | list[str],
    stream: bool = False,
) -> llm.Stream | llm.CallResponse:
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
    questions = [queries] if isinstance(queries, str) else queries
    final_query = "\n".join(questions) if len(questions) > 1 else questions[0]

    logger.debug(f"Received query: {final_query} for session {session_id}")
    logger.debug("Starting on-demand user representation generation")

    start_time = asyncio.get_event_loop().time()

    # Setup phase - create resources we'll need for all operations

    # 1. Fetch latest user message & chat history
    async with tracked_db("chat.load_history") as db_history:
        stmt = (
            select(models.Message)
            .where(models.Message.app_id == app_id)
            .where(models.Message.user_id == user_id)
            .where(models.Message.session_id == session_id)
            .where(models.Message.is_user)
            .order_by(models.Message.id.desc())
            .limit(1)
        )
        latest_messages = await db_history.execute(stmt)
        latest_message = latest_messages.scalar_one_or_none()
        latest_message_id = latest_message.public_id if latest_message else None
        logger.debug(f"Latest user message ID: {latest_message_id}")

        chat_history, _, _ = await history.get_summarized_history(
            db_history, session_id, summary_type=history.SummaryType.SHORT
        )
        if not chat_history:
            logger.warning(f"No chat history found for session {session_id}")
            chat_history = f"someone asked this about the user's message: {final_query}"
        logger.debug(f"IDs: {app_id}, {user_id}, {session_id}")
        message_count = len(chat_history.split("\n"))
        logger.debug(f"Retrieved chat history: {message_count} messages")

    # Run short-term inference and long-term facts in parallel
    async def fetch_long_term():
        async with tracked_db("chat.get_collection") as db_embed:
            collection = await crud.get_or_create_user_protected_collection(
                db_embed, app_id, user_id
            )
            collection_id = (
                collection.public_id
            )  # Extract the ID while session is active
        facts = await get_long_term_facts(final_query, app_id, user_id, collection_id)
        return facts

    long_term_task = asyncio.create_task(fetch_long_term())
    short_term_task = asyncio.create_task(run_tom_inference(chat_history, session_id))

    facts, tom_inference = await asyncio.gather(long_term_task, short_term_task)
    logger.debug(f"Retrieved {len(facts)} facts from long-term memory")
    logger.debug(f"TOM inference completed with {len(tom_inference)} characters")

    # Generate a fresh user representation
    logger.debug("Generating user representation")
    async with tracked_db("chat.generate_user_representation") as db_rep:
        user_representation = await generate_user_representation(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
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

    generation_time = asyncio.get_event_loop().time() - start_time
    logger.debug(f"User representation generation completed in {generation_time:.2f}s")

    if stream:
        logger.debug("Calling Dialectic with streaming")
        response = await dialectic_stream(
            final_query, user_representation, chat_history
        )
    else:
        logger.debug("Calling Dialectic with non-streaming")
        query_start_time = asyncio.get_event_loop().time()
        response = await dialectic_call(final_query, user_representation, chat_history)
        query_time = asyncio.get_event_loop().time() - query_start_time
        logger.debug(f"Dialectic response received in {query_time:.2f}s")
    return response


async def get_long_term_facts(
    query: str, app_id: str, user_id: str, collection_id: str
) -> list[str]:
    """
    Generate queries based on the dialectic query and retrieve relevant facts.

    Args:
        query: The user query
        embedding_store: The embedding store to search

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
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
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


async def run_tom_inference(chat_history: str, session_id: str) -> str:
    """
    Run ToM inference on chat history.

    Args:
        chat_history: The chat history
        session_id: The session ID

    Returns:
        The ToM inference
    """
    # Run ToM inference
    logger.debug(f"Running ToM inference for session {session_id}")
    tom_start_time = asyncio.get_event_loop().time()

    # Get chat history length to determine if this is a new conversation
    tom_inference_response = await get_tom_inference(
        chat_history, session_id, method="single_prompt", user_representation=""
    )

    # Extract the prediction from the response
    tom_time = asyncio.get_event_loop().time() - tom_start_time

    logger.debug(f"ToM inference completed in {tom_time:.2f}s")
    prediction = parse_xml_content(tom_inference_response, "prediction")
    logger.debug(f"Prediction length: {len(prediction)} characters")

    return prediction


@with_langfuse()
@llm.call(provider="groq", model="llama-3.1-8b-instant", response_model=list[str])
async def generate_semantic_queries(query: str):
    return c(
        f"""
        Given this query about a user, generate 3 focused search queries that would help retrieve relevant facts about the user.
        Each query should focus on a specific aspect related to the original query, rephrased to maximize semantic search effectiveness.
        For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's favorite cuisine", etc.
        
        Format your response as a list of strings, with each string being a search query.
        Example:
        ["some query about interests", "some query about personality", "some query about experiences"]

        <query>{query}</query>
        """
    )


async def generate_user_representation(
    app_id: str,
    user_id: str,
    session_id: str,
    chat_history: str,
    tom_inference: str,
    facts: list[str],
    db: AsyncSession,
    message_id: Optional[str] = None,
    with_inference: bool = False,
) -> str:
    """
    Generate a user representation by combining long-term facts and short-term context.
    Optionally save it as a metamessage if message_id is provided.
    Only uses existing representations from the same session for continuity.

    Returns:
        The generated user representation.
    """
    logger.debug("Starting user representation generation")
    rep_start_time = asyncio.get_event_loop().time()

    if with_inference:
        # Fetch the latest user representation from the same session
        logger.debug(f"Fetching latest representation for session {session_id}")
        latest_representation_stmt = (
            select(models.Metamessage)
            .where(
                models.Metamessage.session_id == session_id
            )  # only from the same session
            .where(models.Metamessage.label == USER_REPRESENTATION_METAMESSAGE_TYPE)
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
        logger.debug(
            f"Found previous representation: {len(latest_representation)} characters"
        )
        logger.debug(f"Using {len(facts)} facts for representation")

        # Generate the new user representation
        logger.debug("Calling get_user_representation")
        gen_start_time = asyncio.get_event_loop().time()
        user_representation_response = await get_user_representation_long_term(
            chat_history=chat_history,
            session_id=session_id,
            facts=facts,
            user_representation=latest_representation,
            tom_inference=tom_inference,
        )
        gen_time = asyncio.get_event_loop().time() - gen_start_time
        logger.debug(f"get_user_representation completed in {gen_time:.2f}s")

        # Extract the representation from the response
    else:
        representation = f"""
PREDICTION ABOUT THE USER'S CURRENT MENTAL STATE:
{tom_inference}

RELEVANT LONG-TERM FACTS ABOUT THE USER:
{facts}
"""
    logger.debug(f"Representation: {representation}")
    # If message_id is provided, save the representation as a metamessage
    if not representation:
        logger.debug("Empty representation, skipping save")
    else:
        logger.debug(f"Saving representation to message_id: {message_id}")
        save_start = asyncio.get_event_loop().time()
        try:
            # First check if message exists
            message_check_stmt = select(models.Message).where(
                models.Message.public_id == message_id
            )
            message_check = await db.execute(message_check_stmt)
            message_exists = message_check.scalar_one_or_none() is not None

            if not message_exists:
                message_id = None
            else:
                metamessage = models.Metamessage(
                    app_id=app_id,
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id if message_id else None,
                    label=USER_REPRESENTATION_METAMESSAGE_TYPE,
                    content=representation,
                    h_metadata={},
                )
                db.add(metamessage)
                await db.commit()
                save_time = asyncio.get_event_loop().time() - save_start
                logger.debug(f"Representation saved in {save_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during save DB operation: {str(e)}")
            await db.rollback()

    total_time = asyncio.get_event_loop().time() - rep_start_time
    logger.debug(f"Total representation generation completed in {total_time:.2f}s")
    return representation

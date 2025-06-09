import asyncio
import datetime
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

        Your goal is to analyze this information and synthesize these insights that help applications personalize their responses. Please respond in a brief, matter-of-fact, and appropriate manner to convey as much relevant information to the application based on its query and the user's most recent message. You are encouraged to synthesize the information based on three levels of reasoning: abduction, induction, and deduction. You might notice the context falling into these categories naturally, so feel free to start with the hypothesis (abduction), support it with observed patterns (induction), and solidify with explicit facts (deduction). If the context provided doesn't help address the query, write absolutely NOTHING but "No information available". If you are provided in the query with an alternative way to indicate that there is no information available, please use that instead.

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
    Chat with the Dialectic API that builds on-demand user representations.

    This function:
    1. Expands the query to retrieve facts from the vector store
    2. Combines them to answer the query
    """

    # format the query string
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
            embedding_store = CollectionEmbeddingStore(
                db=db_embed,
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
            )
            facts = await get_long_term_observations(final_query, embedding_store)
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
        return response
    else:
        logger.debug("Calling Dialectic with non-streaming")
        query_start_time = asyncio.get_event_loop().time()
        response = await dialectic_call(final_query, user_representation, chat_history)
        query_time = asyncio.get_event_loop().time() - query_start_time
        logger.debug(f"Dialectic response received in {query_time:.2f}s")
        return response


async def get_observations(
    query: str, embedding_store: CollectionEmbeddingStore
) -> str:
    """
    Generate queries based on the dialectic query and retrieve relevant observations.

    Uses both DIA framework contextualized observations and semantic search for comprehensive results.

    Args:
        query: The user query
        embedding_store: The embedding store to search

    Returns:
        String containing all retrieved observations with their context, organized by reasoning level
    """
    logger.info("=== OBSERVATION RETRIEVAL START ===")
    logger.info(f"ORIGINAL QUERY: {query}")
    observation_start_time = asyncio.get_event_loop().time()

    # First, get contextualized observations from DIA framework
    logger.info("RETRIEVING CONTEXTUALIZED OBSERVATIONS from DIA framework...")
    contextualized_observations = (
        await embedding_store.get_contextualized_observations_for_dialectic()
    )

    # Log contextualized observations by level
    for level, observations in contextualized_observations.items():
        logger.info(f"CONTEXTUALIZED {level.upper()}: {len(observations)} observations")
        for i, observation in enumerate(observations[:3]):  # Show first 3
            logger.info(f"  {i+1}. {observation[:100]}...")
        if len(observations) > 3:
            logger.info(f"  ... and {len(observations) - 3} more observations")

    # Generate multiple queries for additional semantic search
    logger.info("GENERATING SEMANTIC QUERIES for additional observations...")
    search_queries = await generate_semantic_queries(query)
    logger.info(f"GENERATED QUERIES: {search_queries}")

    # Create a list of coroutines, one for each query
    async def execute_query(i: int, search_query: str) -> list[tuple[str, str]]:
        logger.info(
            f"EXECUTING SEMANTIC QUERY {i + 1}/{len(search_queries)}: {search_query}"
        )
        query_start = asyncio.get_event_loop().time()
        documents = await embedding_store.get_relevant_observations(
            search_query,
            top_k=5,
            max_distance=0.85,  # Reduced from 10 since we have contextualized observations
        )
        query_time = asyncio.get_event_loop().time() - query_start
        logger.info(
            f"QUERY {i + 1} RESULTS: {len(documents)} observations retrieved in {query_time:.2f}s"
        )

        # Eagerly extract attributes to avoid DetachedInstanceError
        document_data = []
        for doc in documents:
            # Access attributes while the object is still bound to the session
            content = doc.content
            created_at = doc.created_at
            document_data.append((content, created_at))

        # Log the actual observations found using the extracted data
        for j, (content, created_at) in enumerate(document_data):
            logger.info(
                f"  {j+1}. [{created_at.strftime('%Y-%m-%d %H:%M:%S')}] {content}"
            )

        # Convert documents to tuples of (content, timestamp)
        return [
            (content, created_at.strftime("%Y-%m-%d-%H:%M:%S"))
            for content, created_at in document_data
        ]

    # Execute all queries in parallel
    query_tasks = [
        execute_query(i, search_query) for i, search_query in enumerate(search_queries)
    ]
    all_observations_lists = await asyncio.gather(*query_tasks)

    # Combine semantic search observations into a single set to remove duplicates
    semantic_observations = set()
    for observations in all_observations_lists:
        semantic_observations.update(observations)

    # Format the final response with structured sections
    response_parts = []

    # Add contextualized observations organized by reasoning level
    if any(contextualized_observations.values()):
        response_parts.append("=== REASONING-BASED USER UNDERSTANDING ===")

        if contextualized_observations.get("abductive"):
            response_parts.append("\n## ABDUCTIVE (High-level psychological insights):")
            response_parts.extend(contextualized_observations["abductive"])

        if contextualized_observations.get("inductive"):
            response_parts.append("\n## INDUCTIVE (Observed patterns and behaviors):")
            response_parts.extend(contextualized_observations["inductive"])

        if contextualized_observations.get("deductive"):
            response_parts.append(
                "\n## DEDUCTIVE (Explicit observations and statements):"
            )
            response_parts.extend(contextualized_observations["deductive"])

    # Add additional semantic search results if any
    if semantic_observations:
        response_parts.append("\n=== ADDITIONAL RELEVANT OBSERVATIONS ===")
        semantic_observations_formatted = [
            f"[created {timestamp}]: {observation}"
            for observation, timestamp in semantic_observations
        ]
        response_parts.extend(semantic_observations_formatted)

    observations_string = (
        "\n".join(response_parts)
        if response_parts
        else "No relevant observations found."
    )

    total_time = asyncio.get_event_loop().time() - observation_start_time
    total_contextualized = sum(
        len(observations) for observations in contextualized_observations.values()
    )

    logger.info("=== OBSERVATION RETRIEVAL SUMMARY ===")
    logger.info(f"TOTAL TIME: {total_time:.2f}s")
    logger.info(f"CONTEXTUALIZED OBSERVATIONS: {total_contextualized}")
    logger.info(f"SEMANTIC SEARCH OBSERVATIONS: {len(semantic_observations)}")
    logger.info(f"FINAL OBSERVATIONS STRING LENGTH: {len(observations_string)} chars")
    logger.info("=== OBSERVATION RETRIEVAL END ===")

    return observations_string


async def get_long_term_observations(
    query: str, embedding_store: CollectionEmbeddingStore
) -> list[str]:
    """
    Generate queries based on the dialectic query and retrieve relevant observations.

    Returns list of observations for compatibility with the tracked_db pattern.

    Args:
        query: The user query
        embedding_store: The embedding store to search

    Returns:
        List of retrieved observations
    """
    logger.debug(f"Starting long-term observation retrieval for query: {query}")

    # Use the existing get_observations function but return as list for compatibility
    observations_string = await get_observations(query, embedding_store)

    # Convert back to list format expected by generate_user_representation
    if observations_string:
        # Split by lines and filter out empty lines
        observations_list = [
            line.strip() for line in observations_string.split("\n") if line.strip()
        ]
        return observations_list
    else:
        return []


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


@prompt_template()
def query_generation_prompt(query: str) -> str:
    return c(
        f"""
        Given this query about a user, generate 3 focused search queries that would help retrieve relevant facts about the user. To ground your generation, each query should focus on one of the following levels of reasoning: abductive, inductive, and deductive.

        For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's observed eating patterns", "user's most recent meal", etc.
            
        Format your response as a JSON array of strings, with each string being a search query. 
        Respond only in valid JSON, without markdown formatting or quotes, and nothing else.
        Example:
        ["abductive query to retrieve hypotheses", "inductive query to retrieve observed patterns", "deductive query to retrieve explicit facts"]

        <query>{query}</query>
        """
    )


@with_langfuse()
@llm.call(provider="groq", model="llama-3.1-8b-instant", response_model=list[str])
async def generate_semantic_queries(query: str):
    return query_generation_prompt(query)


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
        representation = user_representation_response
    else:
        # Join the facts list with newlines for proper formatting
        facts_formatted = "\n".join(facts) if facts else "No relevant facts found."
        representation = f"""
PREDICTION ABOUT THE USER'S CURRENT MENTAL STATE:
{tom_inference}

RELEVANT LONG-TERM FACTS ABOUT THE USER:
{facts_formatted}
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


# Backward compatibility aliases
async def get_facts(query: str, embedding_store: CollectionEmbeddingStore) -> str:
    """Backward compatibility alias for get_observations."""
    return await get_observations(query, embedding_store)


async def get_long_term_facts(
    query: str, embedding_store: CollectionEmbeddingStore
) -> list[str]:
    """Backward compatibility alias for get_long_term_observations."""
    return await get_long_term_observations(query, embedding_store)

import asyncio
import logging
import os

from langfuse.decorators import langfuse_context, observe  # pyright: ignore
from mirascope import llm
from mirascope.integrations.langfuse import with_langfuse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dependencies import tracked_db
from src.deriver.tom import get_tom_inference
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from src.deriver.tom.long_term import get_user_representation_long_term
from src.deriver.tom.single_prompt import UserRepresentationOutput
from src.utils import history, parse_xml_content
from src.utils.clients import clients
from src.utils.types import track

# Configure logging
logger = logging.getLogger(__name__)


@track("Dialectic Call")
@with_langfuse()
@llm.call(
    provider=(
        settings.LLM.DIALECTIC_PROVIDER
        if settings.LLM.DIALECTIC_PROVIDER != "custom"
        else "openai"
    ),
    model=settings.LLM.DIALECTIC_MODEL,
    client=clients[settings.LLM.DIALECTIC_PROVIDER],
)
async def dialectic_call(
    query: str, working_representation: str, additional_context: str
):
    return f"""
You are operating as a context service that helps maintain psychological understanding of users across applications. Alongside a query, you'll receive: 1) previously collected psychological context about the user that I've maintained, 2) a series of long-term facts about the user, and 3) their current conversation/interaction from the requesting application. Your goal is to analyze this information and provide theory-of-mind insights that help applications personalize their responses.  Please respond in a brief, matter-of-fact, and appropriate manner to convey as much relevant information to the application based on its query and the user's most recent message. You are encouraged to provide any context from the provided resources that helps provide a more complete or nuanced understanding of the user, as long as it is somewhat relevant to the query. If the context provided doesn't help address the query, write absolutely NOTHING but "None".

<query>{query}</query>
<context>{working_representation}</context>
<conversation_history>{additional_context}</conversation_history>
"""


@track("Dialectic Stream")
@with_langfuse()
@llm.call(
    provider=(
        settings.LLM.DIALECTIC_PROVIDER
        if settings.LLM.DIALECTIC_PROVIDER != "custom"
        else "openai"
    ),
    model=settings.LLM.DIALECTIC_MODEL,
    stream=True,
    client=clients[settings.LLM.DIALECTIC_PROVIDER],
)
async def dialectic_stream(
    query: str, working_representation: str, additional_context: str
):
    return f"""
You are operating as a context service that helps maintain psychological understanding of users across applications. Alongside a query, you'll receive: 1) previously collected psychological context about the user that I've maintained, 2) a series of long-term facts about the user, and 3) their current conversation/interaction from the requesting application. Your goal is to analyze this information and provide theory-of-mind insights that help applications personalize their responses.  Please respond in a brief, matter-of-fact, and appropriate manner to convey as much relevant information to the application based on its query and the user's most recent message. You are encouraged to provide any context from the provided resources that helps provide a more complete or nuanced understanding of the user, as long as it is somewhat relevant to the query. If the context provided doesn't help address the query, write absolutely NOTHING but "None".

<query>{query}</query>
<context>{working_representation}</context>
<conversation_history>{additional_context}</conversation_history>
"""


class SemanticQueries(BaseModel):
    queries: list[str]


@with_langfuse()
@llm.call(
    provider=(
        settings.LLM.QUERY_GENERATION_PROVIDER
        if settings.LLM.QUERY_GENERATION_PROVIDER != "custom"
        else "openai"
    ),
    model=settings.LLM.QUERY_GENERATION_MODEL,
    response_model=SemanticQueries,
    client=clients[settings.LLM.QUERY_GENERATION_PROVIDER],
)
async def generate_semantic_queries_llm(query: str):
    return f"""
Given this query about a user, generate 3 focused search queries that would help retrieve relevant facts about the user. Each query should focus on a specific aspect related to the original query, rephrased to maximize semantic search effectiveness.
For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's favorite cuisine", etc.

Format your response as a JSON array of strings, with each string being a search query. 
Respond only in valid JSON, without markdown formatting or quotes, and nothing else.
Example:
["query about interests", "query about personality", "query about experiences"]

<query>{query}</query>
"""


@observe()
async def chat(
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
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

    # Call dialectic with enhanced context

    langfuse_context.update_current_trace(
        session_id=session_name,
        user_id=peer_name,
        release=os.getenv("SENTRY_RELEASE"),
        metadata={"environment": os.getenv("SENTRY_ENVIRONMENT")},
    )

    # Use streaming or non-streaming response based on the request
    logger.debug(f"Calling Dialectic with streaming={stream}")
    if stream:
        logger.debug("Calling Dialectic with streaming")
        response = await dialectic_stream(
            final_query, user_representation, chat_history
        )
        return response
    else:
        logger.debug("Calling Dialectic with non-streaming")
        response = await dialectic_call(final_query, user_representation, chat_history)
        return response


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
            search_query,
            top_k=settings.AGENT.SEMANTIC_SEARCH_TOP_K,
            max_distance=settings.AGENT.SEMANTIC_SEARCH_MAX_DISTANCE,
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
    retrieved_facts: set[str] = set()
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
        chat_history,
        user_representation="",
        method=settings.AGENT.TOM_INFERENCE_METHOD,
    )

    # Extract the prediction from the response
    tom_time = asyncio.get_event_loop().time() - tom_start_time

    logger.debug(f"ToM inference completed in {tom_time:.2f}s")

    # Create a prediction summary from the structured Pydantic object
    prediction = (
        f"Current context: {tom_inference_response.current_state.immediate_context}"
    )
    if tom_inference_response.tentative_inferences:
        prediction += f"\nKey inferences: {', '.join([inf.interpretation for inf in tom_inference_response.tentative_inferences[:3]])}"

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
    try:
        queries_result = await generate_semantic_queries_llm(query)
        queries = queries_result.queries

        # Ensure we always include the original query
        if query not in queries:
            logger.debug("Adding original query to results")
            queries.append(query)

        total_time = asyncio.get_event_loop().time() - query_start
        logger.debug(f"Generated {len(queries)} queries in {total_time:.2f}s")

        return queries
    except Exception as e:
        logger.error(f"Error during query generation: {str(e)}")
        return [query]  # Fallback to original query


async def generate_user_representation(
    workspace_name: str,
    peer_name: str,
    session_name: str | None,
    chat_history: str,
    tom_inference: str,
    facts: list[str],
    db: AsyncSession,
    message_id: str | None = None,
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
        user_representation_response: UserRepresentationOutput = (
            await get_user_representation_long_term(
                chat_history=chat_history,
                facts=facts,
                user_representation=latest_representation,
                tom_inference=tom_inference,
            )
        )
        gen_time = asyncio.get_event_loop().time() - gen_start_time
        logger.debug(f"get_user_representation completed in {gen_time:.2f}s")

        # Extract the representation from the response
        if hasattr(user_representation_response, "current_state"):
            # New Mirascope response model
            representation = f"""
CURRENT STATE: {user_representation_response.current_state}

TENTATIVE PATTERNS:
{chr(10).join([pattern.pattern for pattern in user_representation_response.tentative_patterns])}

KNOWLEDGE GAPS:
{chr(10).join([gap.missing_info for gap in user_representation_response.knowledge_gaps])}

RECENT UPDATES:
{chr(10).join([update.detail for update in user_representation_response.updates.new_information])}
"""
        else:
            # Fallback to XML parsing for backwards compatibility
            representation = parse_xml_content(
                str(user_representation_response), "representation"
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

import asyncio
import datetime
import json
import logging
import os
from collections.abc import Iterable
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

USER_REPRESENTATION_METAMESSAGE_TYPE = "honcho_user_representation"

DEF_DIALECTIC_PROVIDER = ModelProvider.ANTHROPIC
DEF_DIALECTIC_MODEL = "claude-3-7-sonnet-20250219"

DEF_QUERY_GENERATION_PROVIDER = ModelProvider.GROQ
DEF_QUERY_GENERATION_MODEL = "llama-3.1-8b-instant"

QUERY_GENERATION_SYSTEM = """
Given this query about a user, generate 4 focused search queries that would help retrieve relevant facts about the user. To ground your generation, each query should focus on one of the following levels: abductive, inductive, deductive, and explicit observations.

For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's observed eating patterns", "user's most recent meal", etc.
    
Format your response as a JSON array of strings, with each string being a search query. 
Respond only in valid JSON, without markdown formatting or quotes, and nothing else.
Example:
["abductive query to retrieve hypotheses", "inductive query to retrieve observed patterns", "deductive query to retrieve explicit facts"]"""

load_dotenv()


async def get_working_representation_from_trace(session_id: str, db: AsyncSession) -> str:
    """Extract working representation from most recent deriver trace in session."""
    logger.debug(f"Searching for deriver trace in session: {session_id}")
    
    stmt = (
        select(models.Metamessage)
        .where(models.Metamessage.session_id == session_id)
        .where(models.Metamessage.label == "deriver_trace")
        .order_by(models.Metamessage.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    trace_metamessage = result.scalar_one_or_none()
    
    if not trace_metamessage:
        logger.warning(f"No deriver trace found for session: {session_id}")
        # Let's also check what metamessages exist for this session
        debug_stmt = (
            select(models.Metamessage.label, models.Metamessage.created_at)
            .where(models.Metamessage.session_id == session_id)
            .order_by(models.Metamessage.created_at.desc())
            .limit(10)
        )
        debug_result = await db.execute(debug_stmt)
        existing_metamessages = debug_result.fetchall()
        logger.debug(f"Existing metamessages in session {session_id}: {[(label, created_at) for label, created_at in existing_metamessages]}")
        return ""
    
    logger.debug(f"Found deriver trace created at: {trace_metamessage.created_at}")
    
    try:
        trace_data = json.loads(trace_metamessage.content)
        logger.debug(f"Successfully parsed trace data. Keys: {list(trace_data.keys())}")
        
        final_observations = trace_data.get("final_observations", {})
        logger.debug(f"Final observations keys: {list(final_observations.keys()) if final_observations else 'None'}")
        
        if not final_observations:
            logger.warning(f"No final_observations found in trace data. Available keys: {list(trace_data.keys())}")
            return ""
        
        # Format as structured text with normalized observations
        formatted_sections = []
        for level in ['explicit', 'deductive', 'inductive', 'abductive']:
            observations = final_observations.get(level, [])
            logger.debug(f"Level {level}: {len(observations)} observations")
            
            if observations:
                level_name = level.upper()
                formatted_sections.append(f"{level_name} OBSERVATIONS:")
                for obs in observations:
                    # Normalize observation format
                    if isinstance(obs, dict):
                        # Handle structured observations with conclusion/premises
                        if 'conclusion' in obs and 'premises' in obs:
                            conclusion = obs['conclusion']
                            premises = obs.get('premises', [])
                            if premises:
                                premises_text = ", ".join(premises)
                                formatted_sections.append(f"- {conclusion} (based on: {premises_text})")
                            else:
                                formatted_sections.append(f"- {conclusion}")
                        else:
                            # Handle other dict formats - just convert to string
                            formatted_sections.append(f"- {str(obs)}")
                    else:
                        # Handle simple string observations
                        formatted_sections.append(f"- {obs}")
                formatted_sections.append("")
        
        result = "\n".join(formatted_sections) if formatted_sections else ""
        logger.debug(f"Formatted working representation length: {len(result)} characters")
        
        if not result:
            logger.warning(f"Working representation is empty after formatting. Original final_observations: {final_observations}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in deriver trace: {e}")
        logger.debug(f"Raw content (first 500 chars): {trace_metamessage.content[:500]}")
        return ""
    except Exception as e:
        logger.error(f"Error processing deriver trace: {e}")
        return ""


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
    def __init__(self, agent_input: str, working_representation: str, additional_context: str):
        self.agent_input = agent_input
        self.working_representation = working_representation
        self.additional_context = additional_context
        self.client = ModelClient(
            provider=DEF_DIALECTIC_PROVIDER, model=DEF_DIALECTIC_MODEL
        )
        self.system_prompt = """You are operating as a context service that helps maintain psychological understanding of users across applications. You'll receive: 1) working_representation: current understanding from recent conversation analysis, and 2) additional_context: relevant historical observations. Your goal is to synthesize these insights to help applications personalize their responses to the query. Please respond in a brief, matter-of-fact, and appropriate manner. You are encouraged to synthesize the information based on three levels of reasoning: abduction, induction, and deduction. You might notice the context falling into these categories naturally, so feel free to start with the hypothesis (abduction), support it with observed patterns (induction), and solidify with explicit facts (deduction). If the context provided doesn't help address the query, write absolutely NOTHING but "No information available". If you are provided in the query with an alternative way to indicate that there is no information available, please use that instead."""

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
<working_representation>{self.working_representation}</working_representation>
<additional_context>{self.additional_context}</additional_context>
"""
            logger.info("=== DIALECTIC PROMPT ===\n" + prompt + "\n=== END DIALECTIC PROMPT ===")
            logger.debug(
                f"Prompt constructed with working representation length: {len(self.working_representation)} chars, additional context length: {len(self.additional_context)} chars"
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
<working_representation>{self.working_representation}</working_representation>
<global_context>{self.additional_context}</global_context>
"""
            logger.info("=== DIALECTIC PROMPT (STREAM) ===\n" + prompt + "\n=== END DIALECTIC PROMPT ===")

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
    app_id: str,
    user_id: str,
    session_id: str,
    queries: str | list[str],
    stream: bool = False,
) -> schemas.DialecticResponse | MessageStreamManager:
    """
    Chat with the Dialectic API that builds on-demand user representations.

    This function:
    1. Gets working representation from deriver trace
    2. Retrieves additional relevant context with premises
    3. Synthesizes them to answer the query
    """

    # format the query string
    questions = [queries] if isinstance(queries, str) else queries
    final_query = "\n".join(questions) if len(questions) > 1 else questions[0]

    logger.debug(f"Received query: {final_query} for session {session_id}")
    logger.debug("Starting dialectic processing")

    start_time = asyncio.get_event_loop().time()

    # 1. Get working representation from deriver trace
    async with tracked_db("chat.get_working_representation") as db:
        working_representation = await get_working_representation_from_trace(session_id, db)
        logger.debug(f"Retrieved working representation: {len(working_representation)} characters")

    # 2. Get additional context with premises
    async with tracked_db("chat.get_additional_context") as db:
        collection = await crud.get_or_create_user_protected_collection(db, app_id, user_id)
        embedding_store = CollectionEmbeddingStore(
            db=db,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection.public_id,
        )
        additional_context = await get_observations(final_query, embedding_store, include_premises=True)
        logger.debug(f"Retrieved additional context: {len(additional_context)} characters")

    # 3. Create simplified Dialectic
    chain = Dialectic(
        agent_input=final_query,
        working_representation=working_representation,
        additional_context=additional_context,
    )

    generation_time = asyncio.get_event_loop().time() - start_time
    logger.debug(f"Dialectic setup completed in {generation_time:.2f}s")

    langfuse_context.update_current_trace(
        session_id=session_id,
        user_id=user_id,
        release=os.getenv("SENTRY_RELEASE"),
        metadata={"environment": os.getenv("SENTRY_ENVIRONMENT")},
    )

    # 4. Generate response
    logger.debug(f"Calling Dialectic with streaming={stream}")
    query_start_time = asyncio.get_event_loop().time()
    if stream:
        response_stream = await chain.stream()
        logger.debug(
            f"Dialectic stream started after {asyncio.get_event_loop().time() - query_start_time:.2f}s"
        )
        return response_stream

    response = await chain.call()
    logger.debug(f"Dialectic Response: {response[0]['text']}")
    query_time = asyncio.get_event_loop().time() - query_start_time
    total_time = asyncio.get_event_loop().time() - start_time
    logger.debug(
        f"Dialectic response received in {query_time:.2f}s (total: {total_time:.2f}s)"
    )
    return schemas.DialecticResponse(content=response[0]["text"])


async def get_observations(
    query: str, embedding_store: CollectionEmbeddingStore, include_premises: bool = False
) -> str:
    """
    Generate queries based on the dialectic query and retrieve relevant observations.
    
    Uses semantic search to find additional relevant historical context beyond 
    what's already in the working representation.

    Args:
        query: The user query
        embedding_store: The embedding store to search
        include_premises: Whether to include premises from document metadata

    Returns:
        String containing additional relevant observations from semantic search
    """
    logger.info("=== OBSERVATION RETRIEVAL START ===")
    logger.info(f"ORIGINAL QUERY: {query}")
    observation_start_time = asyncio.get_event_loop().time()

    # Generate multiple queries for semantic search
    logger.info("GENERATING SEMANTIC QUERIES for additional context...")
    search_queries = await generate_semantic_queries(query)
    logger.info(f"GENERATED QUERIES: {search_queries}")

    # Create a list of coroutines, one for each query
    async def execute_query(i: int, search_query: str) -> list[tuple[str, str, dict]]:
        logger.info(f"EXECUTING SEMANTIC QUERY {i + 1}/{len(search_queries)}: {search_query}")
        query_start = asyncio.get_event_loop().time()
        documents = await embedding_store.get_relevant_observations(
            search_query, top_k=10, max_distance=0.85  # Increased top_k since we don't have contextualized observations
        )
        query_time = asyncio.get_event_loop().time() - query_start
        logger.info(f"QUERY {i + 1} RESULTS: {len(documents)} observations retrieved in {query_time:.2f}s")
        
        # Eagerly extract attributes to avoid DetachedInstanceError
        document_data = []
        for doc in documents:
            # Access attributes while the object is still bound to the session
            content = doc.content
            created_at = doc.created_at
            metadata = doc.h_metadata or {}
            document_data.append((content, created_at, metadata))
        
        # Log the actual observations found using the extracted data
        for j, (content, created_at, metadata) in enumerate(document_data):
            logger.info(f"  {j+1}. [{created_at.strftime('%Y-%m-%d %H:%M:%S')}] {content}")
        
        # Convert documents to tuples of (content, timestamp, metadata)
        return [(content, created_at.strftime("%Y-%m-%d-%H:%M:%S"), metadata) for content, created_at, metadata in document_data]

    # Execute all queries in parallel
    query_tasks = [
        execute_query(i, search_query) for i, search_query in enumerate(search_queries)
    ]
    all_observations_lists = await asyncio.gather(*query_tasks)

    # Combine semantic search observations and deduplicate manually
    # (can't use set() because tuples contain dicts which are unhashable)
    semantic_observations = []
    seen_content = set()
    
    for observations in all_observations_lists:
        for observation_tuple in observations:
            content, timestamp, metadata = observation_tuple
            # Use content as the deduplication key
            if content not in seen_content:
                semantic_observations.append(observation_tuple)
                seen_content.add(content)

    # Format the final response with just additional context
    response_parts = []
    
    if semantic_observations:
        # Group observations by level and date
        grouped_observations = {}
        
        for observation, timestamp, metadata in semantic_observations:
            # Extract level from metadata, default to 'unknown' if not present
            level = metadata.get('level', 'unknown')
            
            # Convert timestamp to just date (YYYY-MM-DD)
            date = timestamp.split('-')[0] + '-' + timestamp.split('-')[1] + '-' + timestamp.split('-')[2]
            
            # Initialize nested structure if needed
            if level not in grouped_observations:
                grouped_observations[level] = {}
            if date not in grouped_observations[level]:
                grouped_observations[level][date] = []
            
            # Format observation with premises if needed
            if include_premises and metadata.get('premises'):
                premises = metadata['premises']
                premises_text = ", ".join(premises)
                formatted_observation = f"{observation} (based on: {premises_text})"
            else:
                formatted_observation = observation
            
            grouped_observations[level][date].append(formatted_observation)
        
        # Format output by level and date
        for level in sorted(grouped_observations.keys()):
            if level != 'unknown':
                response_parts.append(f"\n{level.upper()} OBSERVATIONS:")
            else:
                response_parts.append(f"\nOBSERVATIONS:")
            
            for date in sorted(grouped_observations[level].keys(), reverse=True):  # Most recent dates first
                observations_for_date = grouped_observations[level][date]
                if len(observations_for_date) == 1:
                    response_parts.append(f"[{date}]: {observations_for_date[0]}")
                else:
                    response_parts.append(f"[{date}]:")
                    for obs in observations_for_date:
                        response_parts.append(f"  - {obs}")

    observations_string = "\n".join(response_parts) if response_parts else "No additional relevant context found."

    total_time = asyncio.get_event_loop().time() - observation_start_time
    
    logger.info("=== OBSERVATION RETRIEVAL SUMMARY ===")
    logger.info(f"TOTAL TIME: {total_time:.2f}s")
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
        observations_list = [line.strip() for line in observations_string.split('\n') if line.strip()]
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


async def generate_semantic_queries(query: str) -> list[str]:
    """
    Generate multiple semantically relevant queries based on the original query using LLM.
    This helps retrieve more diverse and relevant facts from the vector store.

    Args:
        query: The original dialectic query

    Returns:
        A list of semantically relevant queries
    """
    logger.info("=== QUERY GENERATION START ===")
    logger.info(f"INPUT QUERY: {query}")
    query_start = asyncio.get_event_loop().time()

    logger.info("CALLING LLM for query generation...")
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
        logger.info(f"LLM RESPONSE received in {llm_time:.2f}s")
        logger.info(f"RAW LLM RESPONSE: {result}")

        # Parse the JSON response to get a list of queries
        try:
            queries = json.loads(result)
            if not isinstance(queries, list):
                # Fallback if response is not a valid list
                logger.info("LLM response not a list, using as single query")
                queries = [result]
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            logger.info("Failed to parse JSON response, using raw response as query")
            queries = [query]  # Fall back to the original query

        # Ensure we always include the original query
        if query not in queries:
            logger.info("Adding original query to results")
            queries.append(query)

        total_time = asyncio.get_event_loop().time() - query_start
        logger.info(f"GENERATED {len(queries)} QUERIES in {total_time:.2f}s:")
        for i, q in enumerate(queries, 1):
            logger.info(f"  {i}. {q}")
        logger.info("=== QUERY GENERATION END ===")

        return queries
    except Exception as e:
        logger.error(f"Error during API call: {str(e)}")
        raise

# Backward compatibility aliases
async def get_facts(query: str, embedding_store: CollectionEmbeddingStore) -> str:
    """Backward compatibility alias for get_observations."""
    return await get_observations(query, embedding_store)

async def get_long_term_facts(query: str, embedding_store: CollectionEmbeddingStore) -> list[str]:
    """Backward compatibility alias for get_long_term_observations."""
    return await get_long_term_observations(query, embedding_store)

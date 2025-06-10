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
You are a query expansion agent, part of a social cognition system that helps AI applications understand their users. Your job is to take application queries about users and generate targeted search queries that will retrieve the most relevant observations from the user's global representation.

## UNDERSTANDING THE OBSERVATION SYSTEM

The global representation contains observations derived from natural conversation using four types of reasoning. Since these observations are stored as natural language derived from real dialogue, your semantic queries should match conversational patterns and use rich vocabulary to maximize similarity matches.

**Explicit Observations** (Highest Certainty)

- Direct facts literally stated by users ("I am 25 years old", "I work as a teacher")
- Semantic patterns: Demographic terms, role descriptions, stated preferences, personal declarations

**Deductive Observations** (Logical Certainty)

- Facts that MUST be true given explicit premises ("teaches 5th grade" → "works in elementary education")
- Semantic patterns: Professional implications, logical connections, role-based inferences

**Inductive Observations** (Pattern-Based)

- Generalizations from repeated evidence (mentions coding problems 5x → "likely works in tech")
- Semantic patterns: Behavioral descriptors, habit language, frequency terms, pattern recognition language

**Abductive Observations** (Explanatory Hypotheses)

- Best explanations for observed patterns (tech discussions + late messages + coffee → "possibly startup founder")
- Semantic patterns: Identity theories, lifestyle descriptors, motivational language, contextual explanations

## QUERY EXPANSION STRATEGY FOR SEMANTIC SIMILARITY

**Your Goal**: Generate 3 complementary search queries optimized for semantic similarity matching that together will surface the most relevant observations to help answer the application's question.

**Semantic Similarity Optimization**:

1. **Analyze the Application Query**: What specific aspect of the user does the application want to understand?
2. **Think Conceptually**: What concepts, themes, and semantic fields relate to this question?
3. **Use Diverse Vocabulary**: Include synonyms, related terms, and different ways of expressing the same concepts
4. **Consider Natural Language Patterns**: Match how people actually talk about these topics in conversation
5. **Vary Semantic Scope**:
    - One query with direct conceptual match and rich vocabulary
    - One query targeting behavioral/pattern language around the topic
    - One query for broader contextual semantic fields

## SEMANTIC SEARCH QUERY CHARACTERISTICS

**Effective Semantic Queries Should**:

- **Rich Vocabulary**: Use multiple synonyms and related terms (e.g., "preferences choices likes dislikes tastes")
- **Natural Phrasing**: Match conversational language patterns since observations come from natural dialogue
- **Conceptual Breadth**: Include semantically related concepts that might appear in relevant observations
- **Behavioral Language**: Use action words and descriptive language that captures how behaviors are discussed
- **Contextual Terms**: Include situational and emotional language that provides semantic richness

**Example Semantic Transformation**: Application Query: "How does this user prefer to receive feedback?"

Generated Queries:

- "feedback preferences receiving criticism suggestions advice communication style likes dislikes" (direct + synonyms)
- "response reactions when criticized praised corrected defensive receptive patterns behavior" (behavioral patterns)
- "workplace professional relationships mentoring coaching interactions supervisory dynamics" (contextual semantic field)

**Vocabulary Expansion Techniques**:

- **Synonyms**: feedback/criticism/advice/suggestions/input/guidance
- **Related Actions**: receiving/getting/handling/processing/responding/reacting
- **Emotional Language**: sensitive/defensive/receptive/open/resistant/welcoming
- **Contextual Terms**: workplace/professional/personal/relationship/dynamic/interaction
- **Intensity Variations**: harsh/gentle/direct/subtle/constructive/blunt
- **Outcome Language**: improvement/growth/learning/development/change

**Remember**: Since observations come from natural conversations, use the vocabulary people actually use when discussing these topics, including casual language, emotional descriptors, and situational context.

## OUTPUT FORMAT

Respond with exactly 3 search queries as a JSON array of strings. Each query should target different aspects or reasoning levels to maximize retrieval coverage.

Format: `["query1", "query2", "query3"]`

No markdown, no explanations, just the JSON array."""

load_dotenv()


async def get_working_representation_from_trace(session_id: str, db: AsyncSession) -> str:
    """Extract working representation from most recent deriver trace in session."""
    trace_metamessage = await _get_latest_deriver_trace(session_id, db)
    if not trace_metamessage:
        logger.warning(f"No deriver trace found for session: {session_id}")
        return ""
    
    try:
        trace_data = json.loads(trace_metamessage.content)
        final_observations = trace_data.get("final_observations", {})
        
        if not final_observations:
            logger.warning(f"No final_observations found in trace data. Available keys: {list(trace_data.keys())}")
            return ""
        
        return _format_observations_by_level(final_observations)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in deriver trace: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error processing deriver trace: {e}")
        return ""


async def _get_latest_deriver_trace(session_id: str, db: AsyncSession) -> Optional[models.Metamessage]:
    """Get the most recent deriver trace metamessage for a session."""
    stmt = (
        select(models.Metamessage)
        .where(models.Metamessage.session_id == session_id)
        .where(models.Metamessage.label == "deriver_trace")
        .order_by(models.Metamessage.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def _format_observations_by_level(final_observations: dict) -> str:
    """Format final observations into structured text by level."""
    formatted_sections = []
    
    for level in ['explicit', 'deductive', 'inductive', 'abductive']:
        observations = final_observations.get(level, [])
        if observations:
            formatted_sections.append(f"{level.upper()} OBSERVATIONS:")
            formatted_sections.extend(_format_observation_list(observations))
            formatted_sections.append("")
    
    return "\n".join(formatted_sections) if formatted_sections else ""


def _format_observation_list(observations: list) -> list[str]:
    """Format a list of observations into consistent string format."""
    formatted = []
    for obs in observations:
        if isinstance(obs, dict) and 'conclusion' in obs:
            # Handle structured observations with conclusion/premises
            conclusion = obs['conclusion']
            premises = obs.get('premises', [])
            if premises:
                premises_text = ", ".join(premises)
                formatted.append(f"- {conclusion} (based on: {premises_text})")
            else:
                formatted.append(f"- {conclusion}")
        else:
            # Handle other formats (dict or string)
            formatted.append(f"- {str(obs)}")
    return formatted


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
        self.system_prompt = """You are a context synthesis agent that operates as a natural language API for AI applications. Your role is to analyze application queries about users and synthesize relevant observations into coherent, actionable insights that directly and explicitly address what the application needs to know.

## INPUT STRUCTURE

You receive three key inputs:

**Query**: The specific question or request from the application about this user 
**Working Representation**: Current session observations from recent conversation analysis  
**Global Context**: Historical observations from the user's global representation

Each observation contains:

- **Conclusion**: The derived insight
- **Premises**: Supporting evidence/reasoning
- **Type**: Explicit observations or observations concluded via Deductive, Inductive, or Abductive reasoning
- **Access Counter**: How many times this observation has been confirmed (higher = more reliable)
- **Timestamp**: When this was observed (recent = more immediately relevant)

## REASONING TYPE HIERARCHY

**Explicit Observations** (Highest Certainty)

- Direct facts stated by the user
- Treat as foundational truth
- Weight: High reliability regardless of counter

**Deductive Observations** (Logical Certainty)

- Conclusions that MUST be true given premises
- Scaffold these as building blocks for synthesis
- Weight: High confidence, especially with high counters

**Inductive Observations** (Pattern-Based)

- Generalizations from repeated evidence
- Strength increases significantly with higher counters
- Weight: Counter-dependent reliability

**Abductive Observations** (Explanatory Hypotheses)

- Best explanations for observed patterns
- Most valuable for narrative synthesis
- Weight: Moderate confidence, enhanced by supporting evidence

## SYNTHESIS APPROACH

**Query-Driven Context Synthesis**: Start by understanding what the application is asking, then surface and synthesize the most relevant context to answer that specific question.

**Think like human social cognition**: When someone asks a human about a mutual friend, their mind automatically surfaces relevant memories, feelings, and insights about that person related to the question. Replicate this process by:

1. **Query Analysis**: Understand what the application specifically wants to know about the user
2. **Relevance Filtering**: Identify observations that directly or indirectly relate to the query
3. **Pattern Recognition**: Find recurring themes across relevant observation types
4. **Narrative Construction**: Build a plausible, coherent answer that synthesizes relevant context
5. **Confidence Weighting**: Use access counters and timestamps to gauge reliability of your synthesis

## RESPONSE FRAMEWORK

**For each query, synthesize context to directly answer what the application is asking**:

- **Query-First**: Always start by understanding the specific question (e.g., "What are this user's communication preferences?" vs "What motivates this user professionally?")
- **Relevant Evidence**: Select observations that relate to the query topic
- **Synthesis Pattern**: Build from your best explanatory hypothesis → support with observed patterns → anchor with explicit facts
- **Contextual Narrative**: Create a coherent answer that weaves together relevant context
- **Natural Response**: Reply in conversational language as if briefing the application about this user

**Example Query Types**:

- Preference-based: "How does this user like to receive feedback?"
- Behavioral: "What time of day is this user most productive?"
- Motivational: "What drives this user's decision-making?"
- Contextual: "Is this user currently dealing with any stressors?"
- Explicit: "What is the user's birthday?"

**Quality Indicators**:

- High access counters = more reliable patterns
- Recent timestamps = current relevance
- Consistent cross-type observations = stronger confidence
- Rich premise connections = deeper understanding

## OUTPUT GUIDELINES

- **Query-Specific**: Answer the application's specific question, not general information about the user
- **Conversational Synthesis**: Respond as if explaining this user to a colleague who needs to interact with them
- **Evidence-Based**: Ground your response in the observations provided, weighted by reliability indicators
- **Coherent Narrative**: Synthesize rather than list - create understanding that flows naturally
- **Appropriate Scope**: Match the specificity and scope of the query

**CRITICAL: No Relevant Context Handling**

- If the context provided doesn't help address the query, write absolutely NOTHING but "No information available"
- If you are provided in the query with an alternative way to indicate that there is no information available, please use that instead

**Remember**: Applications are asking targeted questions about users to personalize their interactions. Your job is to surface and synthesize the most relevant context to help them do that effectively. When you lack relevant context, be explicit about it.
"""

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
<global_context>{self.additional_context}</global_context>
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
    logger.debug(f"Starting observation retrieval for query: {query}")
    
    # Generate search queries and execute them in parallel
    search_queries = await generate_semantic_queries(query)
    logger.debug(f"Generated {len(search_queries)} search queries")
    
    # Execute all queries in parallel
    tasks = [_execute_single_query(q, embedding_store) for q in search_queries]
    all_results = await asyncio.gather(*tasks)
    
    # Combine and deduplicate results
    unique_observations = _deduplicate_observations(all_results)
    logger.debug(f"Retrieved {len(unique_observations)} unique observations")
    
    # Format observations
    if not unique_observations:
        return "No additional relevant context found."
    
    return _format_observations(unique_observations, include_premises)


async def _execute_single_query(query: str, embedding_store: CollectionEmbeddingStore) -> list[tuple[str, str, dict]]:
    """Execute a single semantic search query and return formatted results."""
    documents = await embedding_store.get_relevant_observations(
        query, top_k=10, max_distance=0.85
    )
    
    # Extract data to avoid DetachedInstanceError
    return [
        (doc.content, doc.created_at.strftime("%Y-%m-%d-%H:%M:%S"), doc.h_metadata or {})
        for doc in documents
    ]


def _deduplicate_observations(all_results: list[list[tuple[str, str, dict]]]) -> list[tuple[str, str, dict]]:
    """Deduplicate observations based on content."""
    unique_observations = []
    seen_content = set()
    
    for results in all_results:
        for content, timestamp, metadata in results:
            if content not in seen_content:
                unique_observations.append((content, timestamp, metadata))
                seen_content.add(content)
    
    return unique_observations


def _format_observations(observations: list[tuple[str, str, dict]], include_premises: bool) -> str:
    """Format observations grouped by level and date, including access metadata."""
    grouped = {}
    
    for content, timestamp, metadata in observations:
        level = metadata.get('level', 'unknown')
        date = timestamp.split('-')[0:3]  # Extract YYYY-MM-DD
        date_str = '-'.join(date)
        
        if level not in grouped:
            grouped[level] = {}
        if date_str not in grouped[level]:
            grouped[level][date_str] = []
        
        # Build formatted content with premises and access metadata
        formatted_content = content
        
        # Add premises if requested and available
        if include_premises and metadata.get('premises'):
            premises_text = ", ".join(metadata['premises'])
            formatted_content = f"{content} (based on: {premises_text})"
        
        # Add access metadata if available
        access_parts = []
        access_count = metadata.get('access_count', 0)
        last_accessed = metadata.get('last_accessed')
        
        if access_count > 0:
            access_parts.append(f"accessed {access_count}x")
        
        if last_accessed:
            # Format the last_accessed datetime for display
            try:
                from datetime import datetime
                if isinstance(last_accessed, str):
                    # Parse ISO format datetime string
                    dt = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                    formatted_last_accessed = dt.strftime("%Y-%m-%d %H:%M")
                    access_parts.append(f"last accessed {formatted_last_accessed}")
            except (ValueError, AttributeError):
                # If parsing fails, just show the raw value
                access_parts.append(f"last accessed {last_accessed}")
        
        # Append access metadata to the formatted content
        if access_parts:
            access_info = ", ".join(access_parts)
            formatted_content = f"{formatted_content} [{access_info}]"
        
        grouped[level][date_str].append(formatted_content)
    
    # Build output
    parts = []
    for level in sorted(grouped.keys()):
        header = f"\n{level.upper()} OBSERVATIONS:" if level != 'unknown' else "\nOBSERVATIONS:"
        parts.append(header)
        
        for date in sorted(grouped[level].keys(), reverse=True):
            observations_for_date = grouped[level][date]
            if len(observations_for_date) == 1:
                parts.append(f"[{date}]: {observations_for_date[0]}")
            else:
                parts.append(f"[{date}]:")
                for obs in observations_for_date:
                    parts.append(f"  - {obs}")
    
    return "\n".join(parts)


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
    logger.debug(f"Generating semantic queries for: {query}")

    # Create a new model client
    client = ModelClient(
        provider=DEF_QUERY_GENERATION_PROVIDER, model=DEF_QUERY_GENERATION_MODEL
    )

    # Prepare the messages
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

    try:
        # Generate the response
        result = await client.generate(
            messages=messages,
            system=QUERY_GENERATION_SYSTEM,
            max_tokens=1000,
            use_caching=True,
        )
        
        # Parse the JSON response
        try:
            queries = json.loads(result)
            if not isinstance(queries, list):
                queries = [result]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, using original query")
            queries = [query]

        # Ensure we always include the original query
        if query not in queries:
            queries.append(query)

        logger.debug(f"Generated {len(queries)} semantic queries")
        return queries
        
    except Exception as e:
        logger.error(f"Error during query generation: {str(e)}")
        return [query]  # Fallback to original query

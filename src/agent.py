import asyncio
import json
import logging
from inspect import cleandoc as c

from dotenv import load_dotenv
from mirascope import llm, prompt_template
from mirascope.integrations.langfuse import with_langfuse
from sentry_sdk.ai.monitoring import ai_track
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from src import crud, models
from src.dependencies import tracked_db
from src.deriver.tom.embeddings import CollectionEmbeddingStore
from src.deriver.models import SemanticQueries

# Configure logging
logger = logging.getLogger(__name__)

USER_REPRESENTATION_METAMESSAGE_TYPE = "honcho_user_representation"

load_dotenv()

# NOTE: Temporarily disabled @with_langfuse() decorators across multiple functions
# due to compatibility issue between Mirascope 1.25.0 and Langfuse 2.60.8 where 
# BaseMessageParam objects don't have a .get() method. The Langfuse integration
# expects message_param to be a dict but it's a Pydantic model.
# Error: AttributeError: 'BaseMessageParam' object has no attribute 'get'
#
# Additionally fixed ReasoningResponse validation error where explicit field
# was expecting list[str] but receiving a single string from LLM.
#
# AFFECTED FUNCTIONS (all temporarily have @with_langfuse() commented out):
# - src/agent.py: dialectic_call, dialectic_stream
# - src/utils/history.py: create_short_summary, create_long_summary  
# - src/deriver/surprise_reasoner.py: critical_analysis_call
# - src/deriver/tom/long_term.py: get_user_representation_long_term, extract_facts_long_term
# - src/deriver/tom/single_prompt.py: tom_inference, user_representation_inference
# - src/deriver/consumer.py: process_user_message (@observe decorator)
#
# TODO: Re-enable all Langfuse decorators when compatibility issue is resolved


@prompt_template()
def dialectic_prompt(query: str, working_representation: str, additional_context: str) -> str:
    return c(
        f"""
        You are a context synthesis agent that operates as a natural language API for AI applications. Your role is to analyze application queries about users and synthesize relevant observations into coherent, actionable insights that directly and explicitly address what the application needs to know.

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

        <query>{query}</query>
        <working_representation>{working_representation}</working_representation>
        <global_context>{additional_context}</global_context>
        """
    )


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


async def _get_latest_deriver_trace(session_id: str, db: AsyncSession) -> models.Metamessage | None:
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


@ai_track("Dialectic Call")
# TODO: Re-enable when Mirascope-Langfuse compatibility issue is fixed
# @with_langfuse()
@llm.call(provider="anthropic", model="claude-3-7-sonnet-20250219")
async def dialectic_call(query: str, working_representation: str, additional_context: str):
    return dialectic_prompt(query, working_representation, additional_context)


@ai_track("Dialectic Stream")
# TODO: Re-enable when Mirascope-Langfuse compatibility issue is fixed
# @with_langfuse()
@llm.call(provider="anthropic", model="claude-3-7-sonnet-20250219", stream=True)
async def dialectic_stream(query: str, working_representation: str, additional_context: str):
    return dialectic_prompt(query, working_representation, additional_context)


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

    # 3. Call dialectic with enhanced context
    generation_time = asyncio.get_event_loop().time() - start_time
    logger.debug(f"Dialectic setup completed in {generation_time:.2f}s")

    # 4. Generate response
    if stream:
        logger.debug("Calling Dialectic with streaming")
        response = await dialectic_stream(
            final_query, working_representation, additional_context
        )
        return response
    else:
        logger.debug("Calling Dialectic with non-streaming")
        query_start_time = asyncio.get_event_loop().time()
        response = await dialectic_call(final_query, working_representation, additional_context)
        query_time = asyncio.get_event_loop().time() - query_start_time
        logger.debug(f"Dialectic response received in {query_time:.2f}s")
        return response


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
    search_queries_result = await generate_semantic_queries(query)
    search_queries = search_queries_result.queries
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



@prompt_template()
def query_generation_prompt(query: str) -> str:
    return c(
        f"""
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

        Respond with exactly 3 search queries as a JSON object with a "queries" field containing an array of strings. Each query should target different aspects or reasoning levels to maximize retrieval coverage.

        Format: `{{"queries": ["query1", "query2", "query3"]}}`

        No markdown, no explanations, just the JSON object.

        <query>{query}</query>
        """
    )


@with_langfuse()
@llm.call(provider="groq", model="llama-3.1-8b-instant", response_model=SemanticQueries)
async def generate_semantic_queries(query: str):
    return query_generation_prompt(query)


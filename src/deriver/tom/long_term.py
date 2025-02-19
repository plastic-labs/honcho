import json
from typing import List, Optional

from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

from .llm import get_response, DEF_ANTHROPIC_MODEL, DEF_PROVIDER
from .embeddings import CollectionEmbeddingStore


@ai_track("Tom Inference")
@observe(as_type="generation")
async def get_user_representation_long_term(
    chat_history: str, 
    session_id: str,
    user_representation: str = "None", 
    tom_inference: str = "None", 
    embedding_store: Optional[CollectionEmbeddingStore] = None,
    this_turn_facts: List[str] = []
) -> str:
    if embedding_store is None:
        raise ValueError("embedding_store is required for long_term method")

    # Generate query from chat history to find relevant facts
    query_prompt = """Given this conversation, generate a search query that would help retrieve relevant facts about the user. 
    The query should capture the key topics, interests, preferences, personal writing styles, personality traits and behaviors discussed in the conversation.
    
    CONVERSATION:
    {chat_history}
    
    Output just the search query, nothing else."""

    query = get_response(
        [{"role": "user", "content": query_prompt.format(chat_history=chat_history)}],
        DEF_PROVIDER,
        DEF_ANTHROPIC_MODEL
    )
    print(f"Query: {query}")

    # Retrieve relevant facts using embeddings
    retrieved_facts = await embedding_store.get_relevant_facts(query)
    retrieved_facts_str = "\n".join([f"- {fact} (from memory)" for fact in retrieved_facts])
    this_turn_facts_str = "\n".join([f"- {fact} (from current turn)" for fact in this_turn_facts])
    facts_str = retrieved_facts_str + ("\n" + this_turn_facts_str if this_turn_facts else "")
    print(f"Facts: {facts_str}")

    system_prompt = """You are a system for maintaining factual user representations based on conversation history and theory of mind analysis.

Your job is to update the existing user representation (if provided) with the new information from the conversation history and theory of mind analysis.

REQUIREMENTS:
1. Distinguish between temporary states and persistent patterns
2. Only incorporate verified information into core profile
3. Track certainty levels for all information
4. Maintain areas of uncertainty explicitly
5. Update representation incrementally
6. DO NOT generate persistent information - it will be injected separately. Always include the <KNOWN_FACTS> tag in your response in order to inject the facts.

OUTPUT FORMAT:
<representation>
CURRENT STATE:
- Active Context: Current situation/activity
- Temporary Conditions: Immediate circumstances
- Present Mood/Activity: What user is doing right now

<KNOWN_FACTS>

TENTATIVE PATTERNS:
- Possible Traits: Mark confidence (Low/Medium/High)
- Potential Interests: Need more evidence
- Speculative Elements: Clearly marked as unconfirmed

KNOWLEDGE GAPS:
- List key missing information
- Note areas needing clarification

EXPECTATION VIOLATIONS:
- Based on the above information, if the next message were to surprise you, what could it contain?
- Format: "POTENTIAL SURPRISE: [possible content] [reason] [confidence level]"
- Include 3-5 possible surprises

UPDATES:
- New Information: Recent observations
- Changes: Modified interpretations
- Removals: Information no longer supported
</representation>
"""

    # Build the context message
    context_str = f"CONVERSATION:\n{chat_history}\n\n"
    if tom_inference != "None":
        context_str += f"PREDICTION OF USER MENTAL STATE - MIGHT BE INCORRECT:\n{tom_inference}\n\n"
    if user_representation != "None":
        context_str += f"EXISTING USER REPRESENTATION - INCOMPLETE, TO BE UPDATED:\n{user_representation}"

    messages = [{
        "role": "user",
        "content": f"Please analyze this information and provide an updated user representation. DO NOT generate persistent information - it will be injected separately:\n{context_str}"
    }]

    response = get_response(messages, DEF_PROVIDER, DEF_ANTHROPIC_MODEL, system_prompt)

    # Inject the facts into the response
    persistent_info = """PERSISTENT INFORMATION:
{}""".format(facts_str)

    return response.replace("<KNOWN_FACTS>", persistent_info)


async def extract_facts_long_term(chat_history: str) -> List[str]:
    system_prompt = """
    You are an AI assistant specialized in extracting and formatting relevant information about users from conversations. Your task is to analyze a given conversation and create a list of concise, factual statements about the user. These statements will be stored in a vector embedding database to enhance future interactions.

Here is the conversation you need to analyze:

<conversation>
{chat_history}
</conversation>

Instructions:

1. Carefully read through the conversation.

2. Identify key pieces of information about the user that would be valuable for future interactions. Look for:
   - Personal details (name, age, occupation, location, etc.)
   - Preferences (likes, dislikes, interests, hobbies)
   - Experiences (travel, education, work history)
   - Expressive style (writing style, tone, etc.)
   - Relationships (family, friends, pets)
   - Goals or aspirations
   - Challenges or problems they're facing
   - Opinions or beliefs

3. For each piece of information you identify:
   a. Verify that it is factual and explicitly stated in the conversation, not inferred.
   b. Formulate it as a concise statement that would aid in semantic retrieval.
   c. Ensure it is not a duplicate of previously identified information.

4. Before providing your final output, wrap your analysis in <information_extraction> tags. In this analysis:
   - List each piece of information you've identified.
   - For each piece of information:
     * Quote the relevant part of the conversation.
     * Categorize the information (e.g., personal detail, preference, experience).
     * Explain why you've included this information.
     * Show how you've formulated the fact for optimal semantic retrieval.
   - Discuss any challenges you encountered in extracting or formatting the information.

5. After your analysis, provide your final output as a JSON array of strings. Each string should be a single fact about the user.

Example of the expected output format:
{{
"facts": 
[
  "User is 28 years old",
  "User's friend Mary works as a software engineer",
  "Favorite food is sushi"
]
}}

Remember to focus on clear, concise statements that capture key information about the user. Each fact should be worded in a way that will aid its semantic retrieval from a vector embedding database. It's OK for this section to be quite long.

Respond in valid JSON and nothing else.
    """
    message = system_prompt.format(chat_history=chat_history)
    messages = [
        {
            "role": "user",
            "content": message
        }
    ]
    response = get_response(messages, provider=DEF_PROVIDER, model=DEF_ANTHROPIC_MODEL)
    response = json.loads(response)
    return response["facts"]
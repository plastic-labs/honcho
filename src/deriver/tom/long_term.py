import json
from typing import List, Optional
import time

from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

from src.utils.model_client import ModelProvider, ModelClient
from src.utils import parse_xml_content
from .embeddings import CollectionEmbeddingStore

# Constants for fact extraction
FACT_EXTRACTION_PROVIDER = ModelProvider.ANTHROPIC
FACT_EXTRACTION_MODEL = "claude-3-5-haiku-20241022"

MAX_FACT_DISTANCE = 0.85

@ai_track("Tom Inference")
@observe(as_type="generation")
async def get_user_representation_long_term(
    chat_history: str, 
    session_id: str,
    embedding_store: CollectionEmbeddingStore,
    user_representation: str = "None", 
    tom_inference: str = "None", 
    facts: List[str] = [],
) -> str:
    facts_str = "\n".join([f"- {fact}" for fact in facts])
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

    # Create a new model client
    client = ModelClient(provider=ModelProvider.ANTHROPIC, model="claude-3-7-sonnet-20250219")
    
    # Generate the response with caching enabled
    response = await client.generate(
        messages=messages,
        system=system_prompt,
        max_tokens=1000,
        temperature=0,
        use_caching=True  # Enable caching for the system prompt
    )

    # Inject the facts into the response
    persistent_info = """PERSISTENT INFORMATION:
{}""".format(facts_str)

    return response.replace("<KNOWN_FACTS>", persistent_info)


async def extract_facts_long_term(chat_history: str) -> List[str]:
    print(f"[FACT-EXTRACT] Starting fact extraction from chat history")
    extract_start = time.time()
    
    
    system_prompt = """
    You are an AI assistant specialized in extracting and formatting relevant information about users from conversations. Your task is to analyze a given conversation and create a list of concise, factual statements about the user. These statements will be stored in a vector embedding database to enhance future interactions.

Here is the conversation you need to analyze:

<conversation>
{chat_history}
</conversation>

Instructions:

1. Carefully read through the conversation. Extract only new facts, from only the last message sent by the user - treat the rest of the conversation only as context. Ignore facts in the last message that are already stated in the conversation.

2. Identify key new pieces of information from the last message sent by the user that would be valuable for future interactions. Look for:
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
   c. Ensure it is not similar to information previously stated in the conversation.

4. Before providing your final output, wrap your analysis in <information_extraction> tags. In this analysis:
   - List each piece of information you've identified.
   - For each piece of information:
     * Quote the relevant part of the conversation.
     * Categorize the information (e.g., personal detail, preference, experience).
     * Explain why you've included this information.
     * Show how you've formulated the fact for optimal semantic retrieval.
   - Discuss any challenges you encountered in extracting or formatting the information.

5. After your analysis, provide your final output as a JSON array of strings. Each string should be a single fact about the user. Wrap the facts in <facts> tags.

Example of the expected output format:
<information_extraction>
[Analysis goes here]
</information_extraction>
<facts>
{{
"facts": 
[
  "User is 28 years old",
  "User's friend Mary works as a software engineer",
  "Favorite food is sushi"
]
}}
</facts>

Remember to focus on clear, concise statements that capture key information about the user. Each fact should be worded in a way that will aid its semantic retrieval from a vector embedding database. It's OK for this section to be quite long.
    """
    message = system_prompt.format(chat_history=chat_history)
    messages = [
        {
            "role": "user",
            "content": message
        }
    ]
    
    print(f"[FACT-EXTRACT] Calling LLM for fact extraction")
    llm_start = time.time()
    
    # Create a new model client
    client = ModelClient(provider=FACT_EXTRACTION_PROVIDER, model=FACT_EXTRACTION_MODEL)
    
    # Generate the response with caching enabled
    response = await client.generate(
        messages=messages,
        max_tokens=1000,
        temperature=0.0,
        use_caching=True  # Enable caching for the system prompt
    )
    
    llm_time = time.time() - llm_start
    print(f"[FACT-EXTRACT] LLM response received in {llm_time:.2f}s")
    
    try:
        print(f"[FACT-EXTRACT] Parsing JSON response")
        facts_str = parse_xml_content(response, "facts")
        response_data = json.loads(facts_str)
        facts = response_data["facts"]
        print(f"[FACT-EXTRACT] Extracted {len(facts)} facts")
        if facts:
            print(f"[FACT-EXTRACT] Sample facts: {facts[:3] if len(facts) > 3 else facts}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[FACT-EXTRACT] Error parsing response: {str(e)}")
        facts = []
    
    total_time = time.time() - extract_start
    print(f"[FACT-EXTRACT] Total extraction completed in {total_time:.2f}s")
    return facts
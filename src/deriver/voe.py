import os
from typing import List
from anthropic import AsyncAnthropic

# Initialize the Anthropic client
anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def user_prediction_thought(chat_history: str) -> str:
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{
                "type": "text",
                "text": """
Generate a prediction about what the user will say based on the way the given conversation has been going. Use your theory of mind skills to impute the user's unobservable mental state. Also list other pieces of data that would help improve your prediction. Be precise and succinct, provide only the output as formatted below, no preface or commentary.

<prediction>
{prediction goes here}
</prediction>

<additional-data>
{numbered list goes here}
</additional-data>
"""
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
Conversation:
<conversation>
{chat_history}
</conversation>
'''
            }
        ]
    )
    return response.content[0].text

async def user_prediction_thought_revision(user_prediction_thought: str, retrieved_context: str, chat_history: str) -> str:
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": """
Revise the theory of mind prediction based on the additional data provided in the following format:

<revision>
{revised theory of mind prediction goes here}
</revision>

Be succinct and precise, provide only the output as formatted above. If you believe there are no changes to be made, output "None".
"""
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
Here is the thought generated previously:

<thought>
{user_prediction_thought}
</thought>

Based on this thought, the following personal data has been retrieved:

<personal_data>
{retrieved_context}
</personal_data>

And here's the conversation history that was used to generate the original thought:

<history>
{chat_history}
</history>
'''
            }
        ]
    )
    return response.content[0].text

async def voe_thought(user_prediction_thought_revision: str, actual: str) -> str:
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": '''
Compare the prediction with the actual user message and assess whether or not the prediction sufficiently captured the actual user message. Be precise and succinct, provide only the output formatted as follows:

<assessment>
{assessment goes here}
</assessment>
'''
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
<prediction>
{user_prediction_thought_revision}
</prediction>

<actual>
{actual}
</actual>
'''
            }
        ]
    )
    return response.content[0].text

async def voe_derive_facts(ai_message: str, user_prediction_thought_revision: str, actual: str, voe_thought: str) -> str:
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": '''
We've been employing a method called Violation of Expectation to better assist a user in conversation. From a previous set of instances, we've generated a prediction about what the user would say in response to the assistant (denoted as "thought") as well as a prediction about how well that thought predicted the actual user message (denoted as "voe-thought").

Derive a fact (or set of facts) about the user based on the difference between the thought and their actual response plus the voe-thought. Provide the fact(s) solely in reference to the Actual response and theory of mind prediction about that response; i.e. do not derive a fact that negates the thought about what they were going to say. Do not speculate anything about the user. Each fact must contain enough specificity to stand alone. If there are many facts, list them out. If there's nothing to derive (i.e. the statements are sufficiently similar), print "None".
'''
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
Most recent AI message:
<ai_message>
{ai_message}
</ai_message>

Thought about what they were going to say:
<thought>
{user_prediction_thought_revision}
</thought>

Actual response:
<actual>
{actual}
</actual>

Theory of mind prediction about that response:
<voe_thought>
{voe_thought}
</voe_thought>
'''
            }
        ]
    )
    return response.content[0].text

async def check_voe_list(existing_facts: List[str], new_facts: List[str]):
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": "Your job is to compare the existing list of facts to the new facts and determine if the existing list sufficiently represents the facts in the new list or not."
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
<existing_facts>
{existing_facts}
</existing_facts>

<new_facts>
{new_facts}
</new_facts>

If you believe there are new facts that are sufficiently different from the ones in the list, output those new facts in a numbered list. If not, output "None". Do not provide extra commentary.'''
            }
        ]
    )
    return response.content[0].text.strip().lower() == "true"
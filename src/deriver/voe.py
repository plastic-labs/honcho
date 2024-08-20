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
                "text": "Generate a \"thought\" that makes a theory of mind prediction about what the user will say based on the way the conversation has been going. Also list other pieces of data that would help improve your prediction."
        }],
        messages=[
            {
                "role": "user",
                "content": f'''Conversation:
<conversation>
{chat_history}
</conversation>

Generate the additional pieces of data as a numbered list.'''
        },
        {
            "role": "assistant",
            "content": "Thought:"
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
            "text": "You are tasked with revising theory of mind \"thoughts\" about what the user is going to say."
        }],
        messages=[
            {
                "role": "user",
                "content": f'''Here is the thought generated previously:

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

Given the thought, conversation history, and personal data, make changes to the thought. If there are no changes to be made, output "None".'''
            },
            {
                "role": "assistant",
                "content": "thought revision:"
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
            "text": "Below is a \"thought\" about what the user was going to say, and then what the user actually said. Generate a theory of mind prediction about the user based on the difference between the \"thought\" and actual response."
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
<thought>
{user_prediction_thought_revision}
</thought>

<actual>
{actual}
</actual>

Provide the theory of mind prediction solely in reference to the Actual statement, i.e. do not generate something that negates the thought. Do not speculate anything about the user.'''
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
            "text": "Below is the most recent AI message we sent to a user, a \"thought\" about what the user was going to say to that, what the user actually responded with, and a theory of mind prediction about the user's response. Derive a fact (or list of facts) about the user based on the difference between the original thought and their actual response plus the theory of mind prediction about that response."
        }],
        messages=[
            {
                "role": "user",
                "content": f'''Most recent AI message:
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

Provide the fact(s) solely in reference to the Actual response and theory of mind prediction about that response; i.e. do not derive a fact that negates the thought about what they were going to say. Do not speculate anything about the user. Each fact must contain enough specificity to stand alone. If there are many facts, list them out. Your response should be a numbered list with each item on a new line, for example: `\n\n1. foo\n\n2. bar\n\n3. baz`. If there's nothing to derive (i.e. the statements are sufficiently similar), print "None".'''
            }
        ]
    )
    return response.content[0].text

async def check_voe_list(existing_facts: List[str], new_fact: str) -> bool:
    response = await anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": "Your job is to compare the existing list of facts to the new fact and determine if the existing list sufficiently represents the new one or not."
        }],
        messages=[
            {
                "role": "user",
                "content": f'''
<existing_facts>
{existing_facts}
</existing_facts>

<new_fact>
{new_fact}
</new_fact>

If you believe the new fact is sufficiently new given the ones in the list, output true. If not, output false. Do not provide extra commentary, only output a boolean value.'''
            }
        ]
    )
    return response.content[0].text.strip().lower() == "true"
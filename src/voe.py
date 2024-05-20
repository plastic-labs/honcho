from typing import List
from mirascope.anthropic import AnthropicCall, AnthropicCallParams


# user prediction thought + additional data to improve prediction
class UserPredictionThought(AnthropicCall):
    prompt_template = """
    USER:
    Generate a "thought" that makes a theory of mind prediction about what the user will say based on the way the conversation has been going. Also list other pieces of data that would help improve your prediction.

    Conversation:
    ```
    {chat_history}
    ```

    Generate the additional pieces of data as a numbered list.

    ASSISTANT:
    Thought:
    """

    chat_history: str
    call_params = AnthropicCallParams(model="claude-3-opus-20240229", temperature=0.4)

# user prediction thought revision given context
class UserPredictionThoughtRevision(AnthropicCall):
    prompt_template = """
    USER:
    You are tasked with revising theory of mind "thoughts" about what the user is going to say. Here is the thought generated previously:

    Thought: ```
    {user_prediction_thought}
    ```

    Based on this thought, the following personal data has been retrieved:

    Personal Data: ```
    {retrieved_context}
    ```

    And here's the conversation history that was used to generate the original thought:

    History: ```
    {chat_history}
    ```
    
    Given the thought, conversation history, and personal data, revise the thought. If there are no changes to be made, output "None".

    ASSISTANT:
    thought revision:
    """
    user_prediction_thought_revision: str
    retrieved_context = str
    chat_history: str
    call_params = AnthropicCallParams(model="claude-3-opus-20240229", temperature=0.4)

# VoE thought
class VoeThought(AnthropicCall):
    prompt_template = """
    USER:
    Below is a "thought" about what the user was going to say, and then what the user actually said. Generate a theory of mind prediction about the user based on the difference between the "thought" and actual response.

    Thought: ```
    {user_prediction_thought_revision}
    ```

    Actual: ```
    {actual}
    ```

    Provide the theory of mind prediction solely in reference to the Actual statement, i.e. do not generate something that negates the thought. Do not speculate anything about the user.
    """
    user_prediction_thought_revision: str
    actual: str
    call_params = AnthropicCallParams(model="claude-3-opus-20240229", temperature=0.4)

# VoE derive facts
class VoeDeriveFacts(AnthropicCall):
    prompt_template = """
    USER:
    Below is the most recent AI message we sent to a user, a "thought" about what the user was going to say to that, what the user actually responded with, and a theory of mind prediction about the user's response. Derive a fact (or list of facts) about the user based on the difference between the original thought and their actual response plus the theory of mind prediction about that response.

    Most recent AI message: ```
    {ai_message}
    ```

    Thought about what they were going to say: ```
    {user_prediction_thought_revision}
    ```

    Actual response: ```
    {actual}
    ```

    Theory of mind prediction about that response: ```
    {voe_thought}
    ```

    Provide the fact(s) solely in reference to the Actual response and theory of mind prediction about that response; i.e. do not derive a fact that negates the thought about what they were going to say. Do not speculate anything about the user. Each fact must contain enough specificity to stand alone. If there are many facts, list them out. Your response should be a numbered list with each item on a new line, for example: `\n\n1. foo\n\n2. bar\n\n3. baz`. If there's nothing to derive (i.e. the statements are sufficiently similar), print "None".
    """
    user_prediction_thought_revision: str
    actual: str
    voe_thought: str
    call_params = AnthropicCallParams(model="claude-3-opus-20240229", temperature=0.4)

# check dups
class CheckVoeList(AnthropicCall):
    prompt_template = """
    USER:
    Please compare the following two lists and keep only unique items:

    Old: ```
    {existing_facts}
    ```

    New: ```
    {facts}
    ```
    
    Remove redundant information from the new list and output the remaining facts. Your response should be a numbered list with each fact on a new line, for example: `\n\n1. foo\n\n2. bar\n\n3. baz`. If there's nothing to remove (i.e. the statements are sufficiently different), print "None".
    """
    existing_facts: List[str]
    facts: List[str]
    call_params = AnthropicCallParams(model="claude-3-opus-20240229", temperature=0.4)

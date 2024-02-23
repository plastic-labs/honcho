import os
from typing import List, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, load_prompt
from langchain_core.messages import AIMessage, HumanMessage

from honcho import Message

load_dotenv()

# langchain prompts
SYSTEM_STATE_COMMENTARY = load_prompt(os.path.join(os.path.dirname(__file__), 'langchain_prompts/state_commentary.yaml'))
SYSTEM_STATE_LABELING = load_prompt(os.path.join(os.path.dirname(__file__), 'langchain_prompts/state_labeling.yaml'))
SYSTEM_STATE_CHECK = load_prompt(os.path.join(os.path.dirname(__file__), 'langchain_prompts/state_check.yaml'))

# quick utility function to convert messages from honcho to langchain
def langchain_message_converter(messages: List[Message]) -> List[Union[AIMessage, HumanMessage]]:
    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


# convert chat history and user input into a string
def format_chat_history(chat_history: List[Message], user_input=None):
    messages = [("user: " + message.content if isinstance(message, HumanMessage) else "ai: " + message.content) for message in chat_history]
    if user_input:
        messages.append(f"user: {user_input}")

    return "\n".join(messages)



class StateExtractor:
    """Wrapper class for all the DSPy and LangChain code for user state labeling and pipeline optimization"""
    lc_gpt_4: ChatOpenAI = ChatOpenAI(model_name = "gpt-4")
    lc_gpt_turbo: ChatOpenAI = ChatOpenAI(model_name = "gpt-3.5-turbo")
    system_state_commentary: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_STATE_COMMENTARY)
    system_state_labeling: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_STATE_LABELING)
    system_state_check: SystemMessagePromptTemplate = SystemMessagePromptTemplate(prompt=SYSTEM_STATE_CHECK)

    def __init__(self) -> None:
        pass

    @classmethod
    async def generate_state_commentary(cls, existing_states: List[str], chat_history: List[Message], input: str) -> str:
        """Generate a commentary on the current state of the user"""
        # format existing states
        existing_states = "\n".join(existing_states)
        # format prompt
        state_commentary = ChatPromptTemplate.from_messages([
            cls.system_state_commentary
        ])
        # LCEL
        chain = state_commentary | cls.lc_gpt_4
        # inference
        response = await chain.ainvoke({
            "chat_history": chat_history,
            "user_input": input,
            "existing_states": existing_states,
        })
        # return output
        return response.content
    
    @classmethod
    async def generate_state_label(cls, existing_states: List[str],  state_commentary: str) -> str:
        """Generate a state label from a commetary on the user's state"""
        # format existing states
        existing_states = "\n".join(existing_states)
        # format prompt
        state_labeling = ChatPromptTemplate.from_messages([
            cls.system_state_labeling,
        ])
        # LCEL
        chain = state_labeling | cls.lc_gpt_4
        # inference
        response = await chain.ainvoke({
            "state_commentary": state_commentary,
            "existing_states": existing_states,
        })

        # strip anything that's not letters
        clean_response =  ''.join(c for c in response.content if c.isalpha())
        # return output
        return clean_response
    
    @classmethod
    async def check_state_exists(cls, existing_states: List[str], state: str):
        """Check if a user state is new or already is stored"""

        # convert existing_states to a formatted string
        existing_states = "\n".join(existing_states)

        # format prompt
        state_check = ChatPromptTemplate.from_messages([
            cls.system_state_check
        ])
        # LCEL
        chain = state_check | cls.lc_gpt_turbo
        # inference
        response = await chain.ainvoke({
            "existing_states": existing_states,
            "state": state,
        })
        # return output
        return response.content

    @classmethod
    async def generate_state(cls, existing_states: List[str], chat_history: List[Message], input: str):
        """"Determine the user's state from the current conversation state"""

        # Generate label
        state_commentary = await cls.generate_state_commentary(existing_states, chat_history, input)
        state_label = await cls.generate_state_label(existing_states, state_commentary)

        # Determine if state is new
        # if True, it doesn't exist, state is new
        # if False, it does exist, state is not new, existing_state was returned
        existing_state = await cls.check_state_exists(existing_states, state_label)
        is_state_new = existing_state == "None"  

        # return existing state if we found one
        if is_state_new:
            return is_state_new, state_label
        else:
            return is_state_new, existing_state

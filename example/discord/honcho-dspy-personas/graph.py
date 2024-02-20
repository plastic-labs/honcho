import os
import dspy
from typing import List
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv
from chain import StateExtractor, format_chat_history

from honcho import Message, Session

load_dotenv()

# Configure DSPy
dspy_gpt4 = dspy.OpenAI(model="gpt-4")
dspy.settings.configure(lm=dspy_gpt4)



# DSPy Signatures
class Thought(dspy.Signature):
    """Generate a thought about the user's needs"""
    user_input = dspy.InputField()
    thought = dspy.OutputField(desc="a prediction about the user's mental state")

class Response(dspy.Signature):
    """Generate a response for the user based on the thought provided"""
    user_input = dspy.InputField()
    thought = dspy.InputField()
    response = dspy.OutputField(desc="keep the conversation going, be engaging")

# DSPy Module
class ChatWithThought(dspy.Module):
    generate_thought = dspy.Predict(Thought)
    generate_response = dspy.Predict(Response)

    def forward(self, user_message: Message, session: Session, chat_input: str):
        session.create_message(is_user=True, content=chat_input)

        # call the thought predictor
        thought = self.generate_thought(user_input=chat_input)
        session.create_metamessage(user_message, metamessage_type="thought", content=thought.thought)

        # call the response predictor
        response = self.generate_response(user_input=chat_input, thought=thought.thought)
        session.create_message(is_user=False, content=response.response)

        return response.response
    
user_state_storage = {}
async def chat(user_message: Message, session: Session, chat_history: List[Message], input: str, optimization_threshold=5):
    # first we need to take the user input and determine the user's state/dimension/persona
    is_state_new, user_state = await StateExtractor.generate_state(chat_history, input)

    # Save the user_state if it's new
    if is_state_new:
        user_state_storage[user_state] = {
            "chat_module": ChatWithThought(),
            "examples": []
        }

    # then, we need to select the pipeline for that derived state/dimension/persona
        # way this would work is to define the optimizer and optimize a chain once examples in a certain dimension exceed a threshold
        # need a way to store the optimized chain and call it given a state/dimension/persona
        # this is the reward model for a user within a state/dimension/persona
    user_state_data = user_state_storage[user_state]

    # Optimize the state's chat module if we've reached the optimization threshold
    examples = user_state_data["examples"]
    if len(examples) >= optimization_threshold:
        metric = None # TODO: Define this

        # Optimize chat module
        optimizer = BootstrapFewShot(metric=metric)
        compiled_chat_module = optimizer.compile(trainset=examples)

        user_state_data["chat_module"] = compiled_chat_module

    # use that pipeline to generate a response
    chat_module = user_state_data["chat_module"]
    chat_input = format_chat_history(chat_history, user_input=input)

    response = chat_module(user_message=user_message, session=session, input=chat_input)

    return response

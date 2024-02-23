import os
import dspy
from dspy import Example
from typing import List, Optional
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dotenv import load_dotenv
from chain import StateExtractor, format_chat_history
from response_metric import metric

from honcho import Message, Session

load_dotenv()

# Configure DSPy
dspy_gpt4 = dspy.OpenAI(model="gpt-4", max_tokens=1000)
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

    def forward(
        self,
        chat_input: str,
        user_message: Optional[Message] = None,
        session: Optional[Session] = None,
        assessment_dimension = None,
    ):
        # call the thought predictor
        thought = self.generate_thought(user_input=chat_input)

        if session and user_message:
            session.create_metamessage(
                user_message, metamessage_type="thought", content=thought.thought
            )

        # call the response predictor
        response = self.generate_response(
            user_input=chat_input, thought=thought.thought
        )

        return response


async def chat(
    user_message: Message,
    session: Session,
    chat_history: List[Message],
    input: str,
    optimization_threshold=3,
):
    user_state_storage = dict(session.user.metadata)
    # first we need to see if the user has any existing states
    existing_states = list(user_state_storage.keys())

    # then we need to take the user input and determine the user's state/dimension/persona
    is_state_new, user_state = await StateExtractor.generate_state(
        existing_states=existing_states, chat_history=chat_history, input=input
    )
    print(f"USER STATE: {user_state}")
    print(f"IS STATE NEW: {is_state_new}")

    # add metamessage to message to keep track of what label got assigned to what message
    if session and user_message:
        session.create_metamessage(
            user_message, metamessage_type="user_state", content=user_state
        )

    user_chat_module = ChatWithThought()

    # TODO: you'd want to initialize user state object from Honcho
    # Save the user_state if it's new
    if is_state_new:
        user_state_storage[user_state] = {"chat_module": {}, "examples": []}

    user_state_data = user_state_storage[user_state]

    # Optimize the state's chat module if we've reached the optimization threshold
    # TODO: read in examples from Honcho User Object
    examples = user_state_data["examples"]
    print(f"Num examples: {len(examples)}")
    session.user.update(metadata=user_state_storage)

    if len(examples) >= optimization_threshold:
        # convert example from dicts to dspy Example objects
        examples = [dspy.Example(**example).with_inputs("chat_input", "ai_response", "assessment_dimension") for example in examples]
        print(examples)
        # Splitting the examples list into train and validation sets
        # train_examples = examples[:-1]  # All but the last item for training
        # val_examples = examples[-1:]  # The last item for validation

        # Optimize chat module
        optimizer = BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=3, max_labeled_demos=3, num_candidate_programs=10, num_threads=4)
        # compiled_chat_module = optimizer.compile(ChatWithThought(), trainset=train_examples, valset=val_examples)
        compiled_chat_module = optimizer.compile(user_chat_module, trainset=examples)
        print(f"COMPILED_CHAT_MODULE: {compiled_chat_module}")

        # user_state_data["chat_module"] = compiled_chat_module.dump_state()
        user_state_storage[user_state][
            "chat_module"
        ] = compiled_chat_module.dump_state()
        print(f"DUMPED_STATE: {compiled_chat_module.dump_state()}")
        user_chat_module = compiled_chat_module

        # save to file for debugging purposes
        # compiled_chat_module.save("module.json")
        # Update User in Honcho
        session.user.update(metadata=user_state_storage)

    # use that pipeline to generate a response
    chat_input = format_chat_history(chat_history, user_input=input)
    response = user_chat_module(
        user_message=user_message, session=session, chat_input=chat_input
    )
    # remove ai prefix
    response = response.response.replace("ai:", "").strip()
    dspy_gpt4.inspect_history(n=2)

    return response

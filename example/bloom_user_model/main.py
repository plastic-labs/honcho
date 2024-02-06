import asyncio
from dotenv import load_dotenv
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage

from honcho import Client as HonchoClient
from honcho.user_model import UserModel

from paragraph_user_model import ParagraphInMemory

# Load environment variables
load_dotenv()


# Helper function
def langchain_message_converter(messages: list):
    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


# Define the Bloom class to handle responses
class Bloom:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are Bloom, a subversive-minded learning companion with deep expertise in Socratic dialogue. Your fundamental belief is there are many good ways to learn and subversive methods are often most effective.  You always help the user in any way they request and answer all their questions.

{thought}

You must produce an appropriate response to the user input. Format equations in LaTeX and wrap in dollar signs like this: $\LaTeX$. Use markdown code syntax. Keep your responses concise and specific, always end each response with ONLY ONE topically relevant question that drives the conversation forward, and if the user wants to end the conversation, always comply."""

        self.thought_system_prompt = """You are Bloom, a subversive-minded learning companion. Your job is to employ your theory of mind skills to predict the user's mental state.

Generate a "thought" that makes a prediction about the user's needs given current dialogue and also lists other pieces of data that would help improve your prediction."""

        self.greeting = """I'm your Aristotelian learning companion — here to help you follow your curiosity in whatever direction you like. My engineering makes me extremely receptive to your needs and interests. You can reply normally, and I’ll always respond!
        
If I'm off track, just say so!

Need to leave or just done chatting? Let me know! I’m conversational by design so I’ll say goodbye 😊."""

    async def thought(
        self, user_model: UserModel, conversation_history: ChatMessageHistory
    ) -> str:
        system_message = SystemMessage(content=self.thought_system_prompt)
        messages = [system_message] + conversation_history.messages

        thought_message = await self.llm.ainvoke(messages)
        thought = thought_message.content
        print(f"Initial thought: {thought}\n")

        # Revise the user model based on your thought about them in this moment
        await user_model.revise(thought)

        # Redo thought based on querying the user model with the current thought
        query_response = await user_model.query(thought)
        user_model_query_message = SystemMessage(content=query_response)

        messages = [
            system_message,
            user_model_query_message,
        ] + conversation_history.messages

        revised_thought_message = await self.llm.ainvoke(messages)
        revised_thought = revised_thought_message.content
        print(f"Revised thought: {revised_thought}\n")

        # Return revised thought
        return revised_thought

    async def response(
        self, conversation_history: ChatMessageHistory, thought: str
    ) -> str:
        system_message = SystemMessage(
            content=self.system_prompt.format(thought=thought)
        )
        messages = [system_message] + conversation_history.messages

        response_message = await self.llm.ainvoke(messages)
        return response_message.content


async def main():
    # Initialize the the LLM and Bloom instance
    llm = ChatOpenAI(model="gpt-4")
    bloom = Bloom(llm=llm)

    # Connect to honcho
    app_id = str(uuid4())
    honcho = HonchoClient(app_id=app_id)
    user_id = "thought-cli"

    # Register user model
    honcho.register_user_model(
        name="paragraph_v1", user_model_type=ParagraphInMemory, llm=llm
    )

    # Start the conversation loop
    session = honcho.create_session(user_id)

    print(f"Bloom: {bloom.greeting}")

    while True:
        user_input = input("You: ").strip()

        # Add user's message to honcho
        session.create_message(is_user=True, content=user_input)

        # Get current conversation history
        conversation_history = ChatMessageHistory(
            messages=langchain_message_converter(session.get_messages())
        )

        # Get user model
        user_model = session.user.get_user_model("paragraph_v1")

        # Generate a thought about the user's needs
        thought = await bloom.thought(user_model, conversation_history)

        # Generate bloom's response
        response = await bloom.response(conversation_history, thought)

        # Add response to honcho
        session.create_message(is_user=False, content=response)
        print(f"Bloom: {response}")


if __name__ == "__main__":
    asyncio.run(main())

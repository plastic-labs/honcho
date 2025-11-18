"""
LangChain Integration with Honcho and OpenAI

This module demonstrates how to build a stateful conversational AI agent using
LangChain chains with prompt templates and Honcho for memory management.
"""
import os
from dotenv import load_dotenv
from honcho import Honcho
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Initialize Honcho for memory management
honcho = Honcho(environment="local")

# Initialize LangChain's OpenAI chat model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create prompt template with message history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history")
])

# Create chain: Prompt -> LLM
chain = prompt | llm


def run_conversation(user_id: str, user_input: str, session_id: str = None):
    """
    Run a conversation turn with Honcho memory management.

    Args:
        user_id: Unique identifier for the user
        user_input: The user's message
        session_id: Optional session ID (defaults to user_id based session)

    Returns:
        The assistant's response string
    """
    if not session_id:
        session_id = f"session_{user_id}"

    # Get peers and session
    user_peer = honcho.peer(user_id)
    assistant_peer = honcho.peer("assistant")
    session = honcho.session(session_id)

    # Store user message in Honcho
    session.add_messages([user_peer.message(user_input)])

    # Get context in OpenAI format from Honcho
    messages = session.get_context().to_openai(assistant=assistant_peer)

    # Invoke the chain with honcho's context
    # MessagesPlaceholder accepts OpenAI dict format directly
    response = chain.invoke({"history": messages})
    assistant_response = response.content

    # Store assistant response in Honcho
    session.add_messages([assistant_peer.message(assistant_response)])

    return assistant_response


if __name__ == "__main__":
    print("Welcome to the AI Assistant! How can I help you today?")
    print("(Powered by LangChain + Honcho)")
    print("-" * 50)

    user_id = "test-user-1234"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        response = run_conversation(user_id, user_input)
        print(f"Assistant: {response}")

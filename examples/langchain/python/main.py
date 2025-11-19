"""
LangGraph Integration with Honcho and OpenAI

This module demonstrates how to build a stateful conversational AI agent using
LangGraph for orchestration, OpenAI for the AI model, and Honcho for memory
management. It creates a chatbot that remembers conversations across sessions.
"""

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from honcho import Honcho, Peer, Session
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
load_dotenv()

honcho = Honcho()

llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class State(TypedDict):
    user_message: str
    assistant_response: str
    user: Peer
    assistant: Peer
    session: Session

def chatbot(state: State):
    user_message = state["user_message"]

    # Get objects from state
    user = state["user"]
    assistant = state["assistant"]
    session = state["session"]
    session.add_messages([user.message(user_message)])

    # Get context in OpenAI format
    messages = session.get_context().to_openai(assistant=assistant)

    # Generate response
    response = llm.chat.completions.create(
        model="gpt-5.1",
        messages=messages
    )
    assistant_response = response.choices[0].message.content

    # Store assistant response
    session.add_messages([assistant.message(assistant_response)])

    return {"assistant_response": assistant_response}

graph = StateGraph(State) \
    .add_node("chatbot", chatbot) \
    .add_edge(START, "chatbot") \
    .add_edge("chatbot", END) \
    .compile()

def run_conversation_turn(user_id: str, user_input: str, session_id: str | None = None):
    if not session_id:
        session_id = f"session_{user_id}"

    # Initialize Honcho objects
    user = honcho.peer(user_id)
    assistant = honcho.peer("assistant")
    session = honcho.session(session_id)

    result = graph.invoke({
        "user_message": user_input,
        "user": user,
        "assistant": assistant,
        "session": session
    })

    return result["assistant_response"]

if __name__ == "__main__":
  print("Welcome to the AI Assistant! How can I help you today?")
  user_id = "test-user-1234"
  while True:
      user_input = input("You: ")
      if user_input.lower() in ['quit', 'exit']:
          break
      response = run_conversation_turn(user_id, user_input)
      print(f"Assistant: {response}\n")

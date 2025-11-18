"""
LangGraph Integration with Honcho and OpenAI

This module demonstrates how to build a stateful conversational AI agent using
LangGraph for orchestration, OpenAI for the AI model, and Honcho for memory
management. It creates a chatbot that remembers conversations across sessions.
"""

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from honcho import Honcho
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
load_dotenv()

honcho = Honcho(
    environment="local"
)

llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class State(TypedDict):
    user_input: str
    assistant_response: str
    user_id: str
    session_id: str

def chatbot(state: State):
    user_message = state["user_input"]

    # Get peers and session
    user_peer = honcho.peer(state["user_id"])
    assistant_peer = honcho.peer("assistant")
    session = honcho.session(state["session_id"])
    session.add_messages([user_peer.message(user_message)])

    # Get context in OpenAI format
    messages = session.get_context().to_openai(assistant=assistant_peer)

    # Generate response
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    assistant_response = response.choices[0].message.content

    # Store assistant response
    session.add_messages([assistant_peer.message(assistant_response)])

    return {"assistant_response": assistant_response}

graph = StateGraph(State) \
    .add_node("chatbot", chatbot) \
    .add_edge(START, "chatbot") \
    .add_edge("chatbot", END) \
    .compile()

def run_conversation(user_id: str, user_input: str, session_id: str = None):
    if not session_id:
        session_id = f"session_{user_id}"

    result = graph.invoke({
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id
    })

    return result["assistant_response"]

if __name__ == "__main__":
  print("Welcome to the AI Assistant! How can I help you today?")
  user_id = "test-user-1234"
  while True:
      user_input = input("You: ")
      if user_input.lower() in ['quit', 'exit']:
          break
      response = run_conversation(user_id, user_input)
      print(f"Assistant: {response}\n")

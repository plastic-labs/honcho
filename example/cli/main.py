from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models.fake import FakeListChatModel

from honcho import Client as HonchoClient
# from client import HonchoClient

honcho = HonchoClient(base_url="http://localhost:8000")

responses = ["Fake LLM Response :)"]
llm = FakeListChatModel(responses=responses)
system = SystemMessage(content="You are world class technical documentation writer. Be as concise as possible")

user = "CLI-Test"
session = honcho.create_session(user_id=user)
session_id = session["id"]

def langchain_message_converter(messages: List):
    new_messages = []
    for message in messages:
        if message["is_user"]:
            new_messages.append(HumanMessage(content=message["content"]))
        else:
            new_messages.append(AIMessage(content=message["content"]))
    return new_messages

def chat():
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            honcho.delete_session(user, session_id)
            break
        user_message = HumanMessage(content=user_input)
        history = honcho.get_messages_for_session(user, session_id)
        langchain_history = langchain_message_converter(history)
        prompt = ChatPromptTemplate.from_messages([
                system,
                *langchain_history,
                user_message
            ])
        chain = prompt | llm
        response = chain.invoke({})
        print(type(response))
        print(f"AI: {response.content}")
        honcho.create_message_for_session(user, session_id, is_user=True, content=user_input)
        honcho.create_message_for_session(user, session_id, is_user=False, content=response.content)

chat()

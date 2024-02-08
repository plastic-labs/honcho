from typing import List
from uuid import uuid4

from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models.fake import FakeListChatModel

from honcho import Client as HonchoClient

app_id = str(uuid4())

honcho = HonchoClient(app_id=app_id)

responses = ["Fake LLM Response :)"]
llm = FakeListChatModel(responses=responses)
system = SystemMessage(
    content="You are world class technical documentation writer. Be as concise as possible"
)

user = "CLI-Test"
session = honcho.create_session(user_id=user)


def langchain_message_converter(messages: List):
    new_messages = []
    for message in messages:
        if message.is_user:
            new_messages.append(HumanMessage(content=message.content))
        else:
            new_messages.append(AIMessage(content=message.content))
    return new_messages


def chat():
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            session.delete()
            break
        user_message = HumanMessage(content=user_input)
        history = session.get_messages()
        langchain_history = langchain_message_converter(history)
        prompt = ChatPromptTemplate.from_messages(
            [system, *langchain_history, user_message]
        )
        chain = prompt | llm
        response = chain.invoke({})
        print(type(response))
        print(f"AI: {response.content}")
        session.create_message(is_user=True, content=user_input)
        session.create_message(is_user=False, content=response.content)


chat()

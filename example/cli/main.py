from typing import List
from uuid import uuid4

from langchain.prompts import ChatPromptTemplate

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models.fake import FakeListChatModel

from honcho import Honcho
from honcho.ext.langchain import _messages_to_langchain

app_name = str(uuid4())

honcho = Honcho(
    app_name=app_name, base_url="http://localhost:8000"
)  # uncomment to use local
# honcho = Honcho(app_name=app_name)  # uses demo server at https://demo.honcho.dev
honcho.initialize()

responses = ["Fake LLM Response :)"]
llm = FakeListChatModel(responses=responses)
system = SystemMessage(
    content="You are world class technical documentation writer. Be as concise as possible"
)

user_name = "CLI-Test"
user = honcho.create_user(user_name)
session = user.create_session()


def chat():
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            session.close()
            break
        user_message = HumanMessage(content=user_input)
        history = list(session.get_messages_generator())
        langchain_history = _messages_to_langchain(history)
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

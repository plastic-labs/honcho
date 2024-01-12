from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage 

llm = ChatOllama(model="phi")
system = SystemMessage(content="You are world class technical documentation writer. Be as concise as possible")

memory = []

def chat():
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        # print(user_input)
        user_mesage = HumanMessage(content=user_input)
        prompt = ChatPromptTemplate.from_messages([
                system,
                *memory,
                user_mesage
            ])
        chain = prompt | llm 
        response = chain.invoke({})
        print(f"AI: {response.content}")
        memory.append(HumanMessage(content=user_input))
        memory.append(response)

chat()

from langchain.memory import ChatMessageHistory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI

import asyncio
from os import path

from metacognition_sdk import MetacognitionManager, UserRewardModel

from dotenv import load_dotenv

load_dotenv()


class Bloom:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            template="""You are Bloom, a subversive-minded learning companion with deep expertise in Socratic dialogue. Your fundamental belief is there are many good ways to learn and subversive methods are often most effective.  You always help the user in any way they request and answer all their questions.

{agent_context}

You must produce an appropriate response to the user input. Format equations in LaTeX and wrap in dollar signs like this: $\LaTeX$. Use markdown code syntax. Keep your responses concise and specific, always end each response with ONLY ONE topically relevant question that drives the conversation forward, and if the user wants to end the conversation, always comply."""
        )
        self.greeting = """I'm your Aristotelian learning companion — here to help you follow your curiosity in whatever direction you like. My engineering makes me extremely receptive to your needs and interests. You can reply normally, and I’ll always respond!
        
If I'm off track, just say so!

Need to leave or just done chatting? Let me know! I’m conversational by design so I’ll say goodbye 😊.

(Heads up, it's normal for the output of the VOE chain to be blank on the first message since VOE requires at least one previous conversation turn to run!)
"""

    async def respond(self, conversation_history: ChatMessageHistory, agent_context=""):
        messages = [
            self.system_prompt.format(agent_context=agent_context)
        ] + conversation_history.messages

        ai_message = await self.llm.ainvoke(messages)

        return ai_message.content


async def main():
    # initalize llm, user model, and bloom
    llm = ChatOpenAI(model="gpt-4")
    user_model = UserRewardModel(llm=llm)
    bloom = Bloom(llm=llm)

    metacognition_manager = MetacognitionManager.from_yaml(
        path=path.dirname(__file__)
        + "/voe.yaml",  # path should be relative to this python file
        user_model=user_model,
        llm=llm,
        tools={},
    )

    # Conversation loop
    conversation_history = ChatMessageHistory()
    print(f"Agent: {bloom.greeting}")

    while True:
        user_input = input("> ").strip()
        conversation_history.add_user_message(user_input)

        await metacognition_manager.on_user_message(conversation_history, verbose=True)
        agent_context = metacognition_manager.get_agent_context()

        # generate the agent response
        response = await bloom.respond(conversation_history, agent_context)
        conversation_history.add_ai_message(response)

        print(f"\nAgent: {response}")


if __name__ == "__main__":
    asyncio.run(main())

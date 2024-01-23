import asyncio
from os import path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from metacognition_sdk import MetacognitionManager, UserRewardModel
from metacognition_sdk.adapters.llms.openai import ChatOpenAILlmAdapter
from metacognition_sdk.adapters.storage.in_memory import InMemoryUserModelStorageAdapter
from metacognition_sdk.messages import ConversationHistory

# Load environment variables
load_dotenv()


# Define the Bloom class to handle responses
class Bloom:
    def __init__(self, llm: ChatOpenAILlmAdapter):
        self.llm = llm
        self.system_prompt = """You are Bloom, a subversive-minded learning companion with deep expertise in Socratic dialogue. Your fundamental belief is there are many good ways to learn and subversive methods are often most effective.  You always help the user in any way they request and answer all their questions.

{agent_context}

You must produce an appropriate response to the user input. Format equations in LaTeX and wrap in dollar signs like this: $\LaTeX$. Use markdown code syntax. Keep your responses concise and specific, always end each response with ONLY ONE topically relevant question that drives the conversation forward, and if the user wants to end the conversation, always comply."""
        self.greeting = """I'm your Aristotelian learning companion — here to help you follow your curiosity in whatever direction you like. My engineering makes me extremely receptive to your needs and interests. You can reply normally, and I’ll always respond!
        
If I'm off track, just say so!

Need to leave or just done chatting? Let me know! I’m conversational by design so I’ll say goodbye 😊.

(Heads up, it's normal for the output of the VOE chain to be blank on the first message since VOE requires at least one previous conversation turn to run!)
"""

    async def generate_response(
        self, conversation_history: ConversationHistory, agent_context: str
    ) -> str:
        system_prompt = self.system_prompt.format(agent_context=agent_context)
        return await self.llm.chat_inference(conversation_history, system_prompt)


async def main():
    # Initialize the OpenAI client and Bloom instance
    openai_client = AsyncOpenAI()
    llm = ChatOpenAILlmAdapter(openai_client, model="gpt-4")
    bloom = Bloom(llm=llm)

    # Initialize user model and metacognition manager
    storage_adapter = InMemoryUserModelStorageAdapter()  # Store user models in memory
    user_model = UserRewardModel(llm=llm, user_model_storage_adapter=storage_adapter)
    metacognition_manager = MetacognitionManager.from_yaml(
        path=path.join(
            path.dirname(__file__), "voe.yaml"
        ),  # make sure the path is relative to this python file
        user_model=user_model,
        llm=llm,
        verbose=True,
    )

    # Start the conversation loop
    conversation_history = ConversationHistory()
    print(f"Bloom: {bloom.greeting}")

    while True:
        user_input = input("You: ").strip()
        conversation_history.add_user_message(user_input)

        # Process the user message with the metacognition manager
        await metacognition_manager.on_user_message(conversation_history)
        agent_context = metacognition_manager.get_agent_context()

        # Generate and display the response from Bloom
        response = await bloom.generate_response(conversation_history, agent_context)
        conversation_history.add_ai_message(response)
        print(f"\nBloom: {response}")


if __name__ == "__main__":
    asyncio.run(main())

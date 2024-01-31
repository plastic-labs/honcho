import asyncio
from os import path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from honcho.architecture.adapters.langchain import LangchainAdapter
from honcho.user_model.adapters.pickle import PickleUserModelStorageAdapter
from honcho.architecture.messages import ConversationHistory

from honcho import Client as HonchoClient

# Load environment variables
load_dotenv()


# Define the Bloom class to handle responses
class Bloom:
    def __init__(self, llm: LangchainAdapter):
        self.llm = llm
        self.system_prompt = """You are Bloom, a subversive-minded learning companion with deep expertise in Socratic dialogue. Your fundamental belief is there are many good ways to learn and subversive methods are often most effective.  You always help the user in any way they request and answer all their questions.

Here is some potentially useful context about the student: "{agent_context}"

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
    # Initialize the the LLM and Bloom instance
    llm = LangchainAdapter(ChatOpenAI(model="gpt-4"))
    bloom = Bloom(llm=llm)

    # Connect to honcho
    honcho = HonchoClient(base_url="http://localhost:8000")
    user_id = "bloom-voe-cli"

    # Register VOE
    honcho.register_metacognitive_architecture(
        path=path.join(
            path.dirname(__file__), "voe.yaml"
        ),  # make sure the path is relative to this python file
        user_model_storage_adapter_type=PickleUserModelStorageAdapter,
        llm=llm,
        verbose=True,
    )

    # Start the conversation loop
    session = honcho.create_session(user_id)
    session_id = session["id"]

    print(f"Bloom: {bloom.greeting}")

    while True:
        user_input = input("You: ").strip()
        honcho.create_message_for_session(
            user_id=user_id, session_id=session_id, is_user=True, content=user_input
        )

        # Wait for the VOE generated context for Bloom
        agent_context = await honcho.get_context(user_id)

        # Generate Bloom's response
        conversation_history = ConversationHistory.from_honcho_dicts(
            honcho.get_messages_for_session(user_id, session_id)
        )
        response = await bloom.generate_response(conversation_history, agent_context)
        honcho.create_message_for_session(
            user_id=user_id, session_id=session_id, is_user=False, content=response
        )
        print(f"\nBloom: {response}")


if __name__ == "__main__":
    asyncio.run(main())

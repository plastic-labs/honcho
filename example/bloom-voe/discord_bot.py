import os
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI

from honcho.architecture.adapters.openai import ChatOpenAILlmAdapter
from honcho.user_model.adapters.pickle import PickleUserModelStorageAdapter
from honcho.architecture.messages import ConversationHistory
from honcho import Client as HonchoClient
from honcho import LRUCache

# Load environment variables
load_dotenv()


# Define the Bloom class to handle responses
class Bloom:
    def __init__(self, llm: ChatOpenAILlmAdapter):
        self.llm = llm
        self.system_prompt = """You are Bloom, a subversive-minded learning companion with deep expertise in Socratic dialogue. Your fundamental belief is there are many good ways to learn and subversive methods are often most effective. You always help the user in any way they request and answer all their questions.

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


# Initialize the OpenAI client and Bloom instance
openai_client = AsyncOpenAI()
llm = ChatOpenAILlmAdapter(openai_client, model="gpt-4")
bloom = Bloom(llm=llm)

# Connect to honcho
honcho = HonchoClient(base_url="http://localhost:8000")

# Register VOE
honcho.register_metacognitive_architecture(
    path=os.path.join(os.path.dirname(__file__), "voe.yaml"),
    user_model_storage_adapter_type=PickleUserModelStorageAdapter,
    llm=llm,
    verbose=True,
)

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

CACHE = LRUCache(50)  # Support 50 concurrent active conversations cached in memory

bot = discord.Bot(intents=intents)


@bot.event
async def on_ready():
    print(f"Bloom Bot has logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = f"discord_{str(message.author.id)}"
    location_id = str(message.channel.id)
    session_id = get_or_create_session(user_id, location_id)

    print(f"Responding to '{message.author.name}'")
    async with message.channel.typing():
        await process_user_message(message, session_id)

    print(f"Responded to '{message.author.name}'")


async def process_user_message(message, session_id):
    user_id = f"discord_{str(message.author.id)}"
    honcho.create_message_for_session(
        user_id=user_id, session_id=session_id, is_user=True, content=message.content
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

    await message.channel.send(response)


def get_or_create_session(user_id, location_id) -> int:
    key = f"{user_id}+{location_id}"
    session_id = CACHE.get(key)
    if session_id is None:
        session = honcho.create_session(user_id, location_id)
        session_id = session["id"]
        CACHE.put(key, session_id)
    return session_id


@bot.slash_command(name="restart", description="Restart the Conversation")
async def restart_conversation(ctx):
    user_id = f"discord_{str(ctx.author.id)}"
    location_id = str(ctx.channel.id)
    key = f"{user_id}+{location_id}"
    session_id = CACHE.get(key)
    if session_id is not None:
        honcho.delete_session(user_id, session_id)
    session = honcho.create_session(user_id, location_id)
    session_id = session["id"]
    CACHE.put(key, session_id)
    msg = (
        "Great! The conversation has been restarted. What would you like to talk about?"
    )
    honcho.create_message_for_session(user_id, session_id, False, msg)
    await ctx.send(msg)


bot.run(os.environ["DISCORD_BOT_TOKEN"])

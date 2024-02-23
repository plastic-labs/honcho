import os
from uuid import uuid4
import discord
from dotenv import load_dotenv

from honcho import Honcho

load_dotenv()

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

app_name = str(uuid4())

# honcho = Honcho(app_name=app_name, base_url="http://localhost:8000") # uncomment to use local
honcho = Honcho(app_name=app_name)  # uses demo server at https://demo.honcho.dev
honcho.initialize()

bot = discord.Bot(intents=intents)


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = f"discord_{str(message.author.id)}"
    user = honcho.get_or_create_user(user_id)
    location_id = str(message.channel.id)

    sessions = list(user.get_sessions_generator(location_id))
    if len(sessions) > 0:
        session = sessions[0]
    else:
        session = user.create_session(location_id)

    inp = message.content
    session.create_message(is_user=True, content=inp)
    async with message.channel.typing():
        output = "Fake LLM Message"
        await message.channel.send(output)

    session.create_message(is_user=False, content=output)


@bot.slash_command(name="restart", description="Restart the Conversation")
async def restart(ctx):
    user_id = f"discord_{str(ctx.author.id)}"
    user = honcho.get_or_create_user(user_id)
    location_id = str(ctx.channel_id)
    sessions = list(user.get_sessions_generator(location_id))
    sessions[0].close() if len(sessions) > 0 else None

    await ctx.respond(
        "Great! The conversation has been restarted. What would you like to talk about?"
    )


bot.run(os.environ["BOT_TOKEN"])

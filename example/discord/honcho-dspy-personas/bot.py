import os
from uuid import uuid1
import discord
from honcho import Client as HonchoClient
from graph import chat
from chain import langchain_message_converter

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True
intents.reactions = True  # Enable reactions intent

# app_id = str(uuid1())
app_id = "vince-dspy-personas"

#honcho = HonchoClient(app_id=app_id, base_url="http://localhost:8000") # uncomment to use local
honcho = HonchoClient(app_id=app_id) # uses demo server at https://demo.honcho.dev

bot = discord.Bot(intents=intents)

thumbs_up_messages = []  
thumbs_down_messages = []  

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.event
async def on_member_join(member):
    await member.send(
        f"*Hello {member.name}, welcome to the server! This is a demo bot built with Honcho,* "
        "*implementing a naive user modeling method.* "
        "*To get started, just type a message in this channel and the bot will respond.* "
        "*Over time, it will classify the \"state\" you're in and optimize conversations based on that state.* "
        "*You can use the /restart command to restart the conversation at any time.* "
        "*If you have any questions or feedback, feel free to ask in the #honcho channel.* "
        "*Enjoy!*"
    )
    
    
@bot.event
async def on_message(message):
    if message.author == bot.user or message.guild is not None:
        return

    user_id = f"discord_{str(message.author.id)}"
    location_id=str(message.channel.id)

    sessions = list(honcho.get_sessions_generator(user_id, location_id))

    if len(sessions) > 0:
        session = sessions[0]
    else:
        session = honcho.create_session(user_id, location_id)

    history = list(session.get_messages_generator())[:5]
    chat_history = langchain_message_converter(history)

    inp = message.content
    user_message = session.create_message(is_user=True, content=inp)

    async with message.channel.typing():
        response = await chat(
            chat_history=chat_history,
            user_message=user_message,
            session=session,
            input=inp
        )
        await message.channel.send(response)

    session.create_message(is_user=False, content=response)

@bot.event
async def on_reaction_add(reaction, user):
    # Ensure the bot does not react to its own reactions
    if user == bot.user:
        return
    
    user_id = f"discord_{str(reaction.message.author.id)}"
    location_id = str(reaction.message.channel.id)

    # Check if the reaction is a thumbs up
    if str(reaction.emoji) == '👍':
        thumbs_up_messages.append(reaction.message.content)
        print(f"Added to thumbs up: {reaction.message.content}")
    # Check if the reaction is a thumbs down
    elif str(reaction.emoji) == '👎':
        thumbs_down_messages.append(reaction.message.content)
        print(f"Added to thumbs down: {reaction.message.content}")

    # TODO: we need to append these to the examples list within the user state json object
    # append example
    # example = Example(chat_input=chat_input, assessment_dimension=user_state, response=response).with_inputs('chat_input')
    # examples.append(example)
    # user_state_storage[user_state]["examples"] = examples


@bot.slash_command(name = "restart", description = "Restart the Conversation")
async def restart(ctx):
    user_id=f"discord_{str(ctx.author.id)}"
    location_id=str(ctx.channel_id)
    sessions = list(honcho.get_sessions_generator(user_id, location_id))
    sessions[0].close() if len(sessions) > 0 else None

    msg = "Great! The conversation has been restarted. What would you like to talk about?"
    await ctx.respond(msg)

bot.run(os.environ["BOT_TOKEN"])

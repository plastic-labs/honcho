# Simple Roast Bot

The goal of this repo is to demonstrate how to deploy an LLM application using Honcho to manage user data. Here we've implemented a simple Discord bot that interacts with OpenAI's GPT-3.5-Turbo model via LangChain. Oh, and also, it's prompted to roast you.

To run locally, follow these steps:

### Clone the Repository

In your desired location, run the following command in your terminal:
```
git clone git@github.com:plastic-labs/honcho.git
```

### Set Up the Virtual Environment

This project uses different Poetry virtual environments. If you're unfamiliar, take a look at their docs [here](https://python-poetry.org/docs/)

```
cd example/discord/simple-roast-bot
poetry shell # Activate virutal environment
poetry install # install dependencies
```

### Create `.env` File

Copy the `.env.template` file to a `.env` file and specify the `BOT_TOKEN` and `OPENAI_API_KEY`. If you've never built a Discord bot before, check out this [`py-cord` guide](https://guide.pycord.dev/getting-started/creating-your-first-bot) to learn more about how to get a `BOT_TOKEN`. You can generate an `OPENAI_API_KEY` in the [OpenAI developer platform](https://platform.openai.com/docs/overview).

```
BOT_TOKEN=
OPENAI_API_KEY=
```

### Run the Bot

If you're not running Honcho locally, you can run the bot with the following command:
```
python main.py
```

If you are interested in running Honcho locally, follow the setup instructions at the root of this repo.

# Honcho Fact Memory

This example contains code for a simple discord bot built with LangChain that's prompted to derive facts it can store from messages you input. It uses Honcho to organize the data storage on a per-user basis so we can code with user-focused mental model.

## Initial Setup

This project uses [Poetry](https://python-poetry.org/) for dependency and virtual environment management. Navigate to this folderand run the following commands:

```
poetry shell
poetry install
```

## Run the Bot

By default, the bot will reference the hosted version of Honcho at https://demo.honcho.dev and store data there temorarily for up to 7 days. If you'd like to run Honcho locally, follow the instructions in the README at the root of this repository.  

Copy the `.env.template` file to a `.env` file and fill out the `BOT_TOKEN` and `OPENAI_API_KEY` values. To run the bot, use the following command:
```
python bot.py
```

If you have any further questions, feel free to join our [Discord server](https://discord.gg/plasticlabs) and ask in the #honcho channel!

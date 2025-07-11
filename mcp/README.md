# Honcho MCP Server

## To run

```bash
uv sync
source .venv/bin/activate
python server.py
```

## To integrate with Claude Desktop

First, create a `.env` file and save your HONCHO_API_KEY there.

Then, run this command:

```bash
uv pip install fastmcp
fastmcp install claude-desktop server.py --env-file .env
```

It is recommended to paste `instructions.md` into your Claude preferences to show it how to use the server for every conversation.

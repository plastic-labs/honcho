# Honcho MCP Server



## To integrate with Claude Desktop

First, create a `.env` file in this folder (`mcp/`) and save a `HONCHO_API_KEY` there. You may optionally use `HONCHO_USER_NAME` and `HONCHO_ASSISTANT_NAME` to identify yourself and Claude.

Then, run these commands:

```bash
uv pip install fastmcp
source .venv/bin/activate
fastmcp install claude-desktop server.py --env-file .env
```

NOTE: uv must be installed and available in your system PATH. Claude Desktop runs in its own isolated environment and needs uv to manage dependencies.
On macOS, it is recommended to install uv globally with Homebrew so that Claude Desktop will detect it: `brew install uv`. Installing uv with other methods may not make it accessible to Claude Desktop.

It is strongly recommended to paste `instructions.md` into your Claude preferences to show it how to use the server for every conversation.


## To run standalone

First, create a `.env` file in this folder (`mcp/`) and save a `HONCHO_API_KEY` there. You may optionally use `HONCHO_USER_NAME` and `HONCHO_ASSISTANT_NAME` to identify yourself and Claude.

Then:
```bash
uv sync
source .venv/bin/activate
python server.py
```
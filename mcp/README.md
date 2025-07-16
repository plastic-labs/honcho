# Honcho MCP Server

This directory contains two ways to use Honcho's MCP functionality:

1. **Hosted MCP Server**
2. **Local MCP Server** - For integration with Claude Desktop and other MCP clients

## Hosted MCP Server

Go to https://app.honcho.dev and get an API key. Then go to Claude Desktop and navigate to custom MCP servers.

If you don't have node/npm/npx install you will need to do that. Claude Desktop or Claude Code can help!

Add or integrate this into your claude desktop config:
```json
{
  "mcpServers": {
    "honcho": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.honcho.dev",
        "--header",
        "Authorization:${AUTH_HEADER}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>"
      }
    }
  }
}
```

Alternatively you may customize your username, assistant name, and/or workspace ID. All are optional.

```json
{
  "mcpServers": {
    "honcho": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.honcho.dev",
        "--header",
        "Authorization:${AUTH_HEADER}",
        "--header",
        "X-Honcho-User-Name:${USER_NAME}",
        "--header",
        "X-Honcho-Assistant-Name:${ASSISTANT_NAME}",
        "--header",
        "X-Honcho-Workspace-ID:${WORKSPACE_ID}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>",
        "USER_NAME": "<your-name>",
        "ASSISTANT_NAME": "<your-assistant-name>",
        "WORKSPACE_ID": "<your-custom-workspace-id>"
      }
    }
  }
}
```

## Local MCP Server

### To integrate with Claude Desktop

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

### To run standalone

First, create a `.env` file in this folder (`mcp/`) and save a `HONCHO_API_KEY` there. You may optionally use `HONCHO_USER_NAME` and `HONCHO_ASSISTANT_NAME` to identify yourself and Claude.

Then:
```bash
uv sync
source .venv/bin/activate
python server.py
```

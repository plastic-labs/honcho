# Honcho MCP Server

This directory contains two ways to use Honcho's MCP functionality:

1. **Local MCP Server** - For integration with Claude Desktop and other MCP clients
2. **Cloudflare Worker Proxy** - For public REST API access

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

## Cloudflare Worker Proxy

For public access via REST API, you can deploy the Honcho MCP functionality as a Cloudflare Worker. This allows users to access the same functionality through HTTP requests with their own API keys.

### Quick Start

1. Navigate to the cloudflare-worker directory:
   ```bash
   cd cloudflare-worker
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure your worker name in `wrangler.toml`

4. Deploy:
   ```bash
   npx wrangler login
   npm run deploy
   ```

### Usage

The deployed worker provides a REST API with these endpoints:

- `POST /start-conversation` - Start a new conversation
- `POST /add-turn` - Add messages to a conversation
- `POST /get-insights` - Get personalization insights

All requests require a `Authorization: Bearer YOUR_HONCHO_API_KEY` header.

### Documentation

See the `cloudflare-worker/` directory for:
- `README.md` - Complete API documentation and usage examples
- `DEPLOYMENT.md` - Step-by-step deployment guide

This provides the same functionality as the MCP server but accessible as a public REST API that users can integrate into any application.
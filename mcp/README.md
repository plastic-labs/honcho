# Honcho MCP Server

A Cloudflare Worker that implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for [Honcho](https://honcho.dev), providing AI memory and personalization tools to LLM clients like Claude Desktop.

## Quickstart: Use the Hosted Server

1. Get an API key at <https://app.honcho.dev>
2. Add Honcho to your Claude Desktop config:

```json
{
  "mcpServers": {
    "honcho": {
      "command": "bunx",
      "args": [
        "mcp-remote",
        "https://mcp.honcho.dev",
        "--header",
        "Authorization:${AUTH_HEADER}",
        "--header",
        "X-Honcho-User-Name:${USER_NAME}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>",
        "USER_NAME": "<your-name>"
      }
    }
  }
}
```

### Optional Headers

| Header | Default | Description |
| --- | --- | --- |
| `X-Honcho-Assistant-Name` | `"Assistant"` | Name for the assistant peer |
| `X-Honcho-Workspace-ID` | `"default"` | Workspace to operate in |
| `X-Honcho-Base-URL` | `https://api.honcho.dev` | Custom API base URL |

## Available Tools

### Bespoke Flow (Simple)

| Tool | Description |
| --- | --- |
| `start_conversation` | Start a new conversation, returns a session ID |
| `get_personalization_insights` | Ask Honcho about the user for personalized responses |
| `add_turn` | Record user + assistant messages |

### General Tools

**Workspace:** `search_workspace`, `get_workspace_metadata`, `set_workspace_metadata`

**Peers:** `create_peer`, `list_peers`, `chat`, `get_peer_card`, `set_peer_card`, `get_peer_context`, `get_representation`, `get_peer_metadata`, `set_peer_metadata`, `search_peer_messages`

**Sessions:** `create_session`, `list_sessions`, `delete_session`, `clone_session`, `add_peers_to_session`, `remove_peers_from_session`, `get_session_peers`, `add_messages_to_session`, `get_session_messages`, `search_session_messages`, `get_session_context`, `get_session_representation`, `get_session_metadata`, `set_session_metadata`

**Conclusions:** `list_conclusions`, `query_conclusions`, `create_conclusions`, `delete_conclusion`

**System:** `schedule_dream`, `get_queue_status`

## Architecture

```
src/
  index.ts              # Worker entry point — parse config, delegate to MCP handler
  server.ts             # createServer() — registers all tools on an McpServer
  config.ts             # HonchoConfig, parseConfig(), createClient()
  types.ts              # ToolContext, result helpers
  tools/
    bespoke.ts          # start_conversation, add_turn, get_personalization_insights
    workspace.ts        # search, metadata
    peers.ts            # CRUD, chat, card, context, representation, search
    sessions.ts         # CRUD, peers, messages, context, representation, clone
    conclusions.ts      # list, query, create, delete
    system.ts           # dream, queue status
```

Built on:

- **[agents](https://www.npmjs.com/package/agents)** — `createMcpHandler` for Cloudflare Workers
- **[@modelcontextprotocol/sdk](https://www.npmjs.com/package/@modelcontextprotocol/sdk)** — `McpServer` for tool registration
- **[@honcho-ai/sdk](https://www.npmjs.com/package/@honcho-ai/sdk)** v2 — Honcho TypeScript SDK

## Development

### Setup

```bash
bun install
```

### Local dev

```bash
bun dev
```

### Type-check

```bash
bun run tsc --noEmit
```

### Test locally

```bash
bunx mcp-remote http://localhost:8787 \
  --header "Authorization:Bearer <key>" \
  --header "X-Honcho-User-Name:test"
```

### Deploy

```bash
bun run deploy              # production
bun run deploy:staging      # staging
```

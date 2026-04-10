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
| `X-Honcho-Workspace-ID` | `"default"` | Workspace to operate in |
| `X-Honcho-Assistant-Name` | `"Assistant"` | Peer ID for the assistant |
| `X-Honcho-Base-URL` | `"https://api.honcho.dev"` | Honcho API base URL — override for self-hosted instances |

## Self-Hosted Instances

If you run your own Honcho server, pass `X-Honcho-Base-URL` to point the MCP Worker at it.

> **Note:** `http://localhost:8000` resolves from the MCP Worker's runtime (Cloudflare's edge), not from your local machine. To reach a locally running Honcho server, expose it with a tunnel (e.g. [ngrok](https://ngrok.com/), [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)) and use the tunnel URL as `X-Honcho-Base-URL`.

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
        "X-Honcho-User-Name:${USER_NAME}",
        "--header",
        "X-Honcho-Base-URL:${HONCHO_BASE_URL}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>",
        "USER_NAME": "<your-name>",
        "HONCHO_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

For Cursor or other clients with native HTTP MCP support:

```json
{
  "mcpServers": {
    "honcho": {
      "url": "https://mcp.honcho.dev",
      "headers": {
        "Authorization": "Bearer <your-honcho-key>",
        "X-Honcho-User-Name": "<your-name>",
        "X-Honcho-Base-URL": "http://localhost:8000"
      }
    }
  }
}
```

## Available Tools

**Workspace:** `inspect_workspace` (aggregates metadata, configuration, and peer/session IDs), `list_workspaces` (enumerates accessible workspaces), `search` (semantic search scoped by optional peer/session params), `get_metadata`, `set_metadata`

**Peers:** `create_peer`, `list_peers`, `chat`, `get_peer_card`, `set_peer_card`, `get_peer_context`, `get_representation`

**Sessions:** `create_session`, `list_sessions`, `delete_session`, `clone_session`, `add_peers_to_session`, `remove_peers_from_session`, `get_session_peers`, `inspect_session`, `add_messages_to_session`, `get_session_messages`, `get_session_message`, `get_session_context`

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
    workspace.ts        # inspect, list, search, metadata
    peers.ts            # CRUD, chat, card, context, representation
    sessions.ts         # CRUD, peers, messages, inspect, context, clone
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

# Honcho MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for [Honcho](https://honcho.dev). The hosted path is a Cloudflare Worker; the same tools also run over stdio and over Streamable HTTP (`bun src/http.ts`) for Docker and other long-lived process hosts.

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
        "Authorization:${AUTH_HEADER}"
      ],
      "env": {
        "AUTH_HEADER": "Bearer <your-honcho-key>"
      }
    }
  }
}
```

Every workspace-scoped tool takes a `workspace_id` argument. If you set `X-Honcho-Workspace-ID` on the connection, that value fills `workspace_id` when the argument is omitted. Use `list_workspaces` to discover IDs.

## Available Tools

**Workspace:** `list_workspaces` (id, metadata, created_at), `create_workspace` (get-or-create with optional metadata), `inspect_workspace` (aggregates metadata, configuration, and peer/session IDs), `search` (semantic search scoped by optional peer/session params), `get_metadata`, `set_metadata`

**Peers:** `create_peer`, `list_peers`, `chat`, `get_peer_card`, `set_peer_card`, `get_peer_context`, `get_representation`

**Sessions:** `create_session`, `list_sessions`, `delete_session`, `clone_session`, `add_peers_to_session`, `remove_peers_from_session`, `get_session_peers`, `inspect_session`, `add_messages_to_session`, `get_session_messages`, `get_session_message`, `get_session_context`

**Conclusions:** `list_conclusions`, `query_conclusions`, `create_conclusions`, `delete_conclusion`

**System:** `schedule_dream`, `get_queue_status`

## Architecture

```
src/
  index.ts              # Worker entry point — parse config, delegate to MCP handler
  stdio.ts              # Local stdio host (bun src/stdio.ts)
  http.ts               # Streamable HTTP host (bun src/http.ts / Docker)
  server.ts             # createServer() — registers all tools on an McpServer
  config.ts             # HonchoConfig, parseConfig(), createClientFactory()
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

## Self-Hosted Honcho

If you run Honcho yourself, point this server at it with `HONCHO_API_URL`.
When unset, requests go to `https://api.honcho.dev`.

**Cloudflare Worker (`bun run dev` / `bun run deploy`):** create `mcp/.dev.vars`:

```
HONCHO_API_URL=http://127.0.0.1:28000
```

For a deployed Worker: `wrangler secret put HONCHO_API_URL`.

## HTTP host

For Docker or any platform that runs a long-lived process, use the Streamable
HTTP entry instead of the Worker. Clients keep the same `mcp-remote` shape as
`https://mcp.honcho.dev`. Sessions live in process memory — run one instance.

```bash
cd mcp && bun install
HONCHO_API_URL=http://127.0.0.1:8000 bun run http
```

```bash
bunx mcp-remote http://127.0.0.1:3000 \
  --header "Authorization:Bearer <key>"
```

Auth is the `Authorization: Bearer` header (same as the Worker). Established
sessions still require that same bearer. Optional `X-Honcho-Workspace-ID`
fills `workspace_id` when the tool argument is omitted.

`HOST` defaults to `0.0.0.0`, `PORT` to `3000`. `GET /health` is unauthenticated.
MCP is served at `/` and `/mcp`. Idle sessions expire after
`MCP_SESSION_IDLE_MS` (default 30 minutes); `MCP_SESSION_MAX` (default 128)
caps concurrent sessions.

A platform start command is `bun src/http.ts` (or `bun run http` from `mcp/`).
This repo does not ship a `vercel.json`; serverless replicas do not share the
in-memory session map.

### Docker

```bash
docker build -f mcp/Dockerfile -t honcho-mcp mcp
docker run --rm -p 3000:3000 \
  -e HONCHO_API_URL=http://host.docker.internal:8000 \
  honcho-mcp
```

`docker-compose.yml.example` includes an `mcp` service beside `api` and
`deriver` (`HONCHO_API_URL=http://api:8000`, port `127.0.0.1:3000`).

## Local stdio

For a local Honcho instance, or any MCP client that spawns a process, run the
stdio host. `--cwd` loads `mcp/bunfig.toml` (Markdown loader) from this package.

```bash
cd mcp && bun install

claude mcp add honcho \
  -e HONCHO_API_KEY=hch-your-key-here \
  -e HONCHO_API_URL=http://127.0.0.1:28000 \
  -e HONCHO_WORKSPACE_ID=my-workspace \
  -- bun --cwd "$(pwd)" src/stdio.ts
```

`HONCHO_API_URL` defaults to `https://api.honcho.dev`. `HONCHO_WORKSPACE_ID` is
optional; without it, pass `workspace_id` on each tool call.

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

Worker (`bun dev`, port 8787) or HTTP host (`bun run http`, port 3000):

```bash
bunx mcp-remote http://localhost:8787 \
  --header "Authorization:Bearer <key>"
```

### Deploy

```bash
bun run deploy              # production
bun run deploy:staging      # staging
```

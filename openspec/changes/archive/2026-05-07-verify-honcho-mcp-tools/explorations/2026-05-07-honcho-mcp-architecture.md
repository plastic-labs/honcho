# Honcho MCP Architecture Exploration

**Date**: 2026-05-07
**Context**: Exploring how `honcho-mcp` is currently served and structured.

## High-Level Architecture

The `honcho-mcp` service is built as a **Cloudflare Worker** but is containerized via Docker for standalone/local deployment. It exposes the Honcho platform capabilities as Model Context Protocol (MCP) tools over an HTTP/SSE transport layer.

### Serving Stack

1. **Transport & Runtime Layer**: 
   - Uses **Cloudflare Workers** API (`fetch` handler).
   - In production/standalone, it is served via `wrangler dev` running on a Node.js Docker container (listening on `0.0.0.0:8787`).
   - Uses `bun` strictly as a fast package manager during the build stage, but reverts to Node.js for runtime execution to bypass `wrangler` incompatibilities with `bun`.

2. **HTTP/MCP Binding**:
   - The entrypoint (`src/index.ts`) intercepts standard HTTP requests.
   - Parses configuration and authentication from headers (e.g., `Authorization`, `X-Honcho-User-Name`, `X-Honcho-Workspace-ID`).
   - Wraps the official `@modelcontextprotocol/sdk` Server using a custom `createMcpHandler` (from the `agents/mcp` package), which likely implements the HTTP Server-Sent Events (SSE) transport required for MCP over HTTP.

3. **Domain Tools**:
   - `server.ts` registers discrete tool modules into the MCP server:
     - `workspace`
     - `peers`
     - `sessions`
     - `conclusions`
     - `system`

## Flow Diagram

```ascii
┌──────────────────────┐
│  MCP Client (IDE)    │
│  (Cursor, Windsurf)  │
└──────────┬───────────┘
           │ HTTP (SSE / POST)
           ▼
┌────────────────────────────────────────────────────────┐
│ Docker Container (Node.js 22)                          │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ wrangler dev (Port 8787)                         │  │
│  │                                                  │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │ src/index.ts (Fetch Handler)               │  │  │
│  │  │  - Parses CORS & Auth Headers              │  │  │
│  │  │  - Initializes Honcho Client               │  │  │
│  │  │                                            │  │  │
│  │  │  ┌──────────────────────────────────────┐  │  │  │
│  │  │  │ agents/mcp : createMcpHandler()      │  │  │  │
│  │  │  │ (Handles MCP-over-HTTP SSE transport)│  │  │  │
│  │  │  └──────────────────┬───────────────────┘  │  │  │
│  │  │                     ▼                      │  │  │
│  │  │  ┌──────────────────────────────────────┐  │  │  │
│  │  │  │ src/server.ts (McpServer instance)   │  │  │  │
│  │  │  │ ├─ tools/workspace.ts                │  │  │  │
│  │  │  │ ├─ tools/peers.ts                    │  │  │  │
│  │  │  │ ├─ tools/sessions.ts                 │  │  │  │
│  │  │  │ ├─ tools/conclusions.ts              │  │  │  │
│  │  │  │ └─ tools/system.ts                   │  │  │  │
│  │  │  └──────────────────┬───────────────────┘  │  │  │
│  │  └─────────────────────┼──────────────────────┘  │  │
│  └────────────────────────┼─────────────────────────┘  │
└───────────────────────────┼────────────────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ Honcho Backend API   │
                │ (HONCHO_API_URL)     │
                └──────────────────────┘
```

## Observations & Insights

- **Stateless Design**: By extracting config directly from incoming request headers per-fetch, the worker remains completely stateless.
- **Docker/Wrangler Quirks**: The Dockerfile explicitly mentions reverting to the "native NPM/Node context to bypass Wrangler's anti-Bun blocker". This is a crucial operational detail if we intend to modify the CI/CD or build process.
- **Protocol**: Since it uses `agents/mcp` to create the handler mapped to `route: "/"`, it means the server is accessible via standard HTTP (SSE for the MCP message stream and POST for tool execution requests).

---

## Supported MCP Tools

The `honcho-mcp` server exposes a rich set of capabilities organized into five domains. Each tool is called via the MCP protocol and uses the configurations (Workspace ID, Peer ID, etc.) passed in the HTTP headers from the connecting client.

### 1. Workspace Domain (`tools/workspace.ts`)
Manages and queries high-level workspace configuration and search.
- **`inspect_workspace`**: Aggregates workspace metadata, configuration, peer IDs, and session IDs to give a high-level overview.
- **`list_workspaces`**: Discovers available workspaces accessible by the current credentials.
- **`search`**: Semantic search across messages. Can be scoped globally, by peer, or by session.
- **`get_metadata` / `set_metadata`**: Reads or writes Key-Value metadata. Can be scoped to the workspace, a specific peer, or a specific session.

### 2. Peers Domain (`tools/peers.ts`)
Handles the agents or users (peers) participating in the workspace.
- **`create_peer`**: Registers a new participant (returns ID and config).
- **`list_peers`**: Lists all peers in the workspace.
- **`chat`**: Interacts with Honcho's reasoning system to ask natural-language questions about a peer (e.g., "What are this user's preferences?").
- **`get_peer_card` / `set_peer_card`**: Retrieves or updates a compact set of biographical facts about a peer.
- **`get_representation`**: Retrieves a formatted string of conclusions/facts Honcho has derived about a peer.
- **`get_peer_context`**: Combines both the representation (conclusions) and the peer card for a full context dump.

### 3. Sessions Domain (`tools/sessions.ts`)
Manages conversations (sessions) and message routing between peers.
- **`create_session` / `list_sessions` / `delete_session`**: Basic session lifecycle management.
- **`clone_session`**: Forks an existing session (optionally up to a specific message ID) to explore alternative conversation branches.
- **`add_peers_to_session` / `remove_peers_from_session` / `get_session_peers`**: Manages which peers are participating in a conversation.
- **`inspect_session`**: Returns an aggregated view of a session (participants, message count, summaries).
- **`add_messages_to_session`**: Records conversation turns. Crucially, each message must specify the `peer_id` of its author.
- **`get_session_messages` / `get_session_message`**: Retrieves conversation history, with support for pagination and metadata filtering.
- **`get_session_context`**: Provides an optimized context window for LLMs, including recent messages and summaries of older ones.

### 4. Conclusions Domain (`tools/conclusions.ts`)
Interacts with the knowledge graph of derived facts and observations.
- **`list_conclusions`**: Lists facts/observations that Honcho has derived about a peer.
- **`query_conclusions`**: Performs semantic search specifically across derived knowledge (more targeted than general message search).
- **`create_conclusions`**: Manually injects knowledge/facts into Honcho that weren't derived automatically.
- **`delete_conclusion`**: Removes incorrect or outdated knowledge facts.

### 5. System Domain (`tools/system.ts`)
Manages background processing and consolidation.
- **`schedule_dream`**: Triggers a background memory-consolidation task. A "dream" consolidates raw observations into higher-level insights and updates peer cards.
- **`get_queue_status`**: Checks the status of background processing queues (e.g., message derivation, dreaming) to see if Honcho is still processing data.

---

## Tool Verification Strategy

To accomplish the objective of verifying that all MCP tools function as designed, we must execute a structured End-to-End (E2E) testing cycle. The verification will cover these phases:

### Phase A: Setup & Lifecycle Validation
- **`inspect_workspace`** & **`list_workspaces`**: Check initial state.
- **`create_peer`**: Create 2 test peers (e.g., `user-test-1`, `agent-test-2`) and verify via **`list_peers`**.
- **`create_session`**: Establish a test session, then **`add_peers_to_session`**, and verify via **`get_session_peers`** and **`inspect_session`**.

### Phase B: Interaction & Data Insertion
- **`add_messages_to_session`**: Have the peers exchange multiple messages.
- **`get_session_messages`**: Verify messages are stored with correct pagination.
- **`set_metadata` / `get_metadata`**: Test writing and reading metadata on the workspace, peer, and session levels.

### Phase C: Context & Search Verification
- **`search`**: Search for specific keywords used in the test messages (verify global, session, and peer scoping).
- **`get_session_context`**: Validate the returned context payload includes the expected summary/messages format.

### Phase D: Knowledge Graph & Reasoning (Conclusions)
- **`create_conclusions`**: Manually inject facts about a peer.
- **`list_conclusions`** & **`query_conclusions`**: Verify retrieval of the injected facts.
- **`set_peer_card`** & **`get_peer_card`**: Establish and read the biographical peer card.
- **`get_peer_context`** & **`get_representation`**: Ensure the aggregation of conclusions and peer cards works accurately.
- **`schedule_dream`** & **`get_queue_status`**: Trigger the background job and confirm the queue reflects the processing state.
- **`chat`**: Run a natural language query against the peer to see if the knowledge is actively utilized by the Honcho reasoning system.

### Phase E: Teardown Validation
- **`remove_peers_from_session`**: Validate detachment logic.
- **`delete_conclusion`**: Remove the manual conclusion facts.
- **`delete_session`**: Delete the session and verify it no longer appears in **`list_sessions`**.

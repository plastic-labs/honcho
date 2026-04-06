import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Honcho } from "@honcho-ai/sdk";

const HONCHO_BASE_URL = process.env.HONCHO_BASE_URL || "http://honcho-api:8000";
const HONCHO_USER = process.env.HONCHO_USER;
if (!HONCHO_USER) {
  console.error("HONCHO_USER environment variable is required");
  process.exit(1);
}
const HONCHO_WORKSPACE = process.env.HONCHO_WORKSPACE || "default";

function createHonchoClient(): Honcho {
  return new Honcho({
    apiKey: "not-needed",
    baseURL: HONCHO_BASE_URL,
    workspaceId: HONCHO_WORKSPACE,
  });
}

function registerTools(server: McpServer, honcho: Honcho) {
  // inspect_workspace
  server.registerTool(
    "inspect_workspace",
    { description: "Inspect the current workspace at a glance." },
    async () => {
      const metadata = await honcho.getMetadata();
      return { content: [{ type: "text", text: JSON.stringify(metadata) }] };
    },
  );

  // list_workspaces
  server.registerTool(
    "list_workspaces",
    { description: "List workspaces accessible to the current credentials." },
    async () => {
      const ws = await honcho.workspaces();
      return { content: [{ type: "text", text: JSON.stringify(ws) }] };
    },
  );

  // search
  server.registerTool(
    "search",
    {
      description: "Semantic search across messages.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query." },
          peer_id: { type: "string", description: "Optional: scope to peer." },
          session_id: {
            type: "string",
            description: "Optional: scope to session.",
          },
        },
        required: ["query"],
      },
    },
    async ({ query, peer_id, session_id }: any) => {
      const result = await honcho.search(query, {
        filters: { peerId: peer_id, sessionId: session_id },
      });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // create_peer
  server.registerTool(
    "create_peer",
    {
      description: "Get or create a peer.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
        },
        required: ["peer_id"],
      },
    },
    async ({ peer_id }: any) => {
      const peer = await honcho.peer(peer_id);
      return {
        content: [{ type: "text", text: JSON.stringify({ peer_id: peer.id }) }],
      };
    },
  );

  // list_peers
  server.registerTool(
    "list_peers",
    { description: "List peers in the workspace." },
    async () => {
      const result = await honcho.peers();
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // create_session
  server.registerTool(
    "create_session",
    {
      description: "Get or create a session.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
        },
        required: ["session_id"],
      },
    },
    async ({ session_id }: any) => {
      const session = await honcho.session(session_id);
      return {
        content: [
          { type: "text", text: JSON.stringify({ session_id: session.id }) },
        ],
      };
    },
  );

  // list_sessions
  server.registerTool(
    "list_sessions",
    { description: "List sessions in the workspace." },
    async () => {
      const result = await honcho.sessions();
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // add_peers_to_session
  server.registerTool(
    "add_peers_to_session",
    {
      description: "Add peers to a session.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
          peers: { type: "array", items: { type: "string" } },
        },
        required: ["session_id", "peers"],
      },
    },
    async ({ session_id, peers }: any) => {
      const session = await honcho.session(session_id);
      await session.addPeers(peers);
      return {
        content: [{ type: "text", text: JSON.stringify({ success: true }) }],
      };
    },
  );

  // add_messages_to_session
  server.registerTool(
    "add_messages_to_session",
    {
      description: "Add messages to a session.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
          messages: {
            type: "array",
            items: {
              type: "object",
              properties: {
                peer_id: { type: "string" },
                content: { type: "string" },
              },
              required: ["peer_id", "content"],
            },
          },
        },
        required: ["session_id", "messages"],
      },
    },
    async ({ session_id, messages }: any) => {
      const session = await honcho.session(session_id);
      const result = await session.addMessages(
        messages.map((m: any) => ({ peerId: m.peer_id, content: m.content })),
      );
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // get_session_messages
  server.registerTool(
    "get_session_messages",
    {
      description: "Get messages from a session.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
        },
        required: ["session_id"],
      },
    },
    async ({ session_id }: any) => {
      const session = await honcho.session(session_id);
      const result = await session.getMessages();
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // chat
  server.registerTool(
    "chat",
    {
      description: "Ask about a peer's knowledge.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
          query: { type: "string" },
          target_peer_id: { type: "string" },
          session_id: { type: "string" },
          reasoning_level: {
            type: "string",
            enum: ["minimal", "low", "medium", "high", "max"],
          },
        },
        required: ["peer_id", "query"],
      },
    },
    async ({
      peer_id,
      query,
      target_peer_id,
      session_id,
      reasoning_level,
    }: any) => {
      const peer = await honcho.peer(peer_id);
      const result = await peer.chat(query, {
        targetPeerId: target_peer_id,
        sessionId: session_id,
        reasoningLevel: reasoning_level,
      });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // get_peer_card
  server.registerTool(
    "get_peer_card",
    {
      description: "Get peer card - biographical facts.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
          target_peer_id: { type: "string" },
        },
        required: ["peer_id"],
      },
    },
    async ({ peer_id, target_peer_id }: any) => {
      const peer = await honcho.peer(peer_id);
      const result = await peer.getCard(target_peer_id);
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // get_representation
  server.registerTool(
    "get_representation",
    {
      description: "Get formatted representation for a peer.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
          target_peer_id: { type: "string" },
          session_id: { type: "string" },
        },
        required: ["peer_id"],
      },
    },
    async ({ peer_id, target_peer_id, session_id }: any) => {
      const peer = await honcho.peer(peer_id);
      const result = await peer.getRepresentation({
        targetPeerId: target_peer_id,
        sessionId: session_id,
      });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // list_conclusions
  server.registerTool(
    "list_conclusions",
    {
      description: "List conclusions about a peer.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
          target_peer_id: { type: "string" },
        },
        required: ["peer_id"],
      },
    },
    async ({ peer_id, target_peer_id }: any) => {
      const peer = await honcho.peer(peer_id);
      const result = await peer.listConclusions(target_peer_id);
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // query_conclusions
  server.registerTool(
    "query_conclusions",
    {
      description: "Search conclusions.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
          query: { type: "string" },
          target_peer_id: { type: "string" },
          top_k: { type: "number" },
        },
        required: ["peer_id", "query"],
      },
    },
    async ({ peer_id, query, target_peer_id, top_k }: any) => {
      const peer = await honcho.peer(peer_id);
      const result = await peer.queryConclusions(query, {
        targetPeerId: target_peer_id,
        topK: top_k,
      });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // create_conclusions
  server.registerTool(
    "create_conclusions",
    {
      description: "Create conclusions about a peer.",
      inputSchema: {
        type: "object",
        properties: {
          peer_id: { type: "string" },
          target_peer_id: { type: "string" },
          conclusions: { type: "array", items: { type: "string" } },
          session_id: { type: "string" },
        },
        required: ["peer_id", "target_peer_id", "conclusions"],
      },
    },
    async ({ peer_id, target_peer_id, conclusions, session_id }: any) => {
      const peer = await honcho.peer(peer_id);
      const result = await peer.createConclusions(target_peer_id, conclusions, {
        sessionId: session_id,
      });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );

  // get_queue_status
  server.registerTool(
    "get_queue_status",
    { description: "Get background task queue status." },
    async () => {
      const result = await honcho.queueStatus();
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    },
  );
}

async function main() {
  const honcho = createHonchoClient();

  const server = new McpServer(
    { name: "honcho-mcp", version: "1.0.0" },
    { capabilities: { tools: {} } },
  );

  registerTools(server, honcho);

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);

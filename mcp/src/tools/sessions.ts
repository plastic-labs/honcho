import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Message } from "@honcho-ai/sdk";
import type { ToolContext } from "../types.js";
import {
  textResult,
  errorResult,
  formatMessage,
  formatMessages,
  formatSessionSummaries,
} from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── create_session ──────────────────────────────────────────────────
  server.registerTool(
    "create_session",
    {
      description: [
        "Get or create a session with the given ID.",
        "Use this when you need a raw session (for the bespoke flow, use start_conversation instead).",
        "Returns the session ID.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("Unique identifier for the session."),
      },
    },
    async ({ session_id }) => {
      try {
        const session = await ctx.honcho.session(session_id, { metadata: {} });
        return textResult({ session_id: session.id });
      } catch (e) {
        return errorResult(
          `Failed to create session: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── list_sessions ───────────────────────────────────────────────────
  server.registerTool(
    "list_sessions",
    {
      description: [
        "List sessions in the current workspace (paginated).",
        "Use this to discover existing conversations.",
        "Returns session IDs with pagination metadata.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const page = await ctx.honcho.sessions();
        return textResult({
          sessions: page.items.map((s) => ({ id: s.id })),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to list sessions: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── delete_session ──────────────────────────────────────────────────
  server.registerTool(
    "delete_session",
    {
      description: [
        "Delete a session and all its messages.",
        "This cannot be undone.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to delete."),
      },
    },
    async ({ session_id }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        await session.delete();
        return textResult("Session deleted successfully");
      } catch (e) {
        return errorResult(
          `Failed to delete session: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── clone_session ───────────────────────────────────────────────────
  server.registerTool(
    "clone_session",
    {
      description: [
        "Clone a session, optionally up to a specific message.",
        "Use this to fork a conversation — e.g. to explore a different branch.",
        "Returns the new cloned session ID.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to clone."),
        message_id: z
          .string()
          .optional()
          .describe(
            "Optional: clone only up to and including this message. Omit to clone everything.",
          ),
      },
    },
    async ({ session_id, message_id }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const cloned = await session.clone(message_id);
        return textResult({ session_id: cloned.id });
      } catch (e) {
        return errorResult(
          `Failed to clone session: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── add_peers_to_session ────────────────────────────────────────────
  server.registerTool(
    "add_peers_to_session",
    {
      description: [
        "Add one or more peers to a session.",
        "Use this to bring participants into a conversation.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to add peers to."),
        peers: z
          .array(
            z.union([
              z.string().describe("Peer ID with default config."),
              z.object({
                peer_id: z.string().describe("Peer ID."),
                observe_me: z
                  .boolean()
                  .nullable()
                  .optional()
                  .describe("Whether this peer's messages trigger derivation in this session."),
                observe_others: z
                  .boolean()
                  .nullable()
                  .optional()
                  .describe("Whether this peer observes other peers' messages in this session."),
              }).describe("Peer with per-session config."),
            ]),
          )
          .describe("Peers to add — plain IDs or objects with per-session config."),
      },
    },
    async ({ session_id, peers }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const additions = peers.map((p) => {
          if (typeof p === "string") return p;
          const config: { observeMe?: boolean | null; observeOthers?: boolean | null } = {};
          if (p.observe_me !== undefined) config.observeMe = p.observe_me;
          if (p.observe_others !== undefined) config.observeOthers = p.observe_others;
          return Object.keys(config).length > 0
            ? [p.peer_id, config] as [string, typeof config]
            : p.peer_id;
        });
        await session.addPeers(additions);
        return textResult("Peers added to session successfully");
      } catch (e) {
        return errorResult(
          `Failed to add peers: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── remove_peers_from_session ───────────────────────────────────────
  server.registerTool(
    "remove_peers_from_session",
    {
      description: [
        "Remove one or more peers from a session.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to remove peers from."),
        peer_ids: z
          .array(z.string())
          .describe("Peer IDs to remove."),
      },
    },
    async ({ session_id, peer_ids }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        await session.removePeers(peer_ids);
        return textResult("Peers removed from session successfully");
      } catch (e) {
        return errorResult(
          `Failed to remove peers: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_session_peers ───────────────────────────────────────────────
  server.registerTool(
    "get_session_peers",
    {
      description: [
        "Get all peers participating in a session.",
        "Use this to see who is in a conversation.",
        "Returns an array of peer IDs.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to query."),
      },
    },
    async ({ session_id }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const peers = await session.peers();
        return textResult(peers.map((p) => p.id));
      } catch (e) {
        return errorResult(
          `Failed to get session peers: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── inspect_session ─────────────────────────────────────────────────
  server.registerTool(
    "inspect_session",
    {
      description: [
        "Inspect a session at a glance.",
        "Aggregates peer IDs, message count, and available summaries.",
        "Returns a single JSON object.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to inspect."),
      },
    },
    async ({ session_id }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const [peers, messagePage, summaries] = await Promise.all([
          session.peers(),
          session.messages(),
          session.summaries(),
        ]);

        return textResult({
          session_id,
          peers: peers.map((peer) => ({ id: peer.id })),
          message_count: messagePage.total,
          summaries: formatSessionSummaries(summaries),
        });
      } catch (e) {
        return errorResult(
          `Failed to inspect session: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── add_messages_to_session ─────────────────────────────────────────
  server.registerTool(
    "add_messages_to_session",
    {
      description: [
        "Add messages to a session from specific peers.",
        "Use this to record conversation turns. Each message must specify the peer_id of the author.",
        "For the bespoke flow, use start_conversation first to get the user_peer_id and assistant_peer_id.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to add messages to."),
        messages: z
          .array(
            z.object({
              peer_id: z.string().describe("Peer ID authoring this message."),
              content: z.string().describe("Message text."),
              metadata: z
                .record(z.string(), z.unknown())
                .optional()
                .describe("Optional metadata."),
            }),
          )
          .describe("Messages to add."),
      },
    },
    async ({ session_id, messages }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const peerCache = new Map<string, Awaited<ReturnType<typeof ctx.honcho.peer>>>();
        const sessionMessages = [];
        for (const msg of messages) {
          let peer = peerCache.get(msg.peer_id);
          if (!peer) {
            peer = await ctx.honcho.peer(msg.peer_id);
            peerCache.set(msg.peer_id, peer);
          }
          sessionMessages.push(
            msg.metadata
              ? peer.message(msg.content, { metadata: msg.metadata })
              : peer.message(msg.content),
          );
        }
        await session.addMessages(sessionMessages);
        return textResult("Messages added to session successfully");
      } catch (e) {
        return errorResult(
          `Failed to add messages: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_session_messages ────────────────────────────────────────────
  server.registerTool(
    "get_session_messages",
    {
      description: [
        "Get messages from a session (paginated), with optional metadata filtering.",
        "Use this to read the conversation history.",
        "Returns the first page of messages with pagination metadata.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to get messages from."),
        filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe("Optional metadata filter criteria."),
      },
    },
    async ({ session_id, filters }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const page = await session.messages(filters);
        return textResult({
          messages: formatMessages(page.items),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to get messages: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_session_message ─────────────────────────────────────────────
  server.registerTool(
    "get_session_message",
    {
      description: [
        "Get a single message from a session by ID.",
        "Use this when you already know the message ID and need the exact record.",
        "Returns the message object.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session the message belongs to."),
        message_id: z.string().describe("The message ID to fetch."),
      },
    },
    async ({ session_id, message_id }) => {
      try {
        // Workaround: the current @honcho-ai/sdk Session API does not expose
        // a single-message getter, so we fetch the message by ID via raw HTTP.
        const messageData = await ctx.honcho.http.get<{
          id: string;
          content: string;
          peer_id: string;
          session_id: string;
          workspace_id: string;
          metadata: Record<string, unknown>;
          created_at: string;
          token_count: number;
        }>(
          `/v3/workspaces/${ctx.honcho.workspaceId}/sessions/${session_id}/messages/${message_id}`,
        );
        const message = Message.fromApiResponse(messageData);
        return textResult(formatMessage(message));
      } catch (e) {
        return errorResult(
          `Failed to get message: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_session_context ─────────────────────────────────────────────
  server.registerTool(
    "get_session_context",
    {
      description: [
        "Get optimized context for a session, suitable for LLM prompts.",
        "Includes recent messages and an optional summary of older ones.",
        "Use this to build a context window for the next LLM call.",
        "Returns messages, summary, and session ID.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to get context for."),
        summary: z
          .boolean()
          .optional()
          .describe("Include a summary of older messages? Default: true."),
        tokens: z
          .number()
          .optional()
          .describe("Target token budget for the context window."),
      },
    },
    async ({ session_id, summary, tokens }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const context = await session.context({ summary, tokens });
        return textResult({
          session_id: context.sessionId,
          summary: context.summary,
          messages: formatMessages(context.messages),
        });
      } catch (e) {
        return errorResult(
          `Failed to get context: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

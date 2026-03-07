import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, formatMessages } from "../types.js";

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
        "List all sessions in the current workspace.",
        "Use this to discover existing conversations.",
        "Returns an array of session IDs.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const page = await ctx.honcho.sessions();
        const sessions: { id: string }[] = [];
        for await (const session of page) {
          sessions.push({ id: session.id });
        }
        return textResult(sessions);
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
        "Permanently delete a session and all its messages.",
        "Use this to clean up conversations that are no longer needed. This cannot be undone.",
        "Returns a confirmation message.",
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
        "Returns a confirmation message.",
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
        "Use this to remove participants from a conversation.",
        "Returns a confirmation message.",
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

  // ── add_messages_to_session ─────────────────────────────────────────
  server.registerTool(
    "add_messages_to_session",
    {
      description: [
        "Add messages to a session from specific peers.",
        "Use this for multi-peer conversations where you need to attribute messages to specific peers.",
        "For the simple user/assistant flow, use add_turn instead.",
        "Returns a confirmation message.",
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
        "Get all messages from a session, with optional metadata filtering.",
        "Use this to read the conversation history.",
        "Returns a paginated array of messages.",
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
        const messages = [];
        for await (const msg of page) {
          messages.push(msg);
        }
        return textResult(formatMessages(messages));
      } catch (e) {
        return errorResult(
          `Failed to get messages: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── search_session_messages ─────────────────────────────────────────
  server.registerTool(
    "search_session_messages",
    {
      description: [
        "Semantic search across messages in a specific session.",
        "Use this to find relevant messages within a single conversation.",
        "Returns an array of matching messages.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to search in."),
        query: z.string().describe("Search query."),
      },
    },
    async ({ session_id, query }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const messages = await session.search(query);
        return textResult(formatMessages(messages));
      } catch (e) {
        return errorResult(
          `Search failed: ${e instanceof Error ? e.message : String(e)}`,
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
          messages: context.messages.map((msg) => ({
            id: msg.id,
            content: msg.content,
            peer_id: msg.peerId,
            metadata: msg.metadata,
            created_at: msg.createdAt,
          })),
        });
      } catch (e) {
        return errorResult(
          `Failed to get context: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_session_representation ──────────────────────────────────────
  server.registerTool(
    "get_session_representation",
    {
      description: [
        "Get a peer's representation scoped to a specific session.",
        "Use this to see what Honcho has learned about a peer from a single conversation.",
        "Returns a formatted string of session-scoped conclusions.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to scope to."),
        peer_id: z
          .string()
          .describe("The peer to get the representation for."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: get what peer_id knows about target_peer_id in this session.",
          ),
      },
    },
    async ({ session_id, peer_id, target_peer_id }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const rep = await session.representation(peer_id, {
          target: target_peer_id,
        });
        return textResult(rep);
      } catch (e) {
        return errorResult(
          `Failed to get representation: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_session_metadata ────────────────────────────────────────────
  server.registerTool(
    "get_session_metadata",
    {
      description: [
        "Get the metadata dictionary for a session.",
        "Use this to read custom attributes stored on a session.",
        "Returns a JSON object.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to get metadata for."),
      },
    },
    async ({ session_id }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        const metadata = await session.getMetadata();
        return textResult(metadata);
      } catch (e) {
        return errorResult(
          `Failed to get session metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── set_session_metadata ────────────────────────────────────────────
  server.registerTool(
    "set_session_metadata",
    {
      description: [
        "Set metadata for a session (overwrites existing metadata).",
        "Use this to store custom attributes on a session.",
        "Returns a confirmation message.",
      ].join("\n"),
      inputSchema: {
        session_id: z.string().describe("The session to set metadata for."),
        metadata: z
          .record(z.string(), z.unknown())
          .describe("Key-value pairs to set."),
      },
    },
    async ({ session_id, metadata }) => {
      try {
        const session = await ctx.honcho.session(session_id);
        await session.setMetadata(metadata);
        return textResult("Session metadata set successfully");
      } catch (e) {
        return errorResult(
          `Failed to set session metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

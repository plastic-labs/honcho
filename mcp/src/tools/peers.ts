import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, formatMessages } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── create_peer ─────────────────────────────────────────────────────
  server.registerTool(
    "create_peer",
    {
      description: [
        "Get or create a peer with the given ID.",
        "Use this to register a new participant (user or agent) in the workspace.",
        "Returns the peer ID and any configuration that was set.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("Unique identifier for the peer."),
        configuration: z
          .object({
            observeMe: z.boolean().nullable().optional().describe(
              "Whether derivation tasks should be created for this peer's messages. Default: true.",
            ),
          })
          .optional()
          .describe("Optional peer configuration."),
      },
    },
    async ({ peer_id, configuration }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id, { configuration });
        return textResult({ peer_id: peer.id, configuration: peer.configuration });
      } catch (e) {
        return errorResult(
          `Failed to create peer: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── list_peers ──────────────────────────────────────────────────────
  server.registerTool(
    "list_peers",
    {
      description: [
        "List all peers in the current workspace.",
        "Use this to discover which users and agents exist.",
        "Returns an array of peer IDs.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const page = await ctx.honcho.peers();
        const peers: { id: string }[] = [];
        for await (const peer of page) {
          peers.push({ id: peer.id });
        }
        return textResult(peers);
      } catch (e) {
        return errorResult(
          `Failed to list peers: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── chat ────────────────────────────────────────────────────────────
  server.registerTool(
    "chat",
    {
      description: [
        "Ask a natural-language question about a peer's knowledge and get an answer from Honcho's reasoning system.",
        "Use this to query what Honcho knows about any peer — their preferences, history, personality, etc.",
        "Returns a natural-language answer, or 'None' if no relevant information exists.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The peer to query about."),
        query: z.string().describe("Natural-language question."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: query what peer_id knows about this target peer instead of their global representation.",
          ),
        session_id: z
          .string()
          .optional()
          .describe("Optional: scope the query to a specific session."),
        reasoning_level: z
          .enum(["minimal", "low", "medium", "high", "max"])
          .optional()
          .describe("Reasoning effort. Higher = more detailed but slower."),
      },
    },
    async ({ peer_id, query, target_peer_id, session_id, reasoning_level }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const result = await peer.chat(query, {
          target: target_peer_id,
          session: session_id,
          reasoningLevel: reasoning_level,
        });
        return textResult(result ?? "None");
      } catch (e) {
        return errorResult(
          `Chat failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_peer_card ───────────────────────────────────────────────────
  server.registerTool(
    "get_peer_card",
    {
      description: [
        "Get the peer card — a compact set of biographical facts about a peer.",
        "Use this when you need a quick summary of who someone is.",
        "Returns an array of fact strings, or null if no card exists yet.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: get this peer's card about the target instead of their own.",
          ),
      },
    },
    async ({ peer_id, target_peer_id }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const card = await peer.getCard(target_peer_id);
        return textResult(card ?? "No peer card found.");
      } catch (e) {
        return errorResult(
          `Failed to get peer card: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── set_peer_card ───────────────────────────────────────────────────
  server.registerTool(
    "set_peer_card",
    {
      description: [
        "Set or update the peer card — a list of biographical facts about a peer.",
        "Use this to manually establish or correct facts about a peer.",
        "Returns the updated peer card.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The observer peer."),
        peer_card: z
          .array(z.string())
          .describe("Array of fact strings to set as the peer card."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: set this peer's card about the target instead of their own.",
          ),
      },
    },
    async ({ peer_id, peer_card, target_peer_id }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const result = await peer.setCard(peer_card, target_peer_id);
        return textResult(result ?? "Peer card set successfully");
      } catch (e) {
        return errorResult(
          `Failed to set peer card: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_peer_context ────────────────────────────────────────────────
  server.registerTool(
    "get_peer_context",
    {
      description: [
        "Get comprehensive context for a peer — combines their representation (conclusions) and peer card.",
        "Use this when you need the full picture of what Honcho knows about someone.",
        "Returns an object with representation, peer_card, peer_id, and target_id.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .optional()
          .describe("Optional: get context about this target peer."),
        search_query: z
          .string()
          .optional()
          .describe("Optional: semantic search to filter relevant conclusions."),
        max_conclusions: z
          .number()
          .optional()
          .describe("Optional: max number of conclusions to include."),
      },
    },
    async ({ peer_id, target_peer_id, search_query, max_conclusions }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const context = await peer.context({
          target: target_peer_id,
          searchQuery: search_query,
          maxConclusions: max_conclusions,
        });
        return textResult({
          peer_id: context.peerId,
          target_id: context.targetId,
          representation: context.representation,
          peer_card: context.peerCard,
        });
      } catch (e) {
        return errorResult(
          `Failed to get peer context: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_representation ──────────────────────────────────────────────
  server.registerTool(
    "get_representation",
    {
      description: [
        "Get the formatted representation for a peer — a text summary built from their conclusions.",
        "Use this when you want the textual representation without the peer card.",
        "Returns a formatted string of conclusions.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .optional()
          .describe("Optional: get representation about this target peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: scope to a specific session."),
        search_query: z
          .string()
          .optional()
          .describe("Optional: semantic search to filter conclusions."),
        max_conclusions: z
          .number()
          .optional()
          .describe("Optional: max number of conclusions."),
      },
    },
    async ({
      peer_id,
      target_peer_id,
      session_id,
      search_query,
      max_conclusions,
    }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const rep = await peer.representation({
          target: target_peer_id,
          session: session_id,
          searchQuery: search_query,
          maxConclusions: max_conclusions,
        });
        return textResult(rep);
      } catch (e) {
        return errorResult(
          `Failed to get representation: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_peer_metadata ───────────────────────────────────────────────
  server.registerTool(
    "get_peer_metadata",
    {
      description: [
        "Get the metadata dictionary for a peer.",
        "Use this to read custom attributes stored on a peer.",
        "Returns a JSON object of key-value pairs.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The peer to get metadata for."),
      },
    },
    async ({ peer_id }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const metadata = await peer.getMetadata();
        return textResult(metadata);
      } catch (e) {
        return errorResult(
          `Failed to get peer metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── set_peer_metadata ───────────────────────────────────────────────
  server.registerTool(
    "set_peer_metadata",
    {
      description: [
        "Set metadata for a peer (overwrites existing metadata).",
        "Use this to store custom attributes on a peer.",
        "Returns a confirmation message.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The peer to set metadata for."),
        metadata: z
          .record(z.string(), z.unknown())
          .describe("Key-value pairs to set."),
      },
    },
    async ({ peer_id, metadata }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        await peer.setMetadata(metadata);
        return textResult("Peer metadata set successfully");
      } catch (e) {
        return errorResult(
          `Failed to set peer metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── search_peer_messages ────────────────────────────────────────────
  server.registerTool(
    "search_peer_messages",
    {
      description: [
        "Semantic search across all messages authored by a specific peer.",
        "Use this to find what a particular peer has said across all sessions.",
        "Returns an array of matching messages.",
      ].join("\n"),
      inputSchema: {
        peer_id: z.string().describe("The peer whose messages to search."),
        query: z.string().describe("Search query."),
      },
    },
    async ({ peer_id, query }) => {
      try {
        const peer = await ctx.honcho.peer(peer_id);
        const messages = await peer.search(query);
        return textResult(formatMessages(messages));
      } catch (e) {
        return errorResult(
          `Search failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

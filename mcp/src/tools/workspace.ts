import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, formatMessages } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── inspect_workspace ───────────────────────────────────────────────
  server.registerTool(
    "inspect_workspace",
    {
      description: [
        "Inspect the current workspace at a glance.",
        "Aggregates workspace metadata, configuration, peer IDs, and session IDs.",
        "Returns the first page of peers/sessions with total counts.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const [metadata, configuration, peerPage, sessionPage] = await Promise.all([
          ctx.honcho.getMetadata(),
          ctx.honcho.getConfiguration(),
          ctx.honcho.peers(),
          ctx.honcho.sessions(),
        ]);

        return textResult({
          workspace_id: ctx.honcho.workspaceId,
          metadata,
          configuration,
          peer_count: peerPage.total,
          peers: peerPage.items.map((p) => ({ id: p.id })),
          session_count: sessionPage.total,
          sessions: sessionPage.items.map((s) => ({ id: s.id })),
        });
      } catch (e) {
        return errorResult(
          `Failed to inspect workspace: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── list_workspaces ─────────────────────────────────────────────────
  server.registerTool(
    "list_workspaces",
    {
      description: [
        "List workspaces accessible to the current credentials (paginated).",
        "Use this to discover available workspaces before selecting or switching context.",
        "Returns workspace IDs with pagination metadata.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const page = await ctx.honcho.workspaces();
        return textResult({
          workspaces: page.items.map((id) => ({ id })),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to list workspaces: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── search ────────────────────────────────────────────────────────
  server.registerTool(
    "search",
    {
      description: [
        "Semantic search across messages and, when peer_id is given, that peer's saved conclusions.",
        "Message scope is determined by which optional params are provided:",
        "- No scope params: search all messages in the workspace.",
        "- peer_id only: search messages authored by that peer across all sessions.",
        "- session_id only: search messages within that session.",
        "Conclusions require peer_id (self-conclusions are searched; conclusion IDs are usable with delete_conclusion).",
        "Returns {messages, conclusions}.",
      ].join("\n"),
      inputSchema: {
        query: z.string().describe("Search query."),
        peer_id: z
          .string()
          .optional()
          .describe("Optional: scope search to messages by this peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: scope search to messages in this session."),
        message_limit: z
          .number()
          .optional()
          .describe("Optional: max message results (1-100, default 10)."),
        message_filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Optional: filters for the message search, e.g. {"created_at": {"gte": "2026-01-01"}}. See https://honcho.dev/docs/v3/documentation/features/advanced/using-filters',
          ),
        conclusion_top_k: z
          .number()
          .optional()
          .describe("Optional: max conclusion results (default 10)."),
        conclusion_filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Optional: filters for the conclusion search, e.g. {"level": ["deductive", "inductive"]} to only return conclusions derived during dreaming. Levels: explicit (extracted directly from messages), deductive, inductive, contradiction. The session_id param does not scope conclusions; use {"session_id": ...} here for that.',
          ),
      },
    },
    async ({
      query,
      peer_id,
      session_id,
      message_limit,
      message_filters,
      conclusion_top_k,
      conclusion_filters,
    }) => {
      try {
        const peer = peer_id ? await ctx.honcho.peer(peer_id) : null;
        const messageOptions = {
          filters: message_filters,
          limit: message_limit,
        };

        const searchMessages = async () => {
          if (session_id) {
            const session = await ctx.honcho.session(session_id);
            return session.search(query, messageOptions);
          }
          if (peer) {
            return peer.search(query, messageOptions);
          }
          return ctx.honcho.search(query, messageOptions);
        };

        // Conclusion search needs an (observer, observed) pair, so it only
        // runs when peer_id is given.
        const searchConclusions = async () => {
          if (!peer) {
            return [];
          }
          try {
            return await peer.conclusions.query(
              query,
              conclusion_top_k,
              undefined,
              conclusion_filters,
            );
          } catch (e) {
            if (conclusion_filters) throw e;
            return [];
          }
        };

        const [messages, conclusions] = await Promise.all([
          searchMessages(),
          searchConclusions(),
        ]);
        return textResult({
          messages: formatMessages(messages),
          conclusions: conclusions.map((c) => ({
            id: c.id,
            content: c.content,
            level: c.level,
            created_at: c.createdAt,
          })),
        });
      } catch (e) {
        return errorResult(
          `Search failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_metadata ──────────────────────────────────────────────────
  server.registerTool(
    "get_metadata",
    {
      description: [
        "Get metadata for a resource. Scope is determined by which optional params are provided:",
        "- No scope params: get workspace metadata.",
        "- peer_id only: get peer metadata.",
        "- session_id only: get session metadata.",
      ].join("\n"),
      inputSchema: {
        peer_id: z
          .string()
          .optional()
          .describe("Optional: get metadata for this peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: get metadata for this session."),
      },
    },
    async ({ peer_id, session_id }) => {
      try {
        let metadata;
        if (session_id) {
          const session = await ctx.honcho.session(session_id);
          metadata = await session.getMetadata();
        } else if (peer_id) {
          const peer = await ctx.honcho.peer(peer_id);
          metadata = await peer.getMetadata();
        } else {
          metadata = await ctx.honcho.getMetadata();
        }
        return textResult(metadata);
      } catch (e) {
        return errorResult(
          `Failed to get metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── set_metadata ──────────────────────────────────────────────────
  server.registerTool(
    "set_metadata",
    {
      description: [
        "Set metadata for a resource. Overwrites existing metadata.",
        "Scope is determined by which optional params are provided:",
        "- No scope params: set workspace metadata.",
        "- peer_id only: set peer metadata.",
        "- session_id only: set session metadata.",
      ].join("\n"),
      inputSchema: {
        metadata: z
          .record(z.string(), z.unknown())
          .describe("Key-value pairs to set as metadata."),
        peer_id: z
          .string()
          .optional()
          .describe("Optional: set metadata for this peer."),
        session_id: z
          .string()
          .optional()
          .describe("Optional: set metadata for this session."),
      },
    },
    async ({ metadata, peer_id, session_id }) => {
      try {
        if (session_id) {
          const session = await ctx.honcho.session(session_id);
          await session.setMetadata(metadata);
          return textResult("Session metadata set successfully");
        } else if (peer_id) {
          const peer = await ctx.honcho.peer(peer_id);
          await peer.setMetadata(metadata);
          return textResult("Peer metadata set successfully");
        } else {
          await ctx.honcho.setMetadata(metadata);
          return textResult("Workspace metadata set successfully");
        }
      } catch (e) {
        return errorResult(
          `Failed to set metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

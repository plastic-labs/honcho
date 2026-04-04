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
        "Semantic search across messages. Scope is determined by which optional params are provided:",
        "- No scope params: search all messages in the workspace.",
        "- peer_id only: search messages authored by that peer across all sessions.",
        "- session_id only: search messages within that session.",
        "Returns an array of matching messages with their content, peer, and session info.",
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
      },
    },
    async ({ query, peer_id, session_id }) => {
      try {
        let messages;
        if (session_id) {
          const session = await ctx.honcho.session(session_id);
          messages = await session.search(query);
        } else if (peer_id) {
          const peer = await ctx.honcho.peer(peer_id);
          messages = await peer.search(query);
        } else {
          messages = await ctx.honcho.search(query);
        }
        return textResult(formatMessages(messages));
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

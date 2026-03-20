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

  // ── search_workspace ────────────────────────────────────────────────
  server.registerTool(
    "search_workspace",
    {
      description: [
        "Semantic search across all messages in the workspace.",
        "Use this to find past conversations or messages from any peer/session.",
        "Returns an array of matching messages with their content, peer, and session info.",
      ].join("\n"),
      inputSchema: {
        query: z.string().describe("Search query."),
      },
    },
    async ({ query }) => {
      try {
        const messages = await ctx.honcho.search(query);
        return textResult(formatMessages(messages));
      } catch (e) {
        return errorResult(
          `Search failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_workspace_metadata ──────────────────────────────────────────
  server.registerTool(
    "get_workspace_metadata",
    {
      description: [
        "Get metadata for the current workspace.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const metadata = await ctx.honcho.getMetadata();
        return textResult(metadata);
      } catch (e) {
        return errorResult(
          `Failed to get metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── set_workspace_metadata ──────────────────────────────────────────
  server.registerTool(
    "set_workspace_metadata",
    {
      description: [
        "Set metadata for the current workspace.",
        "Overwrites existing metadata.",
      ].join("\n"),
      inputSchema: {
        metadata: z
          .record(z.string(), z.unknown())
          .describe("Key-value pairs to set as workspace metadata."),
      },
    },
    async ({ metadata }) => {
      try {
        await ctx.honcho.setMetadata(metadata);
        return textResult("Workspace metadata set successfully");
      } catch (e) {
        return errorResult(
          `Failed to set metadata: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, formatMessages } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
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
        "Get the metadata dictionary for the current workspace.",
        "Use this to read workspace-level settings or custom attributes.",
        "Returns a JSON object of key-value pairs.",
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
        "Set metadata for the current workspace (overwrites existing metadata).",
        "Use this to store workspace-level settings or custom attributes.",
        "Returns a confirmation message.",
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

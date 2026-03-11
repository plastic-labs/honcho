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
        "Returns a single JSON object.",
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

        const peers: { id: string }[] = [];
        for await (const peer of peerPage) {
          peers.push({ id: peer.id });
        }

        const sessions: { id: string }[] = [];
        for await (const session of sessionPage) {
          sessions.push({ id: session.id });
        }

        return textResult({
          workspace_id: ctx.honcho.workspaceId,
          metadata,
          configuration,
          peers,
          sessions,
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
        "List all workspaces accessible to the current credentials.",
        "Use this to discover available workspaces before selecting or switching context.",
        "Returns an array of workspace IDs.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const page = await ctx.honcho.workspaces();
        const workspaces: { id: string }[] = [];
        for await (const workspace of page) {
          workspaces.push({ id: workspace });
        }
        return textResult(workspaces);
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

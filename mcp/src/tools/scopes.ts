import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, workspaceIdSchema } from "../types.js";

const pageSchema = z.number().int().min(1).optional().describe("Page number (1-indexed).");
const sizeSchema = z
  .number()
  .int()
  .min(1)
  .max(100)
  .optional()
  .describe("Results per page (max 100).");

export function register(server: McpServer, ctx: ToolContext) {
  // ── list_scopes ─────────────────────────────────────────────────────
  server.registerTool(
    "list_scopes",
    {
      description: [
        "List the scopes in a workspace (paginated).",
        "A scope is a named set of sessions that acts as a recall boundary: chat with scope=<name> answers only from that scope's sessions.",
        "Returns each scope's id, metadata, and created_at.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        page: pageSchema,
        size: sizeSchema,
      },
    },
    async ({ workspace_id, page: pageNum, size }) => {
      try {
        const page = await ctx.clientFor(workspace_id).scopes({ page: pageNum, size });
        return textResult({
          scopes: page.items.map((s) => ({
            id: s.id,
            metadata: s.metadata ?? {},
            created_at: s.createdAt,
          })),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to list scopes: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_scope_sessions ──────────────────────────────────────────────
  server.registerTool(
    "get_scope_sessions",
    {
      description: [
        "List the sessions that belong to a scope (paginated).",
        "Use this to see which conversations a recall boundary covers.",
        "Returns session IDs with pagination metadata.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        scope_id: z.string().describe("The scope to list sessions for."),
        page: pageSchema,
        size: sizeSchema,
      },
    },
    async ({ workspace_id, scope_id, page: pageNum, size }) => {
      try {
        const scope = await ctx.clientFor(workspace_id).scope(scope_id);
        const page = await scope.sessions({ page: pageNum, size });
        return textResult({
          scope_id,
          sessions: page.items.map((s) => ({ id: s.id })),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to list scope sessions: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

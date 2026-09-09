import { z } from "zod";
import { Scope } from "@honcho-ai/sdk";
import type { Honcho } from "@honcho-ai/sdk";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, workspaceIdSchema } from "../types.js";

/**
 * Reference an existing scope without the get-or-create round trip that
 * `honcho.scope()` performs, so read/remove tools never create a scope as a
 * side effect. The server returns 404 if the scope does not exist.
 */
function existingScope(honcho: Honcho, scopeId: string): Scope {
  return new Scope(scopeId, honcho.workspaceId, honcho.http);
}

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

  // ── create_scope ────────────────────────────────────────────────────
  server.registerTool(
    "create_scope",
    {
      description: [
        "Get or create a scope with the given ID.",
        "A scope is a named set of sessions that acts as a recall boundary. Optional metadata (e.g. a label, description, example queries) is stored with it.",
        "Sessions are added separately with add_sessions_to_scope. Returns the scope's id, metadata, and created_at.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        scope_id: z
          .string()
          .describe("Scope name, unique within the workspace (e.g. 'therapy', 'honcho-core')."),
        metadata: z
          .record(z.string(), z.unknown())
          .optional()
          .describe("Optional metadata to store with the scope."),
      },
    },
    async ({ workspace_id, scope_id, metadata }) => {
      try {
        const scope = await ctx
          .clientFor(workspace_id)
          .scope(scope_id, metadata ? { metadata } : undefined);
        return textResult({
          id: scope.id,
          metadata: scope.metadata ?? {},
          created_at: scope.createdAt,
        });
      } catch (e) {
        return errorResult(
          `Failed to create scope: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── add_sessions_to_scope ───────────────────────────────────────────
  server.registerTool(
    "add_sessions_to_scope",
    {
      description: [
        "Add sessions to a scope. Every session must already exist; re-adding a member is a no-op.",
        "Sessions that already hold messages are backfilled into the scope asynchronously — poll get_scope_status before relying on scoped recall for them.",
        "At most 100 sessions per call.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        scope_id: z.string().describe("The scope to add sessions to."),
        session_ids: z
          .array(z.string())
          .min(1)
          .max(100)
          .describe("Session IDs to add (max 100)."),
      },
    },
    async ({ workspace_id, scope_id, session_ids }) => {
      try {
        const scope = await ctx.clientFor(workspace_id).scope(scope_id);
        await scope.addSessions(session_ids);
        return textResult({ scope_id, added: session_ids.length });
      } catch (e) {
        return errorResult(
          `Failed to add sessions to scope: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── remove_session_from_scope ───────────────────────────────────────
  server.registerTool(
    "remove_session_from_scope",
    {
      description: [
        "Remove a session from a scope.",
        "Conclusions derived while it was a member are reconciled out asynchronously; the session itself is untouched.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        scope_id: z.string().describe("The scope to remove the session from."),
        session_id: z.string().describe("The session to remove."),
      },
    },
    async ({ workspace_id, scope_id, session_id }) => {
      try {
        const scope = existingScope(ctx.clientFor(workspace_id), scope_id);
        await scope.removeSession(session_id);
        return textResult({ scope_id, removed: session_id });
      } catch (e) {
        return errorResult(
          `Failed to remove session from scope: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_scope_status ────────────────────────────────────────────────
  server.registerTool(
    "get_scope_status",
    {
      description: [
        "Get backfill progress for a scope, keyed by session ID.",
        "Each entry is pending, completed, or failed. Only sessions with a backfill enqueued appear, so an empty result means nothing is outstanding.",
        "Use this after add_sessions_to_scope to tell 'the scope hasn't caught up yet' apart from 'there is nothing to recall'.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        scope_id: z.string().describe("The scope to check."),
      },
    },
    async ({ workspace_id, scope_id }) => {
      try {
        const scope = existingScope(ctx.clientFor(workspace_id), scope_id);
        const status = await scope.status();
        const entries = Object.entries(status.backfillStatus);
        const counts = { pending: 0, completed: 0, failed: 0 };
        const sessions: Record<string, { state: string; updated_at: string; docs_copied?: number }> = {};
        for (const [sid, st] of entries) {
          counts[st.state] += 1;
          sessions[sid] = { state: st.state, updated_at: st.updatedAt };
          if (st.docsCopied !== undefined) sessions[sid].docs_copied = st.docsCopied;
        }
        return textResult({ scope_id, ...counts, sessions });
      } catch (e) {
        return errorResult(
          `Failed to get scope status: ${e instanceof Error ? e.message : String(e)}`,
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
        const scope = existingScope(ctx.clientFor(workspace_id), scope_id);
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

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult, workspaceIdSchema } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── list_conclusions ────────────────────────────────────────────────
  server.registerTool(
    "list_conclusions",
    {
      description: [
        "List conclusions (facts and observations) that Honcho has derived about a peer (paginated).",
        "Use this to see what Honcho has learned. If no target is given, returns self-conclusions.",
        "Returns conclusion objects with pagination metadata.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: list conclusions about this target. Omit for self-conclusions.",
          ),
      },
    },
    async ({ workspace_id, peer_id, target_peer_id }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = target_peer_id
          ? peer.conclusionsOf(target_peer_id)
          : peer.conclusions;
        const page = await scope.list();
        return textResult({
          conclusions: page.items.map((c) => ({
            id: c.id,
            content: c.content,
            observer_id: c.observerId,
            observed_id: c.observedId,
            session_id: c.sessionId,
            created_at: c.createdAt,
          })),
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return errorResult(
          `Failed to list conclusions: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── query_conclusions ───────────────────────────────────────────────
  server.registerTool(
    "query_conclusions",
    {
      description: [
        "Semantic search across a peer's conclusions.",
        "Use this to find specific knowledge Honcho has derived — more targeted than list_conclusions.",
        "Returns an array of matching conclusions ranked by relevance.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        query: z.string().describe("Semantic search query."),
        target_peer_id: z
          .string()
          .optional()
          .describe("Optional: search conclusions about this target."),
        top_k: z
          .number()
          .optional()
          .describe("Max results to return."),
        filters: z
          .record(z.string(), z.unknown())
          .optional()
          .describe(
            'Optional: filter criteria, e.g. {"level": ["deductive", "inductive"]} to only return conclusions derived during dreaming. Levels: explicit (extracted directly from messages), deductive, inductive, contradiction. See https://honcho.dev/docs/v3/documentation/features/advanced/using-filters',
          ),
      },
    },
    async ({ workspace_id, peer_id, query, target_peer_id, top_k, filters }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = target_peer_id
          ? peer.conclusionsOf(target_peer_id)
          : peer.conclusions;
        const conclusions = await scope.query(query, top_k, undefined, filters);
        return textResult(
          conclusions.map((c) => ({
            id: c.id,
            content: c.content,
            level: c.level,
            observer_id: c.observerId,
            observed_id: c.observedId,
            session_id: c.sessionId,
            created_at: c.createdAt,
          })),
        );
      } catch (e) {
        return errorResult(
          `Query failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── create_conclusions ──────────────────────────────────────────────
  server.registerTool(
    "create_conclusions",
    {
      description: [
        "Manually create conclusions (facts/observations) about a peer.",
        "Use this to inject knowledge into Honcho that wasn't derived from conversation.",
        "Returns the number of conclusions created.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .describe("The peer the conclusions are about."),
        conclusions: z
          .array(z.string())
          .describe("Conclusion content strings to create."),
        session_id: z
          .string()
          .optional()
          .describe(
            "Optional: associate conclusions with a session. Omit for global conclusions.",
          ),
      },
    },
    async ({
      workspace_id,
      peer_id,
      target_peer_id,
      conclusions,
      session_id,
    }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = peer.conclusionsOf(target_peer_id);
        const params = conclusions.map((content) => ({
          content,
          sessionId: session_id,
        }));
        await scope.create(params);
        return textResult(
          `Created ${conclusions.length} conclusion${conclusions.length === 1 ? "" : "s"} successfully`,
        );
      } catch (e) {
        return errorResult(
          `Failed to create conclusions: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── delete_conclusion ───────────────────────────────────────────────
  server.registerTool(
    "delete_conclusion",
    {
      description: [
        "Delete a specific conclusion by ID.",
        "Use query_conclusions or list_conclusions to find the ID first.",
        "Use this to remove incorrect or outdated knowledge.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer."),
        target_peer_id: z
          .string()
          .describe("The peer the conclusion is about."),
        conclusion_id: z.string().describe("The conclusion to delete."),
      },
    },
    async ({ workspace_id, peer_id, target_peer_id, conclusion_id }) => {
      try {
        const peer = await ctx.clientFor(workspace_id).peer(peer_id);
        const scope = peer.conclusionsOf(target_peer_id);
        await scope.delete(conclusion_id);
        return textResult("Conclusion deleted successfully");
      } catch (e) {
        return errorResult(
          `Failed to delete conclusion: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

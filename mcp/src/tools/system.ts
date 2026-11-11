import { z } from "zod";
import type { DreamRunDetail } from "@honcho-ai/sdk";
import { HonchoError } from "@honcho-ai/sdk";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import {
  failedResult,
  isTokenRejected,
  textResult,
  workspaceIdSchema,
} from "../types.js";

/** Most created or removed conclusions `get_dream` returns per section. */
export const DREAM_SECTION_LIMIT = 50;

const DREAM_HISTORY_NOTE =
  "History covers dreams since Representation History was enabled on the server and is kept for its retention window (30 days by default). Past the window a run keeps only the conclusions it created that still exist, and its `pruned_at` is set.";

/** The dream routes refuse peer-scoped keys; say so instead of a bare 401. */
function dreamFailure(ctx: ToolContext, prefix: string, e: unknown) {
  if (
    e instanceof HonchoError &&
    e.status === 401 &&
    !isTokenRejected(e, ctx.config.apiKey)
  ) {
    return failedResult(
      ctx,
      `${prefix} (dream history needs a workspace or admin key)`,
      e,
    );
  }
  return failedResult(ctx, prefix, e);
}

/** Cap the created and removed sections, reporting how many were left out. */
export function formatDreamDetail(detail: DreamRunDetail) {
  const { created, removed, ...run } = detail;
  return {
    ...run,
    created: created.slice(0, DREAM_SECTION_LIMIT),
    created_omitted: Math.max(created.length - DREAM_SECTION_LIMIT, 0),
    removed: removed.slice(0, DREAM_SECTION_LIMIT),
    removed_omitted: Math.max(removed.length - DREAM_SECTION_LIMIT, 0),
  };
}

export function register(server: McpServer, ctx: ToolContext) {
  // ── schedule_dream ──────────────────────────────────────────────────
  server.registerTool(
    "schedule_dream",
    {
      annotations: {
        title: "Schedule Dream",
        readOnlyHint: false,
        destructiveHint: true,
      },
      description: [
        "Schedule a dream — a background memory-consolidation task for a peer.",
        "Dreams consolidate observations into higher-level insights and update peer cards.",
        "Use this after a long conversation to improve Honcho's memory quality.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        peer_id: z.string().describe("The observer peer to dream for."),
        target_peer_id: z
          .string()
          .optional()
          .describe(
            "Optional: dream about this target peer. Omit for self-reflection.",
          ),
        session_id: z
          .string()
          .optional()
          .describe("Optional: scope the dream to a session."),
      },
    },
    async ({ workspace_id, peer_id, target_peer_id, session_id }) => {
      try {
        await ctx.clientFor(workspace_id).scheduleDream({
          observer: peer_id,
          observed: target_peer_id,
          session: session_id,
        });
        return textResult("Dream scheduled successfully");
      } catch (e) {
        return failedResult(ctx, `Failed to schedule dream`, e);
      }
    },
  );

  // ── list_dreams ─────────────────────────────────────────────────────
  server.registerTool(
    "list_dreams",
    {
      annotations: {
        title: "List Dreams",
        readOnlyHint: true,
      },
      description: [
        "List the dreams that changed representations in this workspace, newest first (paginated).",
        "Each dream has its status, start and finish times, the observer and observed peer, and how many conclusions it created and removed and whether it changed the peer card.",
        "Use get_dream with a dream's id to see exactly what it changed.",
        DREAM_HISTORY_NOTE,
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        observed_id: z
          .string()
          .optional()
          .describe("Optional: only dreams about this peer."),
        observer_id: z
          .string()
          .optional()
          .describe(
            "Optional: only dreams from this observer's perspective. With observed_id, one representation.",
          ),
        page: z.number().int().min(1).optional().describe("Page number (1-indexed)."),
        size: z
          .number()
          .int()
          .min(1)
          .max(100)
          .optional()
          .describe("Results per page (max 100)."),
      },
    },
    async ({ workspace_id, observed_id, observer_id, page: pageNum, size }) => {
      try {
        const page = await ctx.clientFor(workspace_id).dreams({
          observed: observed_id,
          observer: observer_id,
          page: pageNum,
          size,
        });
        return textResult({
          dreams: page.items,
          total: page.total,
          page: page.page,
          pages: page.pages,
        });
      } catch (e) {
        return dreamFailure(ctx, "Failed to list dreams", e);
      }
    },
  );

  // ── get_dream ───────────────────────────────────────────────────────
  server.registerTool(
    "get_dream",
    {
      annotations: {
        title: "Get Dream",
        readOnlyHint: true,
      },
      description: [
        "Get what one dream changed in a representation.",
        "Returns the conclusions it created, each with the conclusions it was derived from; the conclusions it removed, as they read when removed, with `replaced_by` when a newer conclusion superseded one; and the peer card before and after.",
        "Each referenced conclusion has a status: `current` (still in the representation), `removed` (a later dream removed it; `removed_in_run` names that dream), or `unavailable` (deleted, or removed longer ago than history is kept).",
        `At most ${DREAM_SECTION_LIMIT} created and ${DREAM_SECTION_LIMIT} removed conclusions are returned; created_omitted and removed_omitted count the rest.`,
        DREAM_HISTORY_NOTE,
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
        dream_id: z
          .string()
          .regex(/^[A-Za-z0-9_-]+$/, "A dream id from list_dreams")
          .describe("The dream's id, from list_dreams."),
      },
    },
    async ({ workspace_id, dream_id }) => {
      try {
        const detail = await ctx.clientFor(workspace_id).getDream(dream_id);
        return textResult(formatDreamDetail(detail));
      } catch (e) {
        return dreamFailure(ctx, `Failed to get dream ${dream_id}`, e);
      }
    },
  );

  // ── get_queue_status ────────────────────────────────────────────────
  server.registerTool(
    "get_queue_status",
    {
      annotations: {
        title: "Get Queue Status",
        readOnlyHint: true,
      },
      description: [
        "Get the current processing queue status for background tasks (message derivation, dreams).",
        "Use this to check if Honcho is still processing messages before querying for insights.",
        "Returns work unit counts: total, completed, in-progress, and pending.",
      ].join("\n"),
      inputSchema: {
        workspace_id: workspaceIdSchema(ctx),
      },
    },
    async ({ workspace_id }) => {
      try {
        const status = await ctx.clientFor(workspace_id).queueStatus();
        return textResult(status);
      } catch (e) {
        return failedResult(ctx, `Failed to get queue status`, e);
      }
    },
  );
}

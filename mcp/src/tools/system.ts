import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolContext } from "../types.js";
import { textResult, errorResult } from "../types.js";

export function register(server: McpServer, ctx: ToolContext) {
  // ── schedule_dream ──────────────────────────────────────────────────
  server.registerTool(
    "schedule_dream",
    {
      description: [
        "Schedule a dream — a background memory-consolidation task for a peer.",
        "Dreams consolidate observations into higher-level insights and update peer cards.",
        "Use this after a long conversation to improve Honcho's memory quality.",
        "Returns a confirmation message.",
      ].join("\n"),
      inputSchema: {
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
    async ({ peer_id, target_peer_id, session_id }) => {
      try {
        await ctx.honcho.scheduleDream({
          observer: peer_id,
          observed: target_peer_id,
          session: session_id,
        });
        return textResult("Dream scheduled successfully");
      } catch (e) {
        return errorResult(
          `Failed to schedule dream: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );

  // ── get_queue_status ────────────────────────────────────────────────
  server.registerTool(
    "get_queue_status",
    {
      description: [
        "Get the current processing queue status for background tasks (message derivation, dreams).",
        "Use this to check if Honcho is still processing messages before querying for insights.",
        "Returns work unit counts: total, completed, in-progress, and pending.",
      ].join("\n"),
      inputSchema: {},
    },
    async () => {
      try {
        const status = await ctx.honcho.queueStatus();
        return textResult(status);
      } catch (e) {
        return errorResult(
          `Failed to get queue status: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
  );
}

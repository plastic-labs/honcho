/**
 * Query Honcho memory via the Dialectic API — exposed as an agent tool.
 */

import { tool } from "@openai/agents";
import { z } from "zod";
import { getClient } from "./client.js";
import type { HonchoContext } from "./client.js";

export const queryMemoryTool = tool({
  name: "query_memory",
  description:
    "Query what Honcho knows about the current user using natural language. " +
    "Use this when the user asks what you remember about them.",
  parameters: z.object({
    query: z.string().describe("Natural language question about the user"),
  }),
  execute: async ({ query }, runContext) => {
    const trimmed = query.trim();
    if (!trimmed) {
      throw new Error("query must not be empty");
    }

    const ctx = runContext?.context as HonchoContext | undefined;
    if (!ctx?.userId) {
      throw new Error("Missing Honcho context (userId)");
    }
    try {
      const honcho = getClient();
      const peer = honcho.peer(ctx.userId);
      const response = await peer.chat(trimmed);
      return response ?? "No relevant information found in memory.";
    } catch (err) {
      throw new Error(
        `Failed to query Honcho memory: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  },
});

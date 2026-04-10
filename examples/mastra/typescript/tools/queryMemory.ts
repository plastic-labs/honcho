import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { getClient } from "./client.js";
import type { HonchoContext } from "./client.js";

export function makeQueryMemoryTool(ctx: HonchoContext) {
  return createTool({
    id: "query_memory",
    description:
      "Query Honcho's Dialectic API to recall facts about the current user. " +
      "Use this when the user asks what you remember about them.",
    inputSchema: z.object({
      query: z
        .string()
        .describe("Natural language question about the user"),
    }),
    execute: async ({ context: { query } }) => {
      const honcho = getClient();
      const peer = honcho.peer(ctx.userId);
      const response = await peer.chat(query);
      return { result: response ?? "No relevant information found in memory." };
    },
  });
}

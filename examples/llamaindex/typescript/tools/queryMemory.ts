import { FunctionTool } from "llamaindex";
import { getClient } from "./client.js";
import type { HonchoContext } from "./client.js";

export function makeQueryMemoryTool(ctx: HonchoContext): FunctionTool {
  return new FunctionTool(
    async ({ query }: { query: string }): Promise<string> => {
      const honcho = getClient();
      const peer = honcho.peer(ctx.userId);
      const response = await peer.chat(query);
      return response ?? "No relevant information found in memory.";
    },
    {
      name: "query_memory",
      description:
        "Query Honcho's Dialectic API to recall facts about the current user. " +
        "Use this when the user asks what you remember about them.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "Natural language question about the user, e.g. 'What are my hobbies?'",
          },
        },
        required: ["query"],
      },
    }
  );
}

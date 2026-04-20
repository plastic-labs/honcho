/**
 * LlamaIndex (TypeScript) + Honcho persistent memory integration.
 *
 * Demonstrates a conversational agent that remembers users across sessions.
 * Honcho stores every message and builds a long-term representation of the user;
 * the agent injects that context into its chat history on every turn and can
 * query memory on demand via the ``query_memory`` tool.
 *
 * Usage:
 *   bun run main.ts
 *
 * Environment variables:
 *   HONCHO_API_KEY      Required. Your Honcho API key from honcho.dev.
 *   HONCHO_WORKSPACE_ID Optional. Workspace ID (default: "default").
 *   OPENAI_API_KEY      Required. Your OpenAI API key.
 */

import { ReActAgent, OpenAI } from "llamaindex";
import * as readline from "readline/promises";

import { createContext } from "./tools/client.js";
import type { HonchoContext } from "./tools/client.js";
import { getContext } from "./tools/getContext.js";
import { makeQueryMemoryTool } from "./tools/queryMemory.js";
import { saveMemory } from "./tools/saveMemory.js";

async function chat(
  userId: string,
  message: string,
  sessionId: string
): Promise<string> {
  const ctx: HonchoContext = createContext(userId, sessionId);

  const base =
    "You are a helpful assistant with persistent memory powered by Honcho. " +
    "You remember users across conversations. " +
    "When a user asks what you remember about them, use the query_memory tool.";

  const history = await getContext(ctx, 2000);
  const systemContent =
    history.length > 0
      ? `${base}\n\n## Conversation History\n${history
          .map(
            (m) =>
              `${m.role.charAt(0).toUpperCase() + m.role.slice(1)}: ${m.content}`
          )
          .join("\n")}`
      : base;

  const llm = new OpenAI({ model: "gpt-4.1-mini" });
  const agent = new ReActAgent({
    tools: [makeQueryMemoryTool(ctx)],
    llm,
    verbose: false,
  });

  await saveMemory(userId, message, "user", sessionId);

  const result = await agent.chat({
    message,
    chatHistory: [{ role: "system", content: systemContent }],
  });
  const response = result.message.content as string;

  await saveMemory(userId, response, "assistant", sessionId);

  return response;
}

async function main() {
  console.log("LlamaIndex HonchoMemoryAgent — type 'quit' to exit\n");
  const userId = "demo-user";
  const sessionId = "demo-session";

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = (await rl.question("You: ")).trim();
    if (!userInput) continue;
    if (["quit", "exit"].includes(userInput.toLowerCase())) {
      rl.close();
      break;
    }
    const response = await chat(userId, userInput, sessionId);
    console.log(`Agent: ${response}\n`);
  }
}

main().catch(console.error);

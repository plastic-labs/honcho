/**
 * Vercel AI SDK + Honcho persistent memory integration.
 *
 * Demonstrates a conversational agent that remembers users across sessions.
 * Honcho stores every message and builds a long-term representation of the user;
 * the agent injects that context into its system prompt on every turn and can
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

import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import * as readline from "readline/promises";

import { createContext } from "./tools/client.js";
import type { HonchoContext } from "./tools/client.js";
import { getContext } from "./tools/getContext.js";
import { makeQueryMemoryTool } from "./tools/queryMemory.js";
import { saveMemory } from "./tools/saveMemory.js";

export async function chat(
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
  const system =
    history.length > 0
      ? `${base}\n\n## Conversation History\n${history
          .map(
            (m) =>
              `${m.role.charAt(0).toUpperCase() + m.role.slice(1)}: ${m.content}`
          )
          .join("\n")}`
      : base;

  await saveMemory(userId, message, "user", sessionId);

  const { text } = await generateText({
    model: openai("gpt-4.1-mini"),
    system,
    prompt: message,
    tools: { query_memory: makeQueryMemoryTool(ctx) },
    maxSteps: 5,
  });

  await saveMemory(userId, text, "assistant", sessionId);
  return text;
}

async function main() {
  console.log("Vercel AI HonchoMemoryAgent — type 'quit' to exit\n");
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
    try {
      const response = await chat(userId, userInput, sessionId);
      console.log(`Agent: ${response}\n`);
    } catch (err) {
      console.error(`Error: ${err instanceof Error ? err.message : String(err)}\n`);
    }
  }
}

main().catch(console.error);

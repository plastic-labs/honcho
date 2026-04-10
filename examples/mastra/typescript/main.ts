/**
 * Mastra + Honcho persistent memory integration.
 *
 * Demonstrates a conversational agent that remembers users across sessions.
 * Honcho stores every message and builds a long-term representation of the user;
 * the agent injects that context into its instructions on every turn and can
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

import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";
import * as readline from "readline/promises";

import { createContext } from "./tools/client.js";
import type { HonchoContext } from "./tools/client.js";
import { getContext } from "./tools/getContext.js";
import { makeQueryMemoryTool } from "./tools/queryMemory.js";
import { saveMemory } from "./tools/saveMemory.js";

async function buildInstructions(ctx: HonchoContext): Promise<string> {
  const base =
    "You are a helpful assistant with persistent memory powered by Honcho. " +
    "You remember users across conversations. " +
    "When a user asks what you remember about them, use the query_memory tool.";

  const history = await getContext(ctx, 2000);
  if (history.length === 0) return base;

  const formatted = history
    .map(
      (m) =>
        `${m.role.charAt(0).toUpperCase() + m.role.slice(1)}: ${m.content}`
    )
    .join("\n");

  return `${base}\n\n## Conversation History\n${formatted}`;
}

export async function chat(
  userId: string,
  message: string,
  sessionId: string
): Promise<string> {
  const ctx: HonchoContext = createContext(userId, sessionId);

  await saveMemory(userId, message, "user", sessionId);

  const instructions = await buildInstructions(ctx);

  const agent = new Agent({
    name: "HonchoMemoryAgent",
    instructions,
    model: openai("gpt-4.1-mini"),
    tools: { query_memory: makeQueryMemoryTool(ctx) },
  });

  const result = await agent.generate(message);
  const response = result.text;

  await saveMemory(userId, response, "assistant", sessionId);
  return response;
}

async function main() {
  console.log("Mastra HonchoMemoryAgent — type 'quit' to exit\n");
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

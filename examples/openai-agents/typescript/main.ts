/**
 * OpenAI Agents JS + Honcho persistent memory integration.
 *
 * Demonstrates a conversational agent that remembers users across sessions.
 * Honcho stores every message and builds a long-term representation of the user;
 * the agent injects that context into its instructions on every turn and can
 * query memory on demand via the `query_memory` tool.
 *
 * Usage:
 *   bun run main.ts
 *
 * Environment variables:
 *   HONCHO_API_KEY      Required. Your Honcho API key from honcho.dev.
 *   HONCHO_WORKSPACE_ID Optional. Workspace ID (default: "default").
 *   OPENAI_API_KEY      Required. Your OpenAI API key.
 */

import { Agent, run } from "@openai/agents";
import type { RunContext } from "@openai/agents";
import * as readline from "readline/promises";
import { randomUUID } from "crypto";

import { createContext, getClient } from "./tools/client.js";
import type { HonchoContext } from "./tools/client.js";
import { getContext } from "./tools/getContext.js";
import { queryMemoryTool } from "./tools/queryMemory.js";
import { saveMemory } from "./tools/saveMemory.js";

function buildInstructions(
  runContext: RunContext<HonchoContext>,
  _agent: Agent<HonchoContext>
): string {
  const base =
    "You are a helpful assistant with persistent memory powered by Honcho. " +
    "You remember users across conversations. " +
    "When a user asks what you remember about them, use the query_memory tool.";

  return base;
}

const honchoAgent = new Agent<HonchoContext>({
  name: "HonchoMemoryAgent",
  instructions: buildInstructions,
  tools: [queryMemoryTool],
  model: "gpt-4.1-mini",
});

async function setupSession(
  userId: string,
  sessionId: string,
  assistantId = "assistant"
): Promise<void> {
  const honcho = getClient();
  const userPeer = honcho.peer(userId);
  const assistantPeer = honcho.peer(assistantId);
  const session = honcho.session(sessionId);
  await session.addPeers([userPeer, assistantPeer]);
}

export async function chat(
  userId: string,
  message: string,
  sessionId: string,
  assistantId = "assistant"
): Promise<string> {
  const ctx = createContext(userId, sessionId, assistantId);

  // Fetch prior history BEFORE saving the current turn to avoid duplicating it
  const history = await getContext(ctx, 2000);
  const input = [...history, { role: "user" as const, content: message }];

  await saveMemory(userId, message, "user", sessionId, assistantId);

  const result = await run(honchoAgent, input, { context: ctx });
  const response = result.finalOutput ?? "";

  await saveMemory(userId, response, "assistant", sessionId, assistantId);
  return response;
}

async function main() {
  console.log("OpenAI Agents HonchoMemoryAgent — type 'quit' to exit\n");
  // Replace "demo-user" with a real user identifier in production.
  const userId = "demo-user";
  // A fresh session ID per run prevents history from accumulating across runs.
  const sessionId = randomUUID();

  await setupSession(userId, sessionId);

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

/**
 * OpenAI Agents SDK (TypeScript) integration with Honcho persistent memory.
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

import { Agent, run } from '@openai/agents';
import type { RunContext } from '@openai/agents';
import * as readline from 'readline/promises';

import { createContext } from './tools/client.js';
import type { HonchoContext } from './tools/client.js';
import { getContext } from './tools/getContext.js';
import { queryMemory } from './tools/queryMemory.js';
import { saveMemory } from './tools/saveMemory.js';

/**
 * Build dynamic system instructions that include Honcho memory context.
 *
 * Called by the OpenAI Agents SDK before every LLM request. Fetches recent
 * conversation history from Honcho and prepends it so the model always has
 * an up-to-date view of the session.
 *
 * @param runContext - Run context wrapping the HonchoContext.
 * @returns System prompt string with injected conversation history.
 */
async function honchoInstructions(
  runContext: RunContext<HonchoContext>
): Promise<string> {
  const base =
    'You are a helpful assistant with persistent memory powered by Honcho. ' +
    'You remember users across conversations. ' +
    'When a user asks what you remember about them, use the query_memory tool.';

  const history = await getContext(runContext.context, 2000);
  if (history.length === 0) return base;

  const formatted = history
    .map((m) => `${m.role.charAt(0).toUpperCase() + m.role.slice(1)}: ${m.content}`)
    .join('\n');

  return `${base}\n\n## Conversation History\n${formatted}`;
}

const honchoAgent = new Agent<HonchoContext>({
  name: 'HonchoMemoryAgent',
  instructions: honchoInstructions,
  tools: [queryMemory],
  model: 'gpt-4.1-mini',
});

/**
 * Run one conversation turn with persistent Honcho memory.
 *
 * Saves the user message to Honcho before the agent runs, then saves the
 * assistant reply afterwards. The dynamic instructions callable injects
 * the full Honcho context for every turn automatically.
 *
 * @param userId - Unique identifier for the user.
 * @param message - The user's input message.
 * @param sessionId - Identifier for the current conversation session.
 * @returns The agent's response as a string.
 */
export async function chat(
  userId: string,
  message: string,
  sessionId: string
): Promise<string> {
  const ctx = createContext(userId, sessionId);

  // Persist user message before the agent runs so it's available in context
  await saveMemory(userId, message, 'user', sessionId);

  const result = await run(honchoAgent, message, { context: ctx });
  const response = String(result.finalOutput);

  // Persist assistant response after the run
  await saveMemory(userId, response, 'assistant', sessionId);

  return response;
}

async function main() {
  console.log("HonchoMemoryAgent — type 'quit' to exit\n");
  const userId = 'demo-user';
  const sessionId = 'demo-session';

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = (await rl.question('You: ')).trim();
    if (!userInput) continue;
    if (userInput.toLowerCase() === 'quit' || userInput.toLowerCase() === 'exit') {
      rl.close();
      break;
    }
    const response = await chat(userId, userInput, sessionId);
    console.log(`Agent: ${response}\n`);
  }
}

main().catch(console.error);

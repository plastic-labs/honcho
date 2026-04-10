/**
 * Query Honcho memory via the Dialectic API — exposed as an agent tool.
 */

import { tool } from '@openai/agents';
import type { RunContext } from '@openai/agents';
import { z } from 'zod';

import type { HonchoContext } from './client.js';
import { getClient } from './client.js';

/**
 * Agent tool that queries what Honcho knows about the current user.
 *
 * Sends a natural language question to Honcho's Dialectic API and returns an
 * answer grounded in the peer's long-term memory and stored observations. The
 * agent calls this tool when the user asks about their own history, preferences,
 * or past conversations.
 */
export const queryMemory = tool({
  name: 'query_memory',
  description:
    'Query what Honcho knows about the current user using natural language. ' +
    'Use this when the user asks what you remember about them, their preferences, ' +
    'or anything from previous conversations.',
  parameters: z.object({
    query: z
      .string()
      .describe(
        'Natural language question about the user, e.g. "What are my hobbies?" ' +
          'or "What did we discuss last time?"'
      ),
  }),
  execute: async (
    { query }: { query: string },
    runContext?: RunContext<HonchoContext>
  ): Promise<string> => {
    if (!query) throw new Error('query must not be empty');

    const ctx = runContext!.context;
    const honcho = getClient();
    const peer = honcho.peer(ctx.userId);
    const response = await peer.chat(query);

    return response ?? 'No relevant information found in memory.';
  },
});

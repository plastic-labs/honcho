/**
 * Retrieve Honcho conversation context formatted for LLM injection.
 */

import type { HonchoContext } from './client.js';
import { getClient } from './client.js';

/**
 * Retrieve conversation context ready for injection into an LLM prompt.
 *
 * Fetches recent messages from a Honcho session within the given token budget
 * and converts them to OpenAI-compatible message format. The returned array is
 * suitable for use in dynamic agent instructions.
 *
 * @param ctx - HonchoContext holding the user, session, and assistant IDs.
 * @param tokens - Maximum number of tokens to include (default: 2000).
 * @returns An array of message objects in OpenAI format:
 *          `[{ role: "user" | "assistant", content: "..." }]`.
 *          Returns an empty array if the session has no messages yet.
 */
export async function getContext(
  ctx: HonchoContext,
  tokens = 2000
): Promise<Array<{ role: string; content: string }>> {
  const honcho = getClient();
  const userPeer = honcho.peer(ctx.userId);
  const assistantPeer = honcho.peer(ctx.assistantId);
  const session = honcho.session(ctx.sessionId);

  await session.addPeers([userPeer, assistantPeer]);

  const sessionCtx = await session.context({ tokens });
  return sessionCtx.toOpenAI(assistantPeer);
}

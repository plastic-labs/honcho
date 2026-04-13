/**
 * Retrieve Honcho conversation context formatted for LLM injection.
 */

import { getClient } from "./client.js";
import type { HonchoContext } from "./client.js";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export async function getContext(
  ctx: HonchoContext,
  tokens = 2000
): Promise<Message[]> {
  const honcho = getClient();
  const userPeer = honcho.peer(ctx.userId);
  const assistantPeer = honcho.peer(ctx.assistantId);
  const session = honcho.session(ctx.sessionId);

  await session.addPeers([userPeer, assistantPeer]);

  const context = await session.context({ tokens });
  return context.toOpenai({ assistant: ctx.assistantId }) as Message[];
}

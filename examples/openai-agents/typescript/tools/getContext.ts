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
  const session = honcho.session(ctx.sessionId);
  const context = await session.context({ tokens });
  // Strip the 'name' field — the OpenAI Responses API does not accept it
  return (context.toOpenai({ assistant: ctx.assistantId }) as Array<Record<string, string>>).map(
    ({ role, content }) => ({ role: role as Message["role"], content })
  );
}

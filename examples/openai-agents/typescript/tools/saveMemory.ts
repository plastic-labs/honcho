/**
 * Save a conversation message to Honcho memory.
 */

import { getClient } from "./client.js";

type Role = "user" | "assistant";

export async function saveMemory(
  userId: string,
  content: string,
  role: Role,
  sessionId: string,
  assistantId = "assistant"
): Promise<void> {
  if (!content) {
    throw new Error("content must not be empty");
  }

  const honcho = getClient();
  const userPeer = honcho.peer(userId);
  const assistantPeer = honcho.peer(assistantId);
  const session = honcho.session(sessionId);

  await session.addPeers([userPeer, assistantPeer]);

  const sender = role === "assistant" ? assistantPeer : userPeer;
  await session.addMessages([sender.message(content)]);
}

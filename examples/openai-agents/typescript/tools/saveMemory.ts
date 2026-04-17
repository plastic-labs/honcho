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
  const cleanedContent = content.trim();
  if (!cleanedContent) {
    throw new Error("content must not be empty");
  }

  const honcho = getClient();
  const userPeer = honcho.peer(userId);
  const assistantPeer = honcho.peer(assistantId);
  const session = honcho.session(sessionId);

  const sender = role === "assistant" ? assistantPeer : userPeer;
  await session.addMessages([sender.message(cleanedContent)]);
}

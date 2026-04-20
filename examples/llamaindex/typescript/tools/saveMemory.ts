import { getClient } from "./client.js";

export async function saveMemory(
  userId: string,
  content: string,
  role: "user" | "assistant",
  sessionId: string,
  assistantId = "assistant"
): Promise<void> {
  const honcho = getClient();
  const userPeer = honcho.peer(userId);
  const assistantPeer = honcho.peer(assistantId);
  const session = honcho.session(sessionId);
  await session.addPeers([userPeer, assistantPeer]);
  const sender = role === "assistant" ? assistantPeer : userPeer;
  await session.addMessages([sender.message(content)]);
}

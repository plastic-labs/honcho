/**
 * Save a conversation message to Honcho memory.
 */

import { getClient } from './client.js';

/**
 * Save a single conversation turn to Honcho memory.
 *
 * Creates the peer and session if they do not already exist. Registers both
 * peers in the session on first use, then persists the message.
 *
 * @param userId - Unique identifier for the user peer.
 * @param content - Text content of the message to save.
 * @param role - Either "user" or "assistant". Any value other than "assistant"
 *               is treated as the user peer.
 * @param sessionId - Identifier for the conversation session.
 * @param assistantId - Peer ID for the assistant (default: "assistant").
 * @returns A confirmation string describing what was saved.
 * @throws {Error} If content is empty.
 */
export async function saveMemory(
  userId: string,
  content: string,
  role: string,
  sessionId: string,
  assistantId = 'assistant'
): Promise<string> {
  if (!content) throw new Error('content must not be empty');

  const honcho = getClient();
  const userPeer = honcho.peer(userId);
  const assistantPeer = honcho.peer(assistantId);
  const session = honcho.session(sessionId);

  await session.addPeers([userPeer, assistantPeer]);

  const sender = role === 'assistant' ? assistantPeer : userPeer;
  await session.addMessages([sender.message(content)]);

  return `Saved ${role} message to session '${sessionId}' for user '${userId}'.`;
}

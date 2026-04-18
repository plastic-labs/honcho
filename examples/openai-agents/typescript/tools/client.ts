/**
 * Honcho client initialization and context for OpenAI Agents JS integration.
 */

import Honcho from "honcho-ai";

export interface HonchoContext {
  userId: string;
  sessionId: string;
  assistantId: string;
}

export function createContext(
  userId: string,
  sessionId: string,
  assistantId = "assistant"
): HonchoContext {
  return { userId, sessionId, assistantId };
}

export function getClient(): Honcho {
  const apiKey = process.env.HONCHO_API_KEY;
  if (!apiKey) {
    throw new Error(
      "HONCHO_API_KEY is required. Set it in your environment or .env file."
    );
  }
  const workspaceId = process.env.HONCHO_WORKSPACE_ID ?? "default";
  return new Honcho({ apiKey, workspaceId });
}

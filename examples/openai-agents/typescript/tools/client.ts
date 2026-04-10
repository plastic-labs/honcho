/**
 * Honcho client initialization and context for the OpenAI Agents SDK integration.
 */

import { Honcho } from '@honcho-ai/sdk';
import * as dotenv from 'dotenv';

dotenv.config();

/**
 * Holds Honcho identity for a single conversation turn.
 *
 * Pass this as the `context` argument to `run()`. Tools and the dynamic
 * `instructions` function read from it to resolve the correct peer and session
 * without requiring global state.
 */
export interface HonchoContext {
  /** Unique identifier for the human peer. */
  userId: string;
  /** Identifier for the current conversation session. */
  sessionId: string;
  /** Peer ID for the assistant. Defaults to "assistant". */
  assistantId: string;
}

/**
 * Create a HonchoContext with sensible defaults.
 *
 * @param userId - Unique identifier for the human peer.
 * @param sessionId - Identifier for the current conversation session.
 * @param assistantId - Peer ID for the assistant (default: "assistant").
 */
export function createContext(
  userId: string,
  sessionId: string,
  assistantId = 'assistant'
): HonchoContext {
  return { userId, sessionId, assistantId };
}

/**
 * Initialize and return a Honcho client.
 *
 * Reads HONCHO_API_KEY and HONCHO_WORKSPACE_ID from environment variables.
 * The SDK will throw if HONCHO_API_KEY is not set.
 */
export function getClient(): Honcho {
  return new Honcho({});
}

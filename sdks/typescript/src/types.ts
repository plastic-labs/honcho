import type { Session } from './session'

/**
 * API version used for all Honcho API requests.
 */
export const API_VERSION = 'v3'

// Re-export all API types
export * from './types/api'

/**
 * Shared types for the Honcho TypeScript SDK.
 */

/**
 * Conclusion - external view of a document (theory-of-mind data).
 */
export interface Conclusion {
  id: string
  content: string
  observer_id: string
  observed_id: string
  session_id: string
  created_at: string
}

/**
 * Parameters for creating a conclusion.
 */
export interface ConclusionCreateParam {
  /** The conclusion content/text */
  content: string
  /** The session this conclusion relates to (ID string or Session object) */
  sessionId: string | Session
}

/**
 * Parameters for semantic search of conclusions.
 */
export interface ConclusionQueryParams {
  query: string
  top_k?: number
  distance?: number
  filters?: Record<string, unknown>
}

/**
 * Delta object for streaming dialectic responses.
 */
export interface DialecticStreamDelta {
  content?: string
}

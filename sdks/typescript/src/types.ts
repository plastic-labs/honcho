import type { Session } from './session'

/**
 * Shared types for the Honcho TypeScript SDK.
 */

// ============================================================================
// API Response Types (replacing @honcho-ai/core imports)
// ============================================================================

/**
 * Workspace response from the API.
 */
export interface Workspace {
  id: string
  metadata: Record<string, unknown>
  configuration: Record<string, unknown>
  created_at: string
}

/**
 * Peer response from the API.
 */
export interface PeerResponse {
  id: string
  workspace_id: string
  metadata: Record<string, unknown> | null
  configuration: Record<string, unknown> | null
  created_at: string
}

/**
 * Session response from the API.
 */
export interface SessionResponse {
  id: string
  workspace_id: string
  is_active: boolean
  metadata: Record<string, unknown> | null
  configuration: Record<string, unknown> | null
  created_at: string
}

/**
 * Message from the API.
 */
export interface Message {
  id: string
  content: string
  peer_id: string
  session_id: string
  workspace_id: string
  token_count: number
  metadata?: Record<string, unknown>
  created_at: string
}

/**
 * Parameters for creating a message.
 */
export interface MessageCreateParam {
  peer_id: string
  content: string
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
  created_at?: string
}

/**
 * Deriver status for a specific session.
 */
export interface SessionDeriverStatus {
  session_id: string | null
  total_work_units: number
  completed_work_units: number
  in_progress_work_units: number
  pending_work_units: number
}

/**
 * Deriver status response from the API.
 */
export interface DeriverStatus {
  total_work_units: number
  completed_work_units: number
  in_progress_work_units: number
  pending_work_units: number
  sessions?: Record<string, SessionDeriverStatus>
}

/**
 * Peer card response from the API.
 */
export interface PeerCardResponse {
  peer_card: string[] | null
}

/**
 * Dialectic response from the API (non-streaming).
 */
export interface DialecticResponse {
  content: string
}

/**
 * Summary from the API.
 */
export interface SummaryResponse {
  content: string
  message_id: string
  summary_type: string
  created_at: string
  token_count: number
}

/**
 * Session context response from the API.
 */
export interface SessionContextResponse {
  id: string
  messages: Message[]
  summary?: SummaryResponse
  peer_representation?: string
  peer_card?: string[]
}

/**
 * Peer context response from the API.
 */
export interface PeerContextResponse {
  peer_id: string
  target_id: string
  representation?: Record<string, unknown>
  peer_card?: string[]
}

/**
 * Session summaries response from the API.
 */
export interface SessionSummariesResponse {
  id: string
  short_summary?: SummaryResponse
  long_summary?: SummaryResponse
}

/**
 * Paginated response from the API.
 */
export interface PageResponse<T> {
  items: T[]
  total: number | null
  page: number
  size: number
  pages: number | null
}

/**
 * Session peer config from the API.
 */
export interface SessionPeerConfigResponse {
  observe_me?: boolean
  observe_others?: boolean
}

/**
 * File upload type - supports browser File, Node Buffer, or Uint8Array.
 */
export type Uploadable =
  | File
  | { filename: string; content: Buffer | Uint8Array; content_type: string }

// ============================================================================
// SDK Types
// ============================================================================

/**
 * Observation - external view of a document (theory-of-mind data).
 */
export interface Observation {
  id: string
  content: string
  observer_id: string
  observed_id: string
  session_id: string
  created_at: string
}

/**
 * Parameters for creating an observation.
 */
export interface ObservationCreateParam {
  /** The observation content/text */
  content: string
  /** The session this observation relates to (ID string or Session object) */
  sessionId: string | Session
}

/**
 * Parameters for semantic search of observations.
 */
export interface ObservationQueryParams {
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
  // Future fields can be added here:
  // premises?: string[]
  // tokens?: number
  // analytics?: Record<string, unknown>
}

/**
 * Chunk in a streaming dialectic response.
 */
export interface DialecticStreamChunk {
  delta: DialecticStreamDelta
  done: boolean
}

/**
 * Iterator for streaming dialectic responses with utilities for accessing the final response.
 *
 * Similar to OpenAI and Anthropic streaming patterns, this allows you to:
 * - Iterate over chunks as they arrive
 * - Access the final accumulated response after streaming completes
 *
 * @example
 * ```typescript
 * const stream = await peer.chat("Hello", { stream: true })
 *
 * // Stream chunks
 * for await (const chunk of stream) {
 *   process.stdout.write(chunk)
 * }
 *
 * // Get final response object
 * const final = stream.getFinalResponse()
 * console.log(`\nFull content: ${final.content}`)
 * ```
 */
export class DialecticStreamResponse implements AsyncIterable<string> {
  private iterator: AsyncIterator<string>
  private accumulatedContent: string[] = []
  private _isComplete = false

  constructor(iterator: AsyncIterator<string>) {
    this.iterator = iterator
  }

  [Symbol.asyncIterator](): AsyncIterator<string> {
    return {
      next: async () => {
        const result = await this.iterator.next()
        if (result.done) {
          this._isComplete = true
          return { done: true, value: undefined }
        }
        this.accumulatedContent.push(result.value)
        return { done: false, value: result.value }
      },
    }
  }

  /**
   * Get the final accumulated response after streaming completes.
   *
   * @returns An object with the full content
   *
   * @note This should be called after the stream has been fully consumed.
   *       If called before completion, it returns the content accumulated so far.
   */
  getFinalResponse(): { content: string } {
    return { content: this.accumulatedContent.join('') }
  }

  /**
   * Check if the stream has finished.
   */
  get isComplete(): boolean {
    return this._isComplete
  }
}

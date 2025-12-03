/**
 * Shared types for the Honcho TypeScript SDK.
 */

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

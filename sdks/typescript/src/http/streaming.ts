/**
 * Parse Server-Sent Events from a Response body.
 *
 * Yields parsed JSON data from each "data:" line in the SSE stream.
 */
export async function* parseSSE<T>(
  response: Response
): AsyncGenerator<T, void, undefined> {
  if (!response.body) {
    throw new Error('Response body is null')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6) // Remove "data: " prefix
          if (jsonStr.trim() === '[DONE]') {
            return
          }
          try {
            const data = JSON.parse(jsonStr) as T
            yield data
          } catch {
            // Skip invalid JSON lines
          }
        }
      }
    }

    // Process any remaining data in the buffer
    if (buffer.startsWith('data: ')) {
      const jsonStr = buffer.slice(6)
      if (jsonStr.trim() !== '[DONE]') {
        try {
          const data = JSON.parse(jsonStr) as T
          yield data
        } catch {
          // Skip invalid JSON
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * Chunk data from the dialectic streaming endpoint.
 */
export interface DialecticStreamChunk {
  done: boolean
  delta: {
    content?: string
  }
}

/**
 * Async iterable wrapper for dialectic streaming responses.
 *
 * Provides a convenient interface for iterating over streaming content
 * and collecting the final response.
 */
export class DialecticStreamResponse implements AsyncIterable<string> {
  private generator: AsyncGenerator<string, void, undefined>
  private chunks: string[] = []
  private consumed = false

  constructor(generator: AsyncGenerator<string, void, undefined>) {
    this.generator = generator
  }

  /**
   * Iterate over content chunks as they arrive.
   */
  async *[Symbol.asyncIterator](): AsyncGenerator<string, void, undefined> {
    if (this.consumed) {
      // If already consumed, yield from cached chunks
      for (const chunk of this.chunks) {
        yield chunk
      }
      return
    }

    for await (const chunk of this.generator) {
      this.chunks.push(chunk)
      yield chunk
    }
    this.consumed = true
  }

  /**
   * Get the complete response after streaming finishes.
   */
  async getFinalResponse(): Promise<string> {
    if (!this.consumed) {
      for await (const _ of this) {
        // Consume all chunks
      }
    }
    return this.chunks.join('')
  }

  /**
   * Collect all chunks into an array.
   */
  async toArray(): Promise<string[]> {
    if (!this.consumed) {
      for await (const _ of this) {
        // Consume all chunks
      }
    }
    return [...this.chunks]
  }
}

/**
 * Create a DialecticStreamResponse from an SSE response.
 */
export function createDialecticStream(
  response: Response
): DialecticStreamResponse {
  async function* streamContent(): AsyncGenerator<string, void, undefined> {
    for await (const chunk of parseSSE<DialecticStreamChunk>(response)) {
      if (chunk.done) {
        return
      }
      const content = chunk.delta?.content
      if (content) {
        yield content
      }
    }
  }

  return new DialecticStreamResponse(streamContent())
}

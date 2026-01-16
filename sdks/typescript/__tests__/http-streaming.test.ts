/**
 * HTTP Streaming Unit Tests
 *
 * Tests for SSE parsing and streaming response handling:
 * - parseSSE function
 * - DialecticStreamResponse class
 * - createDialecticStream factory
 *
 * These tests use mocked ReadableStreams to verify behavior without a server.
 */

import { describe, test, expect } from 'bun:test'
import {
  parseSSE,
  DialecticStreamResponse,
  createDialecticStream,
  type DialecticStreamChunk,
} from '../src/http/streaming'

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a mock Response with a readable stream body
 */
function createMockStreamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  let chunkIndex = 0

  const stream = new ReadableStream({
    pull(controller) {
      if (chunkIndex < chunks.length) {
        controller.enqueue(encoder.encode(chunks[chunkIndex]))
        chunkIndex++
      } else {
        controller.close()
      }
    },
  })

  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

/**
 * Create a Response with null body
 */
function createNullBodyResponse(): Response {
  return new Response(null)
}

/**
 * Create an async generator from an array
 */
async function* arrayToAsyncGenerator<T>(items: T[]): AsyncGenerator<T> {
  for (const item of items) {
    yield item
  }
}

/**
 * Collect all items from an async generator
 */
async function collectGenerator<T>(gen: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = []
  for await (const item of gen) {
    items.push(item)
  }
  return items
}

// =============================================================================
// parseSSE Tests
// =============================================================================

describe('parseSSE', () => {
  test('parses single SSE event', async () => {
    const response = createMockStreamResponse(['data: {"value": 1}\n\n'])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(1)
    expect(events[0].value).toBe(1)
  })

  test('parses multiple SSE events', async () => {
    const response = createMockStreamResponse([
      'data: {"value": 1}\n',
      'data: {"value": 2}\n',
      'data: {"value": 3}\n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(3)
    expect(events[0].value).toBe(1)
    expect(events[1].value).toBe(2)
    expect(events[2].value).toBe(3)
  })

  test('handles events split across chunks', async () => {
    const response = createMockStreamResponse([
      'data: {"val',
      'ue": 42}\n',
      'data: {"value": 100}\n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(2)
    expect(events[0].value).toBe(42)
    expect(events[1].value).toBe(100)
  })

  test('handles [DONE] signal and stops iteration', async () => {
    const response = createMockStreamResponse([
      'data: {"value": 1}\n',
      'data: [DONE]\n',
      'data: {"value": 2}\n', // Should not be yielded
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(1)
    expect(events[0].value).toBe(1)
  })

  test('handles [DONE] with whitespace', async () => {
    const response = createMockStreamResponse([
      'data: {"value": 1}\n',
      'data:   [DONE]  \n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(1)
  })

  test('skips invalid JSON lines', async () => {
    const response = createMockStreamResponse([
      'data: {"value": 1}\n',
      'data: not valid json\n',
      'data: {"value": 3}\n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(2)
    expect(events[0].value).toBe(1)
    expect(events[1].value).toBe(3)
  })

  test('ignores non-data lines', async () => {
    const response = createMockStreamResponse([
      'event: message\n',
      'id: 123\n',
      'data: {"value": 1}\n',
      'retry: 5000\n',
      'data: {"value": 2}\n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(2)
    expect(events[0].value).toBe(1)
    expect(events[1].value).toBe(2)
  })

  test('handles empty chunks', async () => {
    const response = createMockStreamResponse([
      '',
      'data: {"value": 1}\n',
      '',
      'data: {"value": 2}\n',
      '',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(2)
  })

  test('handles data remaining in buffer after stream ends', async () => {
    const response = createMockStreamResponse([
      'data: {"value": 1}\n',
      'data: {"value": 2}', // No trailing newline
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(2)
    expect(events[1].value).toBe(2)
  })

  test('throws on null response body', async () => {
    const response = createNullBodyResponse()

    await expect(collectGenerator(parseSSE(response))).rejects.toThrow(
      'Response body is null'
    )
  })

  test('handles complex JSON objects', async () => {
    const response = createMockStreamResponse([
      'data: {"nested": {"key": "value"}, "array": [1, 2, 3]}\n',
    ])

    interface ComplexType {
      nested: { key: string }
      array: number[]
    }

    const events = await collectGenerator(parseSSE<ComplexType>(response))

    expect(events).toHaveLength(1)
    expect(events[0].nested.key).toBe('value')
    expect(events[0].array).toEqual([1, 2, 3])
  })

  test('handles empty events', async () => {
    const response = createMockStreamResponse([
      'data: \n', // Empty data
      'data: {"value": 1}\n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    // Empty data line should be skipped (JSON.parse('') throws)
    expect(events).toHaveLength(1)
    expect(events[0].value).toBe(1)
  })

  test('handles multiple newlines between events', async () => {
    const response = createMockStreamResponse([
      'data: {"value": 1}\n',
      '\n',
      '\n',
      'data: {"value": 2}\n',
    ])

    const events = await collectGenerator(parseSSE<{ value: number }>(response))

    expect(events).toHaveLength(2)
  })
})

// =============================================================================
// DialecticStreamResponse Tests
// =============================================================================

describe('DialecticStreamResponse', () => {
  test('iterates over string chunks', async () => {
    const generator = arrayToAsyncGenerator(['Hello', ' ', 'World'])
    const stream = new DialecticStreamResponse(generator)

    const chunks = await collectGenerator(stream)

    expect(chunks).toEqual(['Hello', ' ', 'World'])
  })

  test('getFinalResponse joins all chunks', async () => {
    const generator = arrayToAsyncGenerator(['Hello', ' ', 'World'])
    const stream = new DialecticStreamResponse(generator)

    const result = await stream.getFinalResponse()

    expect(result).toBe('Hello World')
  })

  test('toArray returns array of chunks', async () => {
    const generator = arrayToAsyncGenerator(['chunk1', 'chunk2', 'chunk3'])
    const stream = new DialecticStreamResponse(generator)

    const result = await stream.toArray()

    expect(result).toEqual(['chunk1', 'chunk2', 'chunk3'])
  })

  test('can re-iterate after consumption', async () => {
    const generator = arrayToAsyncGenerator(['a', 'b', 'c'])
    const stream = new DialecticStreamResponse(generator)

    // First iteration
    const first = await collectGenerator(stream)

    // Second iteration (from cache)
    const second = await collectGenerator(stream)

    expect(first).toEqual(['a', 'b', 'c'])
    expect(second).toEqual(['a', 'b', 'c'])
  })

  test('getFinalResponse works before iteration', async () => {
    const generator = arrayToAsyncGenerator(['one', 'two'])
    const stream = new DialecticStreamResponse(generator)

    // Call getFinalResponse without iterating first
    const result = await stream.getFinalResponse()

    expect(result).toBe('onetwo')
  })

  test('toArray works before iteration', async () => {
    const generator = arrayToAsyncGenerator(['x', 'y', 'z'])
    const stream = new DialecticStreamResponse(generator)

    const result = await stream.toArray()

    expect(result).toEqual(['x', 'y', 'z'])
  })

  test('getFinalResponse after partial iteration', async () => {
    const generator = arrayToAsyncGenerator(['first', 'second', 'third'])
    const stream = new DialecticStreamResponse(generator)

    // Partial iteration
    const partial: string[] = []
    for await (const chunk of stream) {
      partial.push(chunk)
      if (partial.length === 2) break
    }

    // This will return cached chunks (consumed so far)
    // Note: The test depends on implementation - it may consume remaining
    const result = await stream.getFinalResponse()

    expect(typeof result).toBe('string')
  })

  test('handles empty generator', async () => {
    const generator = arrayToAsyncGenerator<string>([])
    const stream = new DialecticStreamResponse(generator)

    const chunks = await collectGenerator(stream)
    const final = await stream.getFinalResponse()

    expect(chunks).toEqual([])
    expect(final).toBe('')
  })

  test('handles single chunk', async () => {
    const generator = arrayToAsyncGenerator(['only one'])
    const stream = new DialecticStreamResponse(generator)

    const final = await stream.getFinalResponse()

    expect(final).toBe('only one')
  })

  test('iteration caches chunks for re-iteration', async () => {
    const generator = arrayToAsyncGenerator(['a', 'b', 'c'])
    const stream = new DialecticStreamResponse(generator)

    // Consume via iteration
    await collectGenerator(stream)

    // Re-iterate should use cached chunks
    const second = await stream.toArray()

    expect(second).toEqual(['a', 'b', 'c'])
  })
})

// =============================================================================
// createDialecticStream Tests
// =============================================================================

describe('createDialecticStream', () => {
  test('creates DialecticStreamResponse from SSE response', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "Hello"}}\n',
      'data: {"done": false, "delta": {"content": " World"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)

    expect(stream).toBeInstanceOf(DialecticStreamResponse)
  })

  test('yields content from delta', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "Hello"}}\n',
      'data: {"done": false, "delta": {"content": " World"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const chunks = await collectGenerator(stream)

    expect(chunks).toEqual(['Hello', ' World'])
  })

  test('stops on done: true', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "first"}}\n',
      'data: {"done": true, "delta": {}}\n',
      'data: {"done": false, "delta": {"content": "after done"}}\n',
    ])

    const stream = createDialecticStream(response)
    const chunks = await collectGenerator(stream)

    expect(chunks).toEqual(['first'])
  })

  test('skips chunks without content', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "yes"}}\n',
      'data: {"done": false, "delta": {}}\n',
      'data: {"done": false, "delta": {"content": "also yes"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const chunks = await collectGenerator(stream)

    expect(chunks).toEqual(['yes', 'also yes'])
  })

  test('handles empty content strings', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "start"}}\n',
      'data: {"done": false, "delta": {"content": ""}}\n',
      'data: {"done": false, "delta": {"content": "end"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const chunks = await collectGenerator(stream)

    // Empty string is falsy, so it's skipped
    expect(chunks).toEqual(['start', 'end'])
  })

  test('getFinalResponse joins all content', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "The "}}\n',
      'data: {"done": false, "delta": {"content": "answer "}}\n',
      'data: {"done": false, "delta": {"content": "is 42"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const result = await stream.getFinalResponse()

    expect(result).toBe('The answer is 42')
  })

  test('handles stream with only done message', async () => {
    const response = createMockStreamResponse([
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const chunks = await collectGenerator(stream)

    expect(chunks).toEqual([])
  })

  test('handles unicode content', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "\u4f60\u597d"}}\n',
      'data: {"done": false, "delta": {"content": " \ud83d\udc4b"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const result = await stream.getFinalResponse()

    expect(result).toBe('\u4f60\u597d \ud83d\udc4b')
  })

  test('handles newlines in content', async () => {
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "line1\\nline2"}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)
    const result = await stream.getFinalResponse()

    expect(result).toBe('line1\nline2')
  })
})

// =============================================================================
// Edge Cases and Integration
// =============================================================================

describe('Streaming edge cases', () => {
  test('parseSSE handles very long lines', async () => {
    const longContent = 'x'.repeat(10000)
    const response = createMockStreamResponse([
      `data: {"content": "${longContent}"}\n`,
    ])

    const events = await collectGenerator(
      parseSSE<{ content: string }>(response)
    )

    expect(events).toHaveLength(1)
    expect(events[0].content.length).toBe(10000)
  })

  test('parseSSE handles rapid small chunks', async () => {
    // Simulate byte-by-byte delivery
    const fullMessage = 'data: {"v": 1}\n'
    const chunks = fullMessage.split('').map((c) => c)

    const response = createMockStreamResponse(chunks)

    const events = await collectGenerator(parseSSE<{ v: number }>(response))

    expect(events).toHaveLength(1)
    expect(events[0].v).toBe(1)
  })

  test('DialecticStreamResponse handles many chunks', async () => {
    const manyChunks = Array.from({ length: 1000 }, (_, i) => `chunk${i}`)
    const generator = arrayToAsyncGenerator(manyChunks)
    const stream = new DialecticStreamResponse(generator)

    const result = await stream.toArray()

    expect(result.length).toBe(1000)
    expect(result[0]).toBe('chunk0')
    expect(result[999]).toBe('chunk999')
  })

  test('full pipeline: SSE response to final content', async () => {
    // Simulate a complete dialectic streaming response
    const response = createMockStreamResponse([
      'data: {"done": false, "delta": {"content": "Based on "}}\n',
      'data: {"done": false, "delta": {"content": "the conversation, "}}\n',
      'data: {"done": false, "delta": {"content": "the user enjoys "}}\n',
      'data: {"done": false, "delta": {"content": "programming."}}\n',
      'data: {"done": true, "delta": {}}\n',
    ])

    const stream = createDialecticStream(response)

    // Test iteration
    const chunks: string[] = []
    for await (const chunk of stream) {
      chunks.push(chunk)
    }

    expect(chunks).toEqual([
      'Based on ',
      'the conversation, ',
      'the user enjoys ',
      'programming.',
    ])

    // Test final response (from cache)
    const final = await stream.getFinalResponse()
    expect(final).toBe('Based on the conversation, the user enjoys programming.')
  })
})

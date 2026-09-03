/**
 * Evidence Unit Tests
 *
 * Evidence rides on the terminal chunk of a dialectic stream, which the
 * stream reader would otherwise discard. These tests drive mocked SSE bodies
 * so they cover that without a server.
 */

import { describe, test, expect } from 'bun:test'
import { createDialecticStream } from '../src/http/streaming'
import type { Evidence } from '../src/types/api'

const EVIDENCE: Evidence = {
  conclusions: [
    {
      id: 'doc-sentinel',
      level: 'deductive',
      content: 'User drinks coffee in the morning',
      created_at: '2026-01-01T00:00:00Z',
      session_id: 'session-1',
      source_ids: ['doc-a', 'doc-b'],
    },
  ],
  messages: [
    {
      id: 'msg-sentinel',
      session_id: 'session-1',
      peer_id: 'alice',
      created_at: '2026-01-01T00:00:00Z',
    },
  ],
  tool_calls: [{ tool_name: 'search_memory', tool_input: { query: 'coffee' } }],
  reasoning_trace_id: null,
}

function mockSSEResponse(lines: string[]): Response {
  const encoder = new TextEncoder()
  let index = 0
  const stream = new ReadableStream({
    pull(controller) {
      if (index < lines.length) {
        controller.enqueue(encoder.encode(lines[index]))
        index++
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

function contentFrame(content: string): string {
  return `data: ${JSON.stringify({ delta: { content }, done: false })}\n\n`
}

function doneFrame(evidence?: Evidence): string {
  return `data: ${JSON.stringify({ done: true, ...(evidence && { evidence })})}\n\n`
}

describe('streamed evidence', () => {
  test('is captured off the terminal chunk', async () => {
    const stream = createDialecticStream(
      mockSSEResponse([
        contentFrame('The user '),
        contentFrame('drinks coffee.'),
        doneFrame(EVIDENCE),
      ])
    )

    const chunks: string[] = []
    for await (const chunk of stream) {
      chunks.push(chunk)
    }

    expect(chunks.join('')).toBe('The user drinks coffee.')
    expect(stream.evidence).toEqual(EVIDENCE)
  })

  test('survives being split across network chunks', async () => {
    const frame = doneFrame(EVIDENCE)
    const midpoint = Math.floor(frame.length / 2)
    const stream = createDialecticStream(
      mockSSEResponse([
        contentFrame('Answer.'),
        frame.slice(0, midpoint),
        frame.slice(midpoint),
      ])
    )

    await stream.getFinalResponse()

    expect(stream.evidence?.conclusions[0]?.id).toBe('doc-sentinel')
  })

  test('is null until the stream has been consumed', async () => {
    const stream = createDialecticStream(
      mockSSEResponse([
        contentFrame('Answer.'),
        doneFrame(EVIDENCE),
      ])
    )

    // Evidence cannot be known before the answer is complete.
    expect(stream.evidence).toBeNull()

    await stream.getFinalResponse()

    expect(stream.evidence).not.toBeNull()
  })

  test('is null when the request did not ask for it', async () => {
    const stream = createDialecticStream(
      mockSSEResponse([
        contentFrame('Answer.'),
        doneFrame(),
      ])
    )

    await stream.getFinalResponse()

    expect(stream.evidence).toBeNull()
  })

  test('does not disturb the content chunks', async () => {
    const withEvidence = createDialecticStream(
      mockSSEResponse([
        contentFrame('a'),
        contentFrame('b'),
        doneFrame(EVIDENCE),
      ])
    )
    const withoutEvidence = createDialecticStream(
      mockSSEResponse([
        contentFrame('a'),
        contentFrame('b'),
        doneFrame(),
      ])
    )

    expect(await withEvidence.toArray()).toEqual(
      await withoutEvidence.toArray()
    )
  })
})

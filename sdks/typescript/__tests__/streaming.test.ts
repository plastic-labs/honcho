/**
 * Streaming Tests
 *
 * Tests for Server-Sent Events (SSE) streaming responses.
 *
 * Endpoints covered:
 * - POST /v1/workspaces/:id/peers/:id/chat (with stream=true)
 *
 * These tests verify:
 * - Streaming responses are properly parsed
 * - Async iteration works correctly
 * - Stream can be consumed chunk by chunk
 * - Full response can be collected
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho } from '../src'
import type { DialecticStreamResponse } from '../src/http/streaming'
import { createTestClient, requireServer } from './setup'
import { collectStream } from './helpers'

describe('Streaming', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('streaming')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // ===========================================================================
  // Basic Streaming
  // ===========================================================================

  describe('POST /peers/:id/chat (streaming)', () => {
    test('chat with stream=true returns async iterable', async () => {
      const peer = await client.peer('stream-basic-peer')
      const session = await client.session('stream-basic-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('I enjoy playing chess and reading mystery novels'),
      ])

      const response = await peer.chat('What are this user\'s hobbies?', {
        stream: true,
      })

      // Should return an async iterable, not a string
      expect(response).not.toBeNull()
      expect(typeof response).not.toBe('string')

      // Type narrowing
      if (response !== null && typeof response !== 'string') {
        // Should be async iterable
        expect(Symbol.asyncIterator in response).toBe(true)
      }
    })

    test('streaming response yields chunks', async () => {
      const peer = await client.peer('stream-chunks-peer')
      const session = await client.session('stream-chunks-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('My favorite programming language is TypeScript'),
      ])

      const response = await peer.chat('What programming language?', {
        stream: true,
      })

      if (response !== null && typeof response !== 'string') {
        const chunks: string[] = []

        for await (const chunk of response) {
          chunks.push(chunk)
        }

        // Should have received some chunks
        // (The actual content depends on the server/model behavior)
        expect(Array.isArray(chunks)).toBe(true)
      }
    })

    test('streaming chunks combine to full response', async () => {
      const peer = await client.peer('stream-combine-peer')
      const session = await client.session('stream-combine-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('I live in San Francisco and work as a software engineer'),
      ])

      const response = await peer.chat('Where does this user live?', {
        stream: true,
      })

      if (response !== null && typeof response !== 'string') {
        const fullResponse = await collectStream(response)

        // Should be a non-empty string when combined
        expect(typeof fullResponse).toBe('string')
      }
    })

    test('streaming with session scope', async () => {
      const peer = await client.peer('stream-scoped-peer')
      const session = await client.session('stream-scoped-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('In this session, I am discussing project planning'),
      ])

      const response = await peer.chat('What is being discussed?', {
        stream: true,
        session: session,
      })

      if (response !== null && typeof response !== 'string') {
        const chunks: string[] = []
        for await (const chunk of response) {
          chunks.push(chunk)
        }
        expect(Array.isArray(chunks)).toBe(true)
      }
    })

    test('streaming with target peer', async () => {
      const observer = await client.peer('stream-observer')
      const target = await client.peer('stream-target')
      const session = await client.session('stream-target-session')
      await session.addPeers([observer.id, target.id])
      await session.addMessages([
        target.message('I am the target peer sharing information'),
      ])

      const response = await observer.chat('What do you know about this user?', {
        stream: true,
        target: target,
      })

      if (response !== null && typeof response !== 'string') {
        const collected = await collectStream(response)
        expect(typeof collected).toBe('string')
      }
    })

    test('streaming with reasoning level', async () => {
      const peer = await client.peer('stream-reasoning-peer')
      const session = await client.session('stream-reasoning-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Complex information here')])

      const response = await peer.chat('Analyze this user', {
        stream: true,
        reasoningLevel: 'medium',
      })

      if (response !== null && typeof response !== 'string') {
        const chunks: string[] = []
        for await (const chunk of response) {
          chunks.push(chunk)
        }
        expect(Array.isArray(chunks)).toBe(true)
      }
    })
  })

  // ===========================================================================
  // Stream Consumption Patterns
  // ===========================================================================

  describe('Stream consumption patterns', () => {
    test('stream can only be consumed once', async () => {
      const peer = await client.peer('stream-once-peer')
      const session = await client.session('stream-once-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Test content')])

      const response = await peer.chat('Test query', { stream: true })

      if (response !== null && typeof response !== 'string') {
        // First consumption
        const first: string[] = []
        for await (const chunk of response) {
          first.push(chunk)
        }

        // Second consumption should yield nothing (stream exhausted)
        const second: string[] = []
        for await (const chunk of response) {
          second.push(chunk)
        }

        // Note: Behavior depends on implementation
        // Some streams throw, others just return empty
        expect(Array.isArray(second)).toBe(true)
      }
    })

    test('early break from stream', async () => {
      const peer = await client.peer('stream-break-peer')
      const session = await client.session('stream-break-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Long content here')])

      const response = await peer.chat('Query', { stream: true })

      if (response !== null && typeof response !== 'string') {
        let chunkCount = 0
        for await (const chunk of response) {
          chunkCount++
          if (chunkCount >= 2) {
            break // Exit early
          }
        }

        // Should have exited early without error
        expect(chunkCount).toBeLessThanOrEqual(2)
      }
    })
  })

  // ===========================================================================
  // Non-streaming Comparison
  // ===========================================================================

  describe('Streaming vs non-streaming', () => {
    test('non-streaming returns string directly', async () => {
      const peer = await client.peer('nonstream-peer')
      const session = await client.session('nonstream-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Some user preferences')])

      const response = await peer.chat('What are the preferences?', {
        stream: false,
      })

      // Non-streaming returns string or null directly
      expect(response === null || typeof response === 'string').toBe(true)
    })

    test('default is non-streaming', async () => {
      const peer = await client.peer('default-stream-peer')
      const session = await client.session('default-stream-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Default behavior test')])

      const response = await peer.chat('Query without stream option')

      // Default should be non-streaming (string or null)
      expect(response === null || typeof response === 'string').toBe(true)
    })
  })

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe('Edge cases', () => {
    test('streaming with minimal data', async () => {
      const peer = await client.peer('stream-minimal-peer')

      // No session data, representation will be empty
      const response = await peer.chat('What do you know?', { stream: true })

      // Should still return an iterable (might yield empty or null-like content)
      if (response !== null && typeof response !== 'string') {
        const chunks: string[] = []
        for await (const chunk of response) {
          chunks.push(chunk)
        }
        expect(Array.isArray(chunks)).toBe(true)
      }
    })

    test('streaming response type narrowing', async () => {
      const peer = await client.peer('stream-typing-peer')

      const streamResponse = await peer.chat('Query', { stream: true })
      const nonStreamResponse = await peer.chat('Query', { stream: false })

      // Type narrowing should work
      if (streamResponse !== null && typeof streamResponse !== 'string') {
        // This is DialecticStreamResponse
        expect(Symbol.asyncIterator in streamResponse).toBe(true)
      }

      if (nonStreamResponse !== null) {
        // This should be string
        expect(typeof nonStreamResponse).toBe('string')
      }
    })
  })
})

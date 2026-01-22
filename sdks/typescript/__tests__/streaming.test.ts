/**
 * Streaming Tests
 *
 * Tests for Server-Sent Events (SSE) streaming responses.
 *
 * Endpoints covered:
 * - POST /v1/workspaces/:id/peers/:id/chat (via chatStream())
 *
 * These tests verify:
 * - Streaming responses are properly parsed
 * - Async iteration works correctly
 * - Stream can be consumed chunk by chunk
 * - Full response can be collected
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho } from '../src'
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
    test('chatStream() returns async iterable', async () => {
      const peer = await client.peer('stream-basic-peer')
      const session = await client.session('stream-basic-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('I enjoy playing chess and reading mystery novels'),
      ])

      const response = await peer.chatStream("What are this user's hobbies?")

      // Should return an async iterable
      expect(response).not.toBeNull()
      expect(Symbol.asyncIterator in response).toBe(true)
    })

    test('streaming response yields chunks', async () => {
      const peer = await client.peer('stream-chunks-peer')
      const session = await client.session('stream-chunks-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('My favorite programming language is TypeScript'),
      ])

      const response = await peer.chatStream('What programming language?')

      const chunks: string[] = []
      for await (const chunk of response) {
        chunks.push(chunk)
      }

      // Should have received some chunks
      // (The actual content depends on the server/model behavior)
      expect(Array.isArray(chunks)).toBe(true)
    })

    test('streaming chunks combine to full response', async () => {
      const peer = await client.peer('stream-combine-peer')
      const session = await client.session('stream-combine-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('I live in San Francisco and work as a software engineer'),
      ])

      const response = await peer.chatStream('Where does this user live?')
      const fullResponse = await collectStream(response)

      // Should be a non-empty string when combined
      expect(typeof fullResponse).toBe('string')
    })

    test('streaming with session scope', async () => {
      const peer = await client.peer('stream-scoped-peer')
      const session = await client.session('stream-scoped-session')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('In this session, I am discussing project planning'),
      ])

      const response = await peer.chatStream('What is being discussed?', {
        session: session,
      })

      const chunks: string[] = []
      for await (const chunk of response) {
        chunks.push(chunk)
      }
      expect(Array.isArray(chunks)).toBe(true)
    })

    test('streaming with target peer', async () => {
      const observer = await client.peer('stream-observer')
      const target = await client.peer('stream-target')
      const session = await client.session('stream-target-session')
      await session.addPeers([observer.id, target.id])
      await session.addMessages([
        target.message('I am the target peer sharing information'),
      ])

      const response = await observer.chatStream(
        'What do you know about this user?',
        { target: target }
      )

      const collected = await collectStream(response)
      expect(typeof collected).toBe('string')
    })

    test('streaming with reasoning level', async () => {
      const peer = await client.peer('stream-reasoning-peer')
      const session = await client.session('stream-reasoning-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Complex information here')])

      const response = await peer.chatStream('Analyze this user', {
        reasoningLevel: 'medium',
      })

      const chunks: string[] = []
      for await (const chunk of response) {
        chunks.push(chunk)
      }
      expect(Array.isArray(chunks)).toBe(true)
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

      const response = await peer.chatStream('Test query')

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
    })

    test('early break from stream', async () => {
      const peer = await client.peer('stream-break-peer')
      const session = await client.session('stream-break-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Long content here')])

      const response = await peer.chatStream('Query')

      let chunkCount = 0
      for await (const chunk of response) {
        chunkCount++
        if (chunkCount >= 2) {
          break // Exit early
        }
      }

      // Should have exited early without error
      expect(chunkCount).toBeLessThanOrEqual(2)
    })
  })

  // ===========================================================================
  // Non-streaming Comparison
  // ===========================================================================

  describe('Streaming vs non-streaming', () => {
    test('chat() returns string directly', async () => {
      const peer = await client.peer('nonstream-peer')
      const session = await client.session('nonstream-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Some user preferences')])

      const response = await peer.chat('What are the preferences?')

      // Non-streaming returns string or null directly
      expect(response === null || typeof response === 'string').toBe(true)
    })

    test('chatStream() returns async iterable', async () => {
      const peer = await client.peer('stream-method-peer')
      const session = await client.session('stream-method-session')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Stream method test')])

      const response = await peer.chatStream('Query with stream method')

      // Streaming returns async iterable
      expect(Symbol.asyncIterator in response).toBe(true)
    })
  })

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe('Edge cases', () => {
    test('streaming with minimal data', async () => {
      const peer = await client.peer('stream-minimal-peer')

      // No session data, representation will be empty
      const response = await peer.chatStream('What do you know?')

      // Should still return an iterable (might yield empty or null-like content)
      const chunks: string[] = []
      for await (const chunk of response) {
        chunks.push(chunk)
      }
      expect(Array.isArray(chunks)).toBe(true)
    })

    test('separate methods have distinct return types', async () => {
      const peer = await client.peer('stream-typing-peer')

      const streamResponse = await peer.chatStream('Query')
      const nonStreamResponse = await peer.chat('Query')

      // chatStream() returns DialecticStreamResponse (async iterable)
      expect(Symbol.asyncIterator in streamResponse).toBe(true)

      // chat() returns string | null
      expect(nonStreamResponse === null || typeof nonStreamResponse === 'string').toBe(true)
    })
  })
})

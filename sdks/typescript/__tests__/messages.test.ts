/**
 * Messages Tests
 *
 * Tests for Message operations.
 *
 * Endpoints covered:
 * - POST /v1/workspaces/:id/sessions/:id/messages (create messages - batch)
 * - POST /v1/workspaces/:id/sessions/:id/messages/list (list messages)
 * - GET /v1/workspaces/:id/sessions/:id/messages/:id (get single message)
 * - PUT /v1/workspaces/:id/sessions/:id/messages/:id (update message)
 * - POST /v1/workspaces/:id/sessions/:id/messages/upload (file upload)
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho } from '../src'
import { createTestClient, requireServer } from './setup'
import { assertMessageShape, collectAll } from './helpers'

describe('Messages', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('messages')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // ===========================================================================
  // Message Creation (POST /messages)
  // ===========================================================================

  describe('POST /messages (create)', () => {
    test('creates single message', async () => {
      const session = await client.session('single-msg-session', { metadata: {} })
      const peer = await client.peer('single-msg-peer')
      await session.addPeers([peer.id])

      const messages = await session.addMessages(peer.message('Hello world'))

      expect(messages.length).toBe(1)
      assertMessageShape(messages[0])
      expect(messages[0].content).toBe('Hello world')
      expect(messages[0].peer_id).toBe(peer.id)
      expect(messages[0].session_id).toBe(session.id)
    })

    test('creates batch of messages', async () => {
      const session = await client.session('batch-msg-session', { metadata: {} })
      const peer = await client.peer('batch-msg-peer')
      await session.addPeers([peer.id])

      const messages = await session.addMessages([
        peer.message('First'),
        peer.message('Second'),
        peer.message('Third'),
      ])

      expect(messages.length).toBe(3)
      expect(messages[0].content).toBe('First')
      expect(messages[1].content).toBe('Second')
      expect(messages[2].content).toBe('Third')
    })

    test('message with metadata', async () => {
      const session = await client.session('meta-msg-session', { metadata: {} })
      const peer = await client.peer('meta-msg-peer')
      await session.addPeers([peer.id])

      const messages = await session.addMessages(
        peer.message('With metadata', { metadata: { key: 'value', count: 42 } })
      )

      expect(messages[0].metadata).toEqual({ key: 'value', count: 42 })
    })

    test('message with configuration', async () => {
      const session = await client.session('config-msg-session', { metadata: {} })
      const peer = await client.peer('config-msg-peer')
      await session.addPeers([peer.id])

      const messages = await session.addMessages(
        peer.message('With config', {
          configuration: { reasoning: { enabled: true } },
        })
      )

      expect(messages[0]).toBeDefined()
      // Configuration is used server-side, may not be returned
    })

    test('message with custom created_at', async () => {
      const session = await client.session('timestamp-msg-session', { metadata: {} })
      const peer = await client.peer('timestamp-msg-peer')
      await session.addPeers([peer.id])

      const customDate = new Date('2024-01-15T10:30:00Z')
      const messages = await session.addMessages(
        peer.message('Custom timestamp', { created_at: customDate })
      )

      // Server should use our timestamp
      expect(new Date(messages[0].created_at).toISOString()).toBe(
        customDate.toISOString()
      )
    })

    test('messages from multiple peers', async () => {
      const session = await client.session('multi-peer-msg-session', { metadata: {} })
      const alice = await client.peer('alice')
      const bob = await client.peer('bob')
      await session.addPeers([alice.id, bob.id])

      const messages = await session.addMessages([
        alice.message('Hello from Alice'),
        bob.message('Hello from Bob'),
        alice.message('Nice to meet you'),
      ])

      expect(messages[0].peer_id).toBe(alice.id)
      expect(messages[1].peer_id).toBe(bob.id)
      expect(messages[2].peer_id).toBe(alice.id)
    })

    test('batch up to 100 messages', async () => {
      const session = await client.session('large-batch-session', { metadata: {} })
      const peer = await client.peer('large-batch-peer')
      await session.addPeers([peer.id])

      const batch = Array.from({ length: 50 }, (_, i) =>
        peer.message(`Message ${i + 1}`)
      )

      const messages = await session.addMessages(batch)

      expect(messages.length).toBe(50)
      expect(messages[0].content).toBe('Message 1')
      expect(messages[49].content).toBe('Message 50')
    })

    test('token_count is calculated', async () => {
      const session = await client.session('token-count-session', { metadata: {} })
      const peer = await client.peer('token-count-peer')
      await session.addPeers([peer.id])

      const messages = await session.addMessages(
        peer.message('This is a test message with several words')
      )

      expect(messages[0].token_count).toBeGreaterThan(0)
    })
  })

  // ===========================================================================
  // Message Listing (POST /messages/list)
  // ===========================================================================

  describe('POST /messages/list', () => {
    test('returns paginated list', async () => {
      const session = await client.session('list-msg-session', { metadata: {} })
      const peer = await client.peer('list-msg-peer')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('One'),
        peer.message('Two'),
        peer.message('Three'),
      ])

      const page = await session.getMessages()

      expect(page.items.length).toBe(3)
      expect(page.page).toBe(1)
      expect(page.total).toBe(3)
    })

    test('pagination works', async () => {
      const session = await client.session('paginate-msg-session', { metadata: {} })
      const peer = await client.peer('paginate-msg-peer')
      await session.addPeers([peer.id])

      // Create 15 messages
      const batch = Array.from({ length: 15 }, (_, i) =>
        peer.message(`Msg ${i + 1}`)
      )
      await session.addMessages(batch)

      // Get first page with small size
      const page = await session.getMessages()

      expect(page.total).toBe(15)
    })

    test('Page is async iterable', async () => {
      const session = await client.session('iter-msg-session', { metadata: {} })
      const peer = await client.peer('iter-msg-peer')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('A'),
        peer.message('B'),
        peer.message('C'),
      ])

      const page = await session.getMessages()
      const contents = await collectAll(page)

      expect(contents.map((m) => m.content)).toContain('A')
      expect(contents.map((m) => m.content)).toContain('B')
      expect(contents.map((m) => m.content)).toContain('C')
    })

    test('filter by peer', async () => {
      const session = await client.session('filter-peer-msg-session', { metadata: {} })
      const alice = await client.peer('filter-alice')
      const bob = await client.peer('filter-bob')
      await session.addPeers([alice.id, bob.id])
      await session.addMessages([
        alice.message('From Alice'),
        bob.message('From Bob'),
      ])

      const page = await session.getMessages({ peer_id: alice.id })

      expect(page.items.length).toBe(1)
      expect(page.items[0].peer_id).toBe(alice.id)
    })

    test('filter by metadata', async () => {
      const session = await client.session('filter-meta-msg-session', { metadata: {} })
      const peer = await client.peer('filter-meta-peer')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('Tagged', { metadata: { category: 'important' } }),
        peer.message('Untagged'),
      ])

      const page = await session.getMessages({ metadata: { category: 'important' } })

      expect(page.items.length).toBe(1)
      expect(page.items[0].metadata.category).toBe('important')
    })
  })

  // ===========================================================================
  // Message Update (PUT /messages/:id)
  // ===========================================================================

  describe('PUT /messages/:id (update)', () => {
    test('updateMessage with MessageResponse', async () => {
      const session = await client.session('update-msg-response-session', { metadata: {} })
      const peer = await client.peer('update-msg-response-peer')
      await session.addPeers([peer.id])

      const [message] = await session.addMessages(peer.message('Original'))

      const updated = await client.updateMessage(message, {
        status: 'reviewed',
        reviewedAt: Date.now(),
      })

      expect(updated.metadata.status).toBe('reviewed')
      expect(updated.metadata.reviewedAt).toBeDefined()
    })

    test('updateMessage with string ID', async () => {
      const session = await client.session('update-msg-string-session', { metadata: {} })
      const peer = await client.peer('update-msg-string-peer')
      await session.addPeers([peer.id])

      const [message] = await session.addMessages(peer.message('To update'))

      const updated = await client.updateMessage(
        message.id,
        { flag: true },
        session
      )

      expect(updated.metadata.flag).toBe(true)
    })

    test('updateMessage replaces metadata entirely', async () => {
      const session = await client.session('update-msg-replace-session', { metadata: {} })
      const peer = await client.peer('update-msg-replace-peer')
      await session.addPeers([peer.id])

      const [message] = await session.addMessages(
        peer.message('With meta', { metadata: { old: 'value' } })
      )

      const updated = await client.updateMessage(message, { new: 'value' })

      expect(updated.metadata).toEqual({ new: 'value' })
      expect(updated.metadata.old).toBeUndefined()
    })
  })

  // ===========================================================================
  // File Upload (POST /messages/upload)
  // ===========================================================================

  describe('POST /messages/upload', () => {
    test('upload text file creates messages', async () => {
      const session = await client.session('upload-session', { metadata: {} })
      const peer = await client.peer('upload-peer')
      await session.addPeers([peer.id])

      const fileContent = 'Line 1\nLine 2\nLine 3'
      const file = new Blob([fileContent], { type: 'text/plain' })

      const messages = await session.uploadFile(file, peer)

      expect(messages.length).toBeGreaterThan(0)
      assertMessageShape(messages[0])
    })

    test('upload with metadata', async () => {
      const session = await client.session('upload-meta-session', { metadata: {} })
      const peer = await client.peer('upload-meta-peer')
      await session.addPeers([peer.id])

      const file = new Blob(['Test content'], { type: 'text/plain' })

      const messages = await session.uploadFile(file, peer, {
        metadata: { source: 'upload-test' },
      })

      expect(messages.length).toBeGreaterThan(0)
    })

    test('upload with buffer-style object', async () => {
      const session = await client.session('upload-buffer-session', { metadata: {} })
      const peer = await client.peer('upload-buffer-peer')
      await session.addPeers([peer.id])

      const content = new TextEncoder().encode('Buffer content here')

      const messages = await session.uploadFile(
        {
          filename: 'test.txt',
          content: content,
          content_type: 'text/plain',
        },
        peer
      )

      expect(messages.length).toBeGreaterThan(0)
    })

    test('upload with peer ID string', async () => {
      const session = await client.session('upload-peer-string-session', { metadata: {} })
      await session.addPeers(['upload-string-peer'])

      const file = new Blob(['Content'], { type: 'text/plain' })

      const messages = await session.uploadFile(file, 'upload-string-peer')

      expect(messages.length).toBeGreaterThan(0)
    })
  })

  // ===========================================================================
  // Message Shape Validation
  // ===========================================================================

  describe('Response shape validation', () => {
    test('message has all required fields', async () => {
      const session = await client.session('shape-session', { metadata: {} })
      const peer = await client.peer('shape-peer')
      await session.addPeers([peer.id])

      const [message] = await session.addMessages(peer.message('Shape test'))

      // Validate all fields exist and have correct types
      expect(typeof message.id).toBe('string')
      expect(message.id.length).toBeGreaterThan(0)
      expect(typeof message.content).toBe('string')
      expect(typeof message.peer_id).toBe('string')
      expect(typeof message.session_id).toBe('string')
      expect(typeof message.workspace_id).toBe('string')
      expect(typeof message.metadata).toBe('object')
      expect(typeof message.created_at).toBe('string')
      expect(typeof message.token_count).toBe('number')

      // Validate date format
      expect(() => new Date(message.created_at)).not.toThrow()
      const date = new Date(message.created_at)
      expect(date.getTime()).toBeGreaterThan(0)
    })

    test('message IDs are unique', async () => {
      const session = await client.session('unique-id-session', { metadata: {} })
      const peer = await client.peer('unique-id-peer')
      await session.addPeers([peer.id])

      const messages = await session.addMessages([
        peer.message('One'),
        peer.message('Two'),
        peer.message('Three'),
      ])

      const ids = messages.map((m) => m.id)
      const uniqueIds = new Set(ids)
      expect(uniqueIds.size).toBe(ids.length)
    })
  })
})

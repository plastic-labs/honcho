/**
 * Peer Tests
 *
 * Comprehensive tests for Peer operations.
 *
 * Endpoints covered:
 * - POST /v3/workspaces/:workspaceId/peers (get-or-create peer)
 * - POST /v3/workspaces/:workspaceId/peers/list (list peers)
 * - PUT /v3/workspaces/:workspaceId/peers/:peerId (update peer)
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/sessions (list peer sessions)
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/chat (dialectic chat)
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/representation (get representation)
 * - GET /v3/workspaces/:workspaceId/peers/:peerId/card (get peer card)
 * - GET /v3/workspaces/:workspaceId/peers/:peerId/context (get peer context)
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/search (search peer messages)
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho, Peer } from '../src'
import { createTestClient, generateId, requireServer } from './setup'
import {
  assertMessageShape,
  assertPeerShape,
  collectStream,
  testMessage,
  testMetadata,
} from './helpers'

describe('Peer', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('peer')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // ===========================================================================
  // Peer Creation (POST /peers)
  // ===========================================================================

  describe('POST /peers (create/get)', () => {
    test('creates peer with just ID', async () => {
      const peer = await client.peer('simple-peer')

      expect(peer).toBeInstanceOf(Peer)
      expect(peer.id).toBe('simple-peer')
      expect(peer.workspaceId).toBe(client.workspaceId)
    })

    test('creates peer with metadata', async () => {
      const metadata = testMetadata({ name: 'Alice' })
      const peer = await client.peer('peer-with-meta', { metadata })

      expect(peer.id).toBe('peer-with-meta')
      expect(peer.metadata).toEqual(metadata)
    })

    test('creates peer with configuration', async () => {
      const config = { observeMe: false }
      const peer = await client.peer('peer-with-config', { configuration: config })

      expect(peer.id).toBe('peer-with-config')
      expect(peer.configuration).toEqual(config)
    })

    test('creates peer with both metadata and configuration', async () => {
      const metadata = { role: 'user' }
      const config = { observeMe: true }
      const peer = await client.peer('peer-with-both', { metadata, configuration: config })

      expect(peer.metadata).toEqual(metadata)
      expect(peer.configuration).toEqual(config)
    })

    test('get-or-create is idempotent', async () => {
      const peer1 = await client.peer('idempotent-peer', {
        metadata: { first: true },
      })
      const peer2 = await client.peer('idempotent-peer', {
        metadata: { second: true },
      })

      expect(peer1.id).toBe(peer2.id)
      // Second call overwrites metadata
      expect(peer2.metadata).toEqual({ second: true })
    })
  })

  // ===========================================================================
  // Peer Listing (POST /peers/list)
  // ===========================================================================

  describe('POST /peers/list', () => {
    test('peers returns Page with items', async () => {
      // Create some peers first
      await client.peer('list-peer-a', { metadata: {} })
      await client.peer('list-peer-b', { metadata: {} })

      const page = await client.peers()

      expect(page.items.length).toBeGreaterThanOrEqual(2)
      expect(page.page).toBe(1)
      expect(page.total).toBeGreaterThanOrEqual(2)
    })

    test('peers with filter narrows results', async () => {
      const uniqueTag = `tag-${Date.now()}`
      await client.peer('filtered-peer', {
        metadata: { uniqueTag },
      })

      const page = await client.peers({ metadata: { uniqueTag } })

      expect(page.items.length).toBe(1)
      expect(page.items[0].id).toBe('filtered-peer')
    })

    test('Page is async iterable', async () => {
      await client.peer('iter-peer-1', { metadata: {} })
      await client.peer('iter-peer-2', { metadata: {} })

      const page = await client.peers()
      const ids: string[] = []

      for await (const peer of page) {
        ids.push(peer.id)
      }

      expect(ids).toContain('iter-peer-1')
      expect(ids).toContain('iter-peer-2')
    })
  })

  // ===========================================================================
  // Peer Update (PUT /peers/:id)
  // ===========================================================================

  describe('PUT /peers/:id (update)', () => {
    test('setMetadata updates peer metadata', async () => {
      const peer = await client.peer('meta-update-peer')

      await peer.setMetadata({ updated: true, count: 42 })
      const metadata = await peer.getMetadata()

      expect(metadata).toEqual({ updated: true, count: 42 })
    })

    test('setConfiguration updates peer configuration', async () => {
      const peer = await client.peer('config-update-peer')

      await peer.setConfiguration({ observeMe: false })
      const config = await peer.getConfiguration()

      expect(config).toEqual({ observeMe: false })
    })

    test('refresh updates cached values', async () => {
      const peer = await client.peer('refresh-peer', {
        metadata: { initial: true },
      })

      // Modify via another reference
      const peer2 = await client.peer('refresh-peer')
      await peer2.setMetadata({ modified: true })

      // Original peer has stale cache
      expect(peer.metadata).toEqual({ initial: true })

      // After refresh, cache is updated
      await peer.refresh()
      expect(peer.metadata).toEqual({ modified: true })
    })
  })

  // ===========================================================================
  // Peer Sessions (POST /peers/:id/sessions/list)
  // ===========================================================================

  describe('POST /peers/:id/sessions/list', () => {
    test('sessions returns sessions peer is in', async () => {
      const peer = await client.peer('session-member-peer')
      const session = await client.session('peer-sessions-test', { metadata: {} })

      await session.addPeers([peer.id])

      const sessions = await peer.sessions()

      expect(sessions.items.length).toBeGreaterThanOrEqual(1)
      const sessionIds = sessions.items.map((s) => s.id)
      expect(sessionIds).toContain('peer-sessions-test')
    })

    test('sessions returns empty for peer in no sessions', async () => {
      const peer = await client.peer('lonely-peer')

      const sessions = await peer.sessions()

      // Peer exists but not in any sessions
      expect(Array.isArray(sessions.items)).toBe(true)
    })

    test('sessions with filters', async () => {
      const peer = await client.peer('filter-sessions-peer')
      const session = await client.session('filterable-session', {
        metadata: { category: 'special' },
      })
      await session.addPeers([peer.id])

      const sessions = await peer.sessions({ metadata: { category: 'special' } })

      expect(sessions.items.length).toBeGreaterThanOrEqual(1)
    })
  })

  // ===========================================================================
  // Message Creation Helper
  // ===========================================================================

  describe('message() helper', () => {
    test('creates message object with peer ID', () => {
      const peer = new Peer('msg-peer', 'workspace', {} as never)

      const msg = peer.message('Hello world')

      expect(msg.peerId).toBe('msg-peer')
      expect(msg.content).toBe('Hello world')
    })

    test('message with metadata', () => {
      const peer = new Peer('msg-peer', 'workspace', {} as never)

      const msg = peer.message('Hello', { metadata: { key: 'value' } })

      expect(msg.metadata).toEqual({ key: 'value' })
    })

    test('message with configuration', () => {
      const peer = new Peer('msg-peer', 'workspace', {} as never)

      const msg = peer.message('Hello', {
        configuration: { reasoning: { enabled: true } },
      })

      expect(msg.configuration).toEqual({ reasoning: { enabled: true } })
    })

    test('message with createdAt string', () => {
      const peer = new Peer('msg-peer', 'workspace', {} as never)
      const timestamp = '2024-01-15T10:30:00Z'

      const msg = peer.message('Hello', { createdAt: timestamp })

      expect(msg.createdAt).toBe(timestamp)
    })

    test('message with createdAt Date', () => {
      const peer = new Peer('msg-peer', 'workspace', {} as never)
      const date = new Date('2024-01-15T10:30:00Z')

      const msg = peer.message('Hello', { createdAt: date })

      expect(msg.createdAt).toBe(date.toISOString())
    })
  })

  // ===========================================================================
  // Search (POST /peers/:id/search)
  // ===========================================================================

  describe('POST /peers/:id/search', () => {
    test('search finds messages from this peer', async () => {
      const peer = await client.peer('search-author-peer')
      const session = await client.session('search-author-session', { metadata: {} })

      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('Unique content for searching xyz789'),
      ])

      const results = await peer.search('unique content searching')

      expect(Array.isArray(results)).toBe(true)
      // Results should be from this peer
      for (const msg of results) {
        expect(msg.peerId).toBe(peer.id)
      }
    })

    test('search with filters', async () => {
      const peer = await client.peer('search-filter-peer')
      const session = await client.session('search-filter-session', { metadata: {} })

      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('Searchable content abc'),
      ])

      const results = await peer.search('searchable', {
        filters: { session_id: session.id },
      })

      for (const msg of results) {
        expect(msg.sessionId).toBe(session.id)
      }
    })

    test('search with limit', async () => {
      const peer = await client.peer('search-limit-peer')

      const results = await peer.search('test', { limit: 3 })

      expect(results.length).toBeLessThanOrEqual(3)
    })
  })

  // ===========================================================================
  // Representation (POST /peers/:id/representation)
  // ===========================================================================

  describe('POST /peers/:id/representation', () => {
    test('representation returns string', async () => {
      const peer = await client.peer('repr-peer')
      const session = await client.session('repr-session', { metadata: {} })

      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('I love programming in TypeScript'),
        peer.message('My favorite color is blue'),
      ])

      const representation = await peer.representation()

      expect(typeof representation).toBe('string')
    })

    test('representation scoped to session', async () => {
      const peer = await client.peer('repr-session-peer')
      const session = await client.session('repr-session-scoped', { metadata: {} })

      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Session-specific content')])

      const representation = await peer.representation({ session })

      expect(typeof representation).toBe('string')
    })

    test('representation with target peer', async () => {
      const observer = await client.peer('repr-observer')
      const observed = await client.peer('repr-observed')
      const session = await client.session('repr-target-session', { metadata: {} })

      await session.addPeers([observer.id, observed.id])
      await session.addMessages([
        observed.message('I am being observed'),
      ])

      const representation = await observer.representation({
        target: observed,
      })

      expect(typeof representation).toBe('string')
    })

    test('representation with options', async () => {
      const peer = await client.peer('repr-options-peer')

      const representation = await peer.representation({
        searchQuery: 'preferences',
        searchTopK: 5,
        maxConclusions: 20,
      })

      expect(typeof representation).toBe('string')
    })
  })

  // ===========================================================================
  // Peer Card (POST /peers/:id/card)
  // ===========================================================================

  describe('POST /peers/:id/card', () => {
    test('card returns string array or null', async () => {
      const peer = await client.peer('card-peer')

      const card = await peer.card()

      // card() returns string[] | null
      expect(card === null || Array.isArray(card)).toBe(true)
    })

    test('card with target peer', async () => {
      const observer = await client.peer('card-observer')
      const observed = await client.peer('card-observed')

      const card = await observer.card(observed)

      // card() returns string[] | null
      expect(card === null || Array.isArray(card)).toBe(true)
    })

    test('card with target ID string', async () => {
      const peer = await client.peer('card-string-peer')

      const card = await peer.card('some-target-id')

      // card() returns string[] | null
      expect(card === null || Array.isArray(card)).toBe(true)
    })

    test('card throws on invalid target type', async () => {
      const peer = await client.peer('card-invalid-peer')

      // Zod throws on invalid type
      await expect(peer.card(123 as never)).rejects.toThrow()
    })

    test('card throws on empty target string', async () => {
      const peer = await client.peer('card-empty-peer')

      // Zod validation requires non-empty string
      await expect(peer.card('')).rejects.toThrow()
    })
  })

  // ===========================================================================
  // Set Peer Card (PUT /peers/:id/card)
  // ===========================================================================

  describe('PUT /peers/:id/card', () => {
    test('setCard sets and returns peer card', async () => {
      const peer = await client.peer('setcard-peer')

      const cardData = ['fact one', 'fact two']
      const result = await peer.setCard(cardData)

      expect(result).toEqual(cardData)

      // Verify with get
      const card = await peer.getCard()
      expect(card).toEqual(cardData)
    })

    test('setCard with target peer', async () => {
      const observer = await client.peer('setcard-observer')
      const observed = await client.peer('setcard-observed')

      const cardData = ['target likes TypeScript', 'target is clever']
      const result = await observer.setCard(cardData, observed)

      expect(result).toEqual(cardData)

      // Verify with get
      const card = await observer.getCard(observed)
      expect(card).toEqual(cardData)
    })

    test('setCard with target ID string', async () => {
      const peer = await client.peer('setcard-string-peer')

      const cardData = ['some fact']
      const result = await peer.setCard(cardData, 'setcard-string-target')

      expect(result).toEqual(cardData)
    })
  })

  // ===========================================================================
  // Peer Context (POST /peers/:id/context)
  // ===========================================================================

  describe('POST /peers/:id/context', () => {
    test('context returns representation and card', async () => {
      const peer = await client.peer('context-peer')

      const context = await peer.context()

      expect(context).toBeDefined()
      expect(context.peerId).toBe(peer.id)
      expect(context.targetId).toBe(peer.id) // Self-context
      expect('representation' in context).toBe(true)
      expect('peerCard' in context).toBe(true)
    })

    test('context with target peer', async () => {
      const observer = await client.peer('context-observer')
      const observed = await client.peer('context-observed')

      const context = await observer.context({ target: observed })

      expect(context.peerId).toBe(observer.id)
      expect(context.targetId).toBe(observed.id)
    })

    test('context with options', async () => {
      const peer = await client.peer('context-options-peer')

      const context = await peer.context({
        searchQuery: 'interests',
        searchTopK: 10,
        maxConclusions: 25,
      })

      expect(context).toBeDefined()
    })
  })

  // ===========================================================================
  // Chat / Dialectic (POST /peers/:id/chat)
  // ===========================================================================

  describe('POST /peers/:id/chat', () => {
    test('chat returns string response', async () => {
      const peer = await client.peer('chat-peer')
      const session = await client.session('chat-session', { metadata: {} })

      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('I enjoy hiking in the mountains'),
      ])

      const response = await peer.chat('What activities does this user enjoy?')

      // Response is string or null
      expect(response === null || typeof response === 'string').toBe(true)
    })

    test('chat with session scope', async () => {
      const peer = await client.peer('chat-session-peer')
      const session = await client.session('chat-scoped-session', { metadata: {} })

      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Session specific info')])

      const response = await peer.chat('What do you know?', {
        session: session,
      })

      expect(response === null || typeof response === 'string').toBe(true)
    })

    test('chat with target peer', async () => {
      const observer = await client.peer('chat-observer')
      const target = await client.peer('chat-target')

      const response = await observer.chat('What do you know about this user?', {
        target: target,
      })

      expect(response === null || typeof response === 'string').toBe(true)
    })

    test('chat with reasoning level', async () => {
      const peer = await client.peer('chat-reasoning-peer')

      const response = await peer.chat('Analyze this user', {
        reasoningLevel: 'high',
      })

      expect(response === null || typeof response === 'string').toBe(true)
    })

    // Streaming tests are in streaming.test.ts
  })

  // ===========================================================================
  // Conclusions Scope
  // ===========================================================================

  describe('Conclusion scope access', () => {
    test('conclusions property returns ConclusionScope for self', async () => {
      const peer = await client.peer('self-conclusions-peer')

      const scope = peer.conclusions

      expect(scope.observer).toBe(peer.id)
      expect(scope.observed).toBe(peer.id)
      expect(scope.workspaceId).toBe(client.workspaceId)
    })

    test('conclusionsOf returns ConclusionScope for target', async () => {
      const observer = await client.peer('obs-conclusions-peer')
      const target = await client.peer('target-conclusions-peer')

      const scope = observer.conclusionsOf(target)

      expect(scope.observer).toBe(observer.id)
      expect(scope.observed).toBe(target.id)
    })

    test('conclusionsOf with string ID', async () => {
      const peer = await client.peer('string-conclusions-peer')

      const scope = peer.conclusionsOf('some-target-id')

      expect(scope.observer).toBe(peer.id)
      expect(scope.observed).toBe('some-target-id')
    })
  })

  // ===========================================================================
  // String Representation
  // ===========================================================================

  describe('toString', () => {
    test('returns readable format', async () => {
      const peer = await client.peer('tostring-peer')

      const str = peer.toString()

      expect(str).toBe("Peer(id='tostring-peer')")
    })
  })
})

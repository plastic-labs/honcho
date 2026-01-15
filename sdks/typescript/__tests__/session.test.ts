/**
 * Session Tests
 *
 * Tests for Session operations.
 *
 * Endpoints covered:
 * - POST /v1/workspaces/:id/sessions (create/get session)
 * - POST /v1/workspaces/:id/sessions/list (list sessions)
 * - PUT /v1/workspaces/:id/sessions/:id (update session)
 * - DELETE /v1/workspaces/:id/sessions/:id (delete session)
 * - POST /v1/workspaces/:id/sessions/:id/clone (clone session)
 * - POST /v1/workspaces/:id/sessions/:id/peers/add (add peers)
 * - POST /v1/workspaces/:id/sessions/:id/peers/set (set peers)
 * - POST /v1/workspaces/:id/sessions/:id/peers/remove (remove peers)
 * - GET /v1/workspaces/:id/sessions/:id/peers (list peers)
 * - GET /v1/workspaces/:id/sessions/:id/peers/:id/config (get peer config)
 * - PUT /v1/workspaces/:id/sessions/:id/peers/:id/config (set peer config)
 * - POST /v1/workspaces/:id/sessions/:id/context (get context)
 * - GET /v1/workspaces/:id/sessions/:id/summaries (get summaries)
 * - POST /v1/workspaces/:id/sessions/:id/search (search)
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho, Session, SessionPeerConfig } from '../src'
import { createTestClient, generateId, requireServer } from './setup'
import { assertMessageShape, testMetadata } from './helpers'

describe('Session', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('session')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // ===========================================================================
  // Session Creation (POST /sessions)
  // ===========================================================================

  describe('POST /sessions (create/get)', () => {
    test('creates session with just ID', async () => {
      const session = await client.session('simple-session')

      expect(session).toBeInstanceOf(Session)
      expect(session.id).toBe('simple-session')
      expect(session.workspaceId).toBe(client.workspaceId)
    })

    test('creates session with metadata', async () => {
      const metadata = testMetadata({ topic: 'testing' })
      const session = await client.session('session-with-meta', { metadata })

      expect(session.metadata).toEqual(metadata)
    })

    test('creates session with configuration', async () => {
      const config = { summary: { enabled: true } }
      const session = await client.session('session-with-config', { config })

      expect(session.configuration).toEqual(config)
    })

    test('get-or-create is idempotent', async () => {
      const session1 = await client.session('idempotent-session', {
        metadata: { version: 1 },
      })
      const session2 = await client.session('idempotent-session', {
        metadata: { version: 2 },
      })

      expect(session1.id).toBe(session2.id)
      expect(session2.metadata).toEqual({ version: 2 })
    })
  })

  // ===========================================================================
  // Session Listing (POST /sessions/list)
  // ===========================================================================

  describe('POST /sessions/list', () => {
    test('getSessions returns paginated list', async () => {
      await client.session('list-session-a', { metadata: {} })
      await client.session('list-session-b', { metadata: {} })

      const page = await client.getSessions()

      expect(page.items.length).toBeGreaterThanOrEqual(2)
      const ids = page.items.map((s) => s.id)
      expect(ids).toContain('list-session-a')
      expect(ids).toContain('list-session-b')
    })

    test('getSessions with filter', async () => {
      const tag = `tag-${Date.now()}`
      await client.session('filtered-session', { metadata: { tag } })

      const page = await client.getSessions({ tag })

      expect(page.items.length).toBe(1)
      expect(page.items[0].id).toBe('filtered-session')
    })
  })

  // ===========================================================================
  // Session Update (PUT /sessions/:id)
  // ===========================================================================

  describe('PUT /sessions/:id (update)', () => {
    test('setMetadata updates session metadata', async () => {
      const session = await client.session('update-meta-session')

      await session.setMetadata({ updated: true, step: 2 })
      const metadata = await session.getMetadata()

      expect(metadata).toEqual({ updated: true, step: 2 })
    })

    test('setConfig updates session configuration', async () => {
      const session = await client.session('update-config-session')

      await session.setConfig({ summary: { enabled: false } })
      const config = await session.getConfig()

      expect(config).toEqual({ summary: { enabled: false } })
    })

    test('refresh updates cached values', async () => {
      const session = await client.session('refresh-session', {
        metadata: { initial: true },
      })

      // Modify via another reference
      const session2 = await client.session('refresh-session')
      await session2.setMetadata({ modified: true })

      // Original has stale cache
      expect(session.metadata).toEqual({ initial: true })

      // After refresh
      await session.refresh()
      expect(session.metadata).toEqual({ modified: true })
    })
  })

  // ===========================================================================
  // Session Deletion (DELETE /sessions/:id)
  // ===========================================================================

  describe('DELETE /sessions/:id', () => {
    test('delete removes session', async () => {
      const session = await client.session('delete-me-session', { metadata: {} })

      await session.delete()

      // Session should not appear in list
      const page = await client.getSessions()
      const ids = page.items.map((s) => s.id)
      expect(ids).not.toContain('delete-me-session')
    })
  })

  // ===========================================================================
  // Session Clone (POST /sessions/:id/clone)
  // ===========================================================================

  describe('POST /sessions/:id/clone', () => {
    test('clone creates copy of session', async () => {
      const original = await client.session('original-session', {
        metadata: { original: true },
      })
      const peer = await client.peer('clone-peer')
      await original.addPeers([peer.id])
      await original.addMessages([peer.message('Message 1')])

      const cloned = await original.clone()

      expect(cloned.id).not.toBe(original.id)
      // Cloned session should have same messages
      const messages = await cloned.getMessages()
      expect(messages.items.length).toBe(1)
    })

    test('clone up to specific message', async () => {
      const original = await client.session('clone-partial-session')
      const peer = await client.peer('clone-partial-peer')
      await original.addPeers([peer.id])

      const messages = await original.addMessages([
        peer.message('First'),
        peer.message('Second'),
        peer.message('Third'),
      ])

      // Clone up to second message
      const cloned = await original.clone(messages[1].id)

      const clonedMessages = await cloned.getMessages()
      expect(clonedMessages.items.length).toBe(2)
    })
  })

  // ===========================================================================
  // Peer Management
  // ===========================================================================

  describe('Peer management', () => {
    describe('POST /sessions/:id/peers/add', () => {
      test('addPeers with string array', async () => {
        const session = await client.session('add-peers-string')

        await session.addPeers(['peer-a', 'peer-b'])

        const peers = await session.getPeers()
        const ids = peers.map((p) => p.id)
        expect(ids).toContain('peer-a')
        expect(ids).toContain('peer-b')
      })

      test('addPeers with Peer objects', async () => {
        const session = await client.session('add-peers-objects')
        const peerA = await client.peer('obj-peer-a')
        const peerB = await client.peer('obj-peer-b')

        await session.addPeers([peerA, peerB])

        const peers = await session.getPeers()
        const ids = peers.map((p) => p.id)
        expect(ids).toContain('obj-peer-a')
        expect(ids).toContain('obj-peer-b')
      })

      test('addPeers with config tuples', async () => {
        const session = await client.session('add-peers-config')

        await session.addPeers([
          ['config-peer-a', { observe_me: true, observe_others: false }],
          ['config-peer-b', { observe_me: false }],
        ])

        const configA = await session.getPeerConfig('config-peer-a')
        expect(configA.observe_me).toBe(true)
        expect(configA.observe_others).toBe(false)
      })

      test('addPeers with single peer', async () => {
        const session = await client.session('add-single-peer')

        await session.addPeers('single-peer')

        const peers = await session.getPeers()
        expect(peers.map((p) => p.id)).toContain('single-peer')
      })
    })

    describe('POST /sessions/:id/peers/set', () => {
      test('setPeers replaces all peers', async () => {
        const session = await client.session('set-peers-session')
        await session.addPeers(['old-peer-a', 'old-peer-b'])

        await session.setPeers(['new-peer-a', 'new-peer-b'])

        const peers = await session.getPeers()
        const ids = peers.map((p) => p.id)
        expect(ids).toContain('new-peer-a')
        expect(ids).toContain('new-peer-b')
        expect(ids).not.toContain('old-peer-a')
      })
    })

    describe('POST /sessions/:id/peers/remove', () => {
      test('removePeers removes specified peers', async () => {
        const session = await client.session('remove-peers-session')
        await session.addPeers(['keep-peer', 'remove-peer'])

        await session.removePeers(['remove-peer'])

        const peers = await session.getPeers()
        const ids = peers.map((p) => p.id)
        expect(ids).toContain('keep-peer')
        expect(ids).not.toContain('remove-peer')
      })

      test('removePeers with Peer objects', async () => {
        const session = await client.session('remove-peer-objects')
        const peer = await client.peer('peer-to-remove')
        await session.addPeers([peer])

        await session.removePeers([peer])

        const peers = await session.getPeers()
        expect(peers.map((p) => p.id)).not.toContain('peer-to-remove')
      })
    })

    describe('GET /sessions/:id/peers', () => {
      test('getPeers returns Peer instances', async () => {
        const session = await client.session('get-peers-session')
        await session.addPeers(['list-peer-1', 'list-peer-2'])

        const peers = await session.getPeers()

        expect(peers.length).toBe(2)
        expect(peers[0].workspaceId).toBe(client.workspaceId)
      })
    })

    describe('GET/PUT /sessions/:id/peers/:id/config', () => {
      test('getPeerConfig returns config', async () => {
        const session = await client.session('get-peer-config-session')
        await session.addPeers([
          ['peer-with-config', { observe_me: true, observe_others: false }],
        ])

        const config = await session.getPeerConfig('peer-with-config')

        expect(config.observe_me).toBe(true)
        expect(config.observe_others).toBe(false)
      })

      test('setPeerConfig updates config', async () => {
        const session = await client.session('set-peer-config-session')
        await session.addPeers(['peer-update-config'])

        await session.setPeerConfig(
          'peer-update-config',
          new SessionPeerConfig(false, true)
        )

        const config = await session.getPeerConfig('peer-update-config')
        expect(config.observe_me).toBe(false)
        expect(config.observe_others).toBe(true)
      })

      test('setPeerConfig with Peer object', async () => {
        const session = await client.session('set-config-peer-obj')
        const peer = await client.peer('config-obj-peer')
        await session.addPeers([peer])

        await session.setPeerConfig(peer, new SessionPeerConfig(true))

        const config = await session.getPeerConfig(peer)
        expect(config.observe_me).toBe(true)
      })
    })
  })

  // ===========================================================================
  // Message Operations
  // ===========================================================================

  describe('Message operations', () => {
    describe('POST /sessions/:id/messages', () => {
      test('addMessages creates messages', async () => {
        const session = await client.session('add-messages-session')
        const peer = await client.peer('add-messages-peer')
        await session.addPeers([peer.id])

        const messages = await session.addMessages([
          peer.message('Hello'),
          peer.message('World'),
        ])

        expect(messages.length).toBe(2)
        assertMessageShape(messages[0])
        expect(messages[0].content).toBe('Hello')
      })

      test('addMessages with single message', async () => {
        const session = await client.session('single-message-session')
        const peer = await client.peer('single-message-peer')
        await session.addPeers([peer.id])

        const messages = await session.addMessages(peer.message('Single'))

        expect(messages.length).toBe(1)
        expect(messages[0].content).toBe('Single')
      })
    })

    describe('POST /sessions/:id/messages/list', () => {
      test('getMessages returns paginated list', async () => {
        const session = await client.session('list-messages-session')
        const peer = await client.peer('list-messages-peer')
        await session.addPeers([peer.id])
        await session.addMessages([
          peer.message('Msg 1'),
          peer.message('Msg 2'),
        ])

        const page = await session.getMessages()

        expect(page.items.length).toBe(2)
        assertMessageShape(page.items[0])
      })

      test('getMessages with filters', async () => {
        const session = await client.session('filter-messages-session')
        const peer = await client.peer('filter-messages-peer')
        await session.addPeers([peer.id])
        await session.addMessages([
          peer.message('Filter me', { metadata: { tag: 'special' } }),
          peer.message('Not this one'),
        ])

        const page = await session.getMessages({ tag: 'special' })

        expect(page.items.length).toBe(1)
        expect(page.items[0].metadata.tag).toBe('special')
      })
    })
  })

  // ===========================================================================
  // Context and Summaries
  // ===========================================================================

  describe('POST /sessions/:id/context', () => {
    test('getContext returns context object', async () => {
      const session = await client.session('context-session')
      const peer = await client.peer('context-peer')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Context content')])

      const context = await session.getContext()

      expect(context).toBeDefined()
      expect(context.sessionId).toBe(session.id)
      expect(Array.isArray(context.messages)).toBe(true)
    })

    test('getContext with options object', async () => {
      const session = await client.session('context-options-session')
      const peer = await client.peer('context-options-peer')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Some content')])

      const context = await session.getContext({
        summary: true,
        tokens: 1000,
        peerTarget: peer.id,
      })

      expect(context).toBeDefined()
    })

    test('getContext with positional args (legacy)', async () => {
      const session = await client.session('context-positional-session')
      const peer = await client.peer('context-positional-peer')
      await session.addPeers([peer.id])

      const context = await session.getContext(true, 500)

      expect(context).toBeDefined()
    })
  })

  describe('GET /sessions/:id/summaries', () => {
    test('getSummaries returns summary object', async () => {
      const session = await client.session('summaries-session')

      const summaries = await session.getSummaries()

      expect(summaries).toBeDefined()
      expect(summaries.sessionId).toBe(session.id)
      // Summaries may be null for sessions without enough messages
      expect('shortSummary' in summaries).toBe(true)
      expect('longSummary' in summaries).toBe(true)
    })
  })

  // ===========================================================================
  // Search
  // ===========================================================================

  describe('POST /sessions/:id/search', () => {
    test('search returns matching messages', async () => {
      const session = await client.session('search-session')
      const peer = await client.peer('search-peer')
      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('The quick brown fox'),
        peer.message('Jumped over the lazy dog'),
      ])

      const results = await session.search('quick brown')

      expect(Array.isArray(results)).toBe(true)
      // Results are from this session
      for (const msg of results) {
        expect(msg.session_id).toBe(session.id)
      }
    })

    test('search with limit', async () => {
      const session = await client.session('search-limit-session')

      const results = await session.search('test', { limit: 5 })

      expect(results.length).toBeLessThanOrEqual(5)
    })
  })

  // ===========================================================================
  // Queue Status
  // ===========================================================================

  describe('Queue status', () => {
    test('getQueueStatus returns status for session', async () => {
      const session = await client.session('queue-status-session')

      const status = await session.getQueueStatus()

      expect(typeof status.totalWorkUnits).toBe('number')
      expect(typeof status.completedWorkUnits).toBe('number')
      expect(typeof status.pendingWorkUnits).toBe('number')
    })

    test('getQueueStatus with observer filter', async () => {
      const session = await client.session('queue-observer-session')
      const peer = await client.peer('queue-observer-peer')

      const status = await session.getQueueStatus({ observer: peer })

      expect(typeof status.totalWorkUnits).toBe('number')
    })
  })

  // ===========================================================================
  // Representation
  // ===========================================================================

  describe('POST /peers/:id/representation (session-scoped)', () => {
    test('getRepresentation returns string', async () => {
      const session = await client.session('repr-session')
      const peer = await client.peer('repr-peer')
      await session.addPeers([peer.id])
      await session.addMessages([peer.message('Learning TypeScript')])

      const representation = await session.getRepresentation(peer)

      expect(typeof representation).toBe('string')
    })

    test('getRepresentation with target', async () => {
      const session = await client.session('repr-target-session')
      const observer = await client.peer('repr-observer')
      const target = await client.peer('repr-target')
      await session.addPeers([observer.id, target.id])

      const representation = await session.getRepresentation(observer, target)

      expect(typeof representation).toBe('string')
    })
  })

  // ===========================================================================
  // String Representation
  // ===========================================================================

  describe('toString', () => {
    test('returns readable format', async () => {
      const session = await client.session('tostring-session')

      const str = session.toString()

      expect(str).toBe("Session(id='tostring-session')")
    })
  })
})

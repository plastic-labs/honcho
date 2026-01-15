/**
 * Client Tests
 *
 * Tests for workspace-level operations via the Honcho client.
 *
 * Endpoints covered:
 * - POST /v1/workspaces (create/get workspace)
 * - POST /v1/workspaces/list (list workspaces)
 * - PUT /v1/workspaces/:id (update workspace)
 * - DELETE /v1/workspaces/:id (delete workspace)
 * - POST /v1/workspaces/:id/search (search messages)
 * - GET /v1/workspaces/:id/queue/status (queue status)
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho } from '../src'
import {
  createTestClient,
  generateId,
  generateWorkspaceId,
  requireServer,
  TEST_CONFIG,
} from './setup'
import { testMetadata } from './helpers'

describe('Honcho Client', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('client')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // ===========================================================================
  // Workspace Creation and Configuration
  // ===========================================================================

  describe('POST /workspaces (create/get)', () => {
    test('client constructor creates workspace on first access', async () => {
      // Workspace was created in beforeAll via getMetadata()
      expect(client.workspaceId).toBeDefined()
      expect(client.workspaceId).toContain('test-client-')
    })

    test('getMetadata returns empty object for new workspace', async () => {
      const metadata = await client.getMetadata()
      expect(metadata).toEqual({})
    })

    test('getConfig returns empty object for new workspace', async () => {
      const config = await client.getConfig()
      expect(config).toEqual({})
    })
  })

  describe('PUT /workspaces/:id (update)', () => {
    test('setMetadata updates workspace metadata', async () => {
      const metadata = testMetadata({ custom: 'value' })
      await client.setMetadata(metadata)

      const fetched = await client.getMetadata()
      expect(fetched).toEqual(metadata)
    })

    test('setConfig updates workspace configuration', async () => {
      const config = {
        reasoning: { enabled: true },
      }
      await client.setConfig(config)

      const fetched = await client.getConfig()
      expect(fetched).toEqual(config)
    })

    test('cached metadata is updated after setMetadata', async () => {
      const metadata = testMetadata({ cached: true })
      await client.setMetadata(metadata)

      // Check cached value without API call
      expect(client.metadata).toEqual(metadata)
    })

    test('refresh updates both metadata and config', async () => {
      // Set values
      await client.setMetadata({ a: 1 })
      await client.setConfig({ b: 2 })

      // Create new client with same workspace (simulates stale cache)
      const freshClient = new Honcho({
        baseURL: TEST_CONFIG.baseURL,
        apiKey: TEST_CONFIG.apiKey,
        workspaceId: client.workspaceId,
      })

      // Before refresh, cache is empty
      expect(freshClient.metadata).toBeUndefined()
      expect(freshClient.configuration).toBeUndefined()

      // After refresh, cache is populated
      // Note: server may add default configuration values like reasoning.enabled
      await freshClient.refresh()
      expect(freshClient.metadata).toEqual({ a: 1 })
      expect(freshClient.configuration).toMatchObject({ b: 2 })
    })
  })

  // ===========================================================================
  // Workspace Listing
  // ===========================================================================

  describe('POST /workspaces/list', () => {
    test('getWorkspaces returns array of workspace IDs', async () => {
      const workspaces = await client.getWorkspaces()

      expect(Array.isArray(workspaces)).toBe(true)
      expect(workspaces).toContain(client.workspaceId)
    })

    test('getWorkspaces with filter narrows results', async () => {
      // Create a workspace with specific metadata
      const uniqueValue = `filter-test-${Date.now()}`
      const testClient = new Honcho({
        baseURL: TEST_CONFIG.baseURL,
        apiKey: TEST_CONFIG.apiKey,
        workspaceId: generateWorkspaceId('filter'),
      })

      try {
        await testClient.setMetadata({ filterKey: uniqueValue })

        // Filter should find this workspace
        const filtered = await client.getWorkspaces({
          metadata: { filterKey: uniqueValue },
        })

        expect(filtered).toContain(testClient.workspaceId)
      } finally {
        await testClient.deleteWorkspace(testClient.workspaceId)
      }
    })
  })

  // ===========================================================================
  // Workspace Deletion
  // ===========================================================================

  describe('DELETE /workspaces/:id', () => {
    test('deleteWorkspace removes workspace', async () => {
      // Create a workspace to delete
      const tempWorkspaceId = generateWorkspaceId('delete')
      const tempClient = new Honcho({
        baseURL: TEST_CONFIG.baseURL,
        apiKey: TEST_CONFIG.apiKey,
        workspaceId: tempWorkspaceId,
      })

      // Ensure it exists
      await tempClient.getMetadata()

      // Delete it (returns void)
      await client.deleteWorkspace(tempWorkspaceId)

      // Verify it's gone from list
      const workspaces = await client.getWorkspaces()
      expect(workspaces).not.toContain(tempWorkspaceId)
    })
  })

  // ===========================================================================
  // Peer and Session Access
  // ===========================================================================

  describe('Peer access', () => {
    test('peer() returns Peer instance without API call', async () => {
      const peer = await client.peer('lazy-peer')

      expect(peer.id).toBe('lazy-peer')
      expect(peer.workspaceId).toBe(client.workspaceId)
    })

    test('peer() with metadata makes API call', async () => {
      const peer = await client.peer('eager-peer', {
        metadata: { created: true },
      })

      expect(peer.id).toBe('eager-peer')
      expect(peer.metadata).toEqual({ created: true })
    })

    test('getPeers returns paginated list', async () => {
      // Create some peers
      await client.peer('list-peer-1', { metadata: {} })
      await client.peer('list-peer-2', { metadata: {} })

      const page = await client.getPeers()

      expect(page.items.length).toBeGreaterThanOrEqual(2)
      expect(page.total).toBeGreaterThanOrEqual(2)

      const ids = page.items.map((p) => p.id)
      expect(ids).toContain('list-peer-1')
      expect(ids).toContain('list-peer-2')
    })
  })

  describe('Session access', () => {
    test('session() returns Session instance without API call', async () => {
      const session = await client.session('lazy-session', { metadata: {} })

      expect(session.id).toBe('lazy-session')
      expect(session.workspaceId).toBe(client.workspaceId)
    })

    test('session() with metadata makes API call', async () => {
      const session = await client.session('eager-session', {
        metadata: { created: true },
      })

      expect(session.id).toBe('eager-session')
      expect(session.metadata).toEqual({ created: true })
    })

    test('getSessions returns paginated list', async () => {
      // Create some sessions
      await client.session('list-session-1', { metadata: {} })
      await client.session('list-session-2', { metadata: {} })

      const page = await client.getSessions()

      expect(page.items.length).toBeGreaterThanOrEqual(2)

      const ids = page.items.map((s) => s.id)
      expect(ids).toContain('list-session-1')
      expect(ids).toContain('list-session-2')
    })
  })

  // ===========================================================================
  // Search
  // ===========================================================================

  describe('POST /workspaces/:id/search', () => {
    test('search returns matching messages', async () => {
      // Setup: create session with messages
      const session = await client.session('search-session', { metadata: {} })
      const peer = await client.peer('search-peer')

      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('The quick brown fox jumps over the lazy dog'),
        peer.message('Hello world, this is a test message'),
      ])

      // Search for content
      const results = await client.search('quick brown fox')

      // Note: Vector search may return results based on semantic similarity
      expect(Array.isArray(results)).toBe(true)
    })

    test('search with filters scopes results', async () => {
      const session = await client.session('search-filtered-session', { metadata: {} })
      const peer = await client.peer('search-filtered-peer')

      await session.addPeers([peer.id])
      await session.addMessages([
        peer.message('Unique searchable content xyz123'),
      ])

      const results = await client.search('unique searchable', {
        filters: { session_id: session.id },
      })

      // All results should be from the specified session
      for (const msg of results) {
        expect(msg.session_id).toBe(session.id)
      }
    })

    test('search with limit constrains results', async () => {
      const results = await client.search('test', { limit: 5 })

      expect(results.length).toBeLessThanOrEqual(5)
    })
  })

  // ===========================================================================
  // Queue Status
  // ===========================================================================

  describe('GET /workspaces/:id/queue/status', () => {
    test('getQueueStatus returns status object', async () => {
      const status = await client.getQueueStatus()

      expect(typeof status.totalWorkUnits).toBe('number')
      expect(typeof status.completedWorkUnits).toBe('number')
      expect(typeof status.inProgressWorkUnits).toBe('number')
      expect(typeof status.pendingWorkUnits).toBe('number')
    })

    test('getQueueStatus with observer filter', async () => {
      const peer = await client.peer('queue-observer')

      const status = await client.getQueueStatus({
        observer: peer,
      })

      expect(typeof status.totalWorkUnits).toBe('number')
    })

    test('getQueueStatus with session filter', async () => {
      const session = await client.session('queue-session', { metadata: {} })

      const status = await client.getQueueStatus({
        session: session,
      })

      expect(typeof status.totalWorkUnits).toBe('number')
    })
  })

  // ===========================================================================
  // Message Updates
  // ===========================================================================

  describe('PUT /sessions/:id/messages/:id', () => {
    test('updateMessage updates message metadata', async () => {
      const session = await client.session('update-msg-session', { metadata: {} })
      const peer = await client.peer('update-msg-peer')

      await session.addPeers([peer.id])
      const [message] = await session.addMessages([
        peer.message('Message to update'),
      ])

      const updated = await client.updateMessage(
        message,
        { updated: true, timestamp: Date.now() }
      )

      expect(updated.metadata.updated).toBe(true)
      expect(updated.metadata.timestamp).toBeDefined()
    })

    test('updateMessage with string ID requires session', async () => {
      const session = await client.session('update-msg-session-2', { metadata: {} })
      const peer = await client.peer('update-msg-peer-2')

      await session.addPeers([peer.id])
      const [message] = await session.addMessages([
        peer.message('Another message'),
      ])

      const updated = await client.updateMessage(
        message.id,
        { fromId: true },
        session
      )

      expect(updated.metadata.fromId).toBe(true)
    })

    test('updateMessage throws without session when using string ID', async () => {
      await expect(
        client.updateMessage('some-message-id', { foo: 'bar' })
      ).rejects.toThrow('session is required')
    })
  })

  // ===========================================================================
  // Client Configuration
  // ===========================================================================

  describe('Client configuration', () => {
    test('baseURL is accessible', () => {
      expect(client.baseURL).toBe(TEST_CONFIG.baseURL)
    })

    test('http client is accessible', () => {
      expect(client.http).toBeDefined()
      expect(client.http.baseURL).toBe(TEST_CONFIG.baseURL)
    })

    test('toString returns readable representation', () => {
      const str = client.toString()
      expect(str).toContain('Honcho')
      expect(str).toContain(client.workspaceId)
    })

    test('environment option sets local URL', () => {
      const localClient = new Honcho({
        environment: 'local',
        workspaceId: 'test',
      })

      expect(localClient.baseURL).toBe('http://localhost:8000')
    })
  })
})

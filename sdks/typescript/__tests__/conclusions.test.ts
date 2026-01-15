/**
 * Conclusions Tests
 *
 * Tests for Conclusion operations via ConclusionScope.
 *
 * Endpoints covered:
 * - POST /v1/workspaces/:id/conclusions (create conclusions)
 * - POST /v1/workspaces/:id/conclusions/list (list conclusions)
 * - POST /v1/workspaces/:id/conclusions/query (semantic search)
 * - DELETE /v1/workspaces/:id/conclusions/:id (delete conclusion)
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho, Conclusion, ConclusionScope } from '../src'
import { createTestClient, requireServer } from './setup'
import { assertConclusionShape } from './helpers'

describe('Conclusions', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('conclusions')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // ===========================================================================
  // ConclusionScope Access
  // ===========================================================================

  describe('ConclusionScope access', () => {
    test('peer.conclusions returns self-scope', async () => {
      const peer = await client.peer('self-scope-peer')

      const scope = peer.conclusions

      expect(scope).toBeInstanceOf(ConclusionScope)
      expect(scope.observer).toBe(peer.id)
      expect(scope.observed).toBe(peer.id)
      expect(scope.workspaceId).toBe(client.workspaceId)
    })

    test('peer.conclusionsOf returns target scope', async () => {
      const observer = await client.peer('observer-peer')
      const target = await client.peer('target-peer')

      const scope = observer.conclusionsOf(target)

      expect(scope.observer).toBe(observer.id)
      expect(scope.observed).toBe(target.id)
    })

    test('conclusionsOf with string ID', async () => {
      const peer = await client.peer('string-target-peer')

      const scope = peer.conclusionsOf('some-target-id')

      expect(scope.observed).toBe('some-target-id')
    })
  })

  // ===========================================================================
  // Conclusion Creation (POST /conclusions)
  // ===========================================================================

  describe('POST /conclusions (create)', () => {
    test('create single conclusion', async () => {
      const peer = await client.peer('create-single-conclusion-peer')
      const session = await client.session('create-single-conclusion-session')

      const conclusions = await peer.conclusions.create({
        content: 'User prefers dark mode',
        sessionId: session.id,
      })

      expect(conclusions.length).toBe(1)
      expect(conclusions[0]).toBeInstanceOf(Conclusion)
      expect(conclusions[0].content).toBe('User prefers dark mode')
      expect(conclusions[0].observerId).toBe(peer.id)
      expect(conclusions[0].observedId).toBe(peer.id)
    })

    test('create multiple conclusions', async () => {
      const peer = await client.peer('create-multi-conclusion-peer')
      const session = await client.session('create-multi-conclusion-session')

      const conclusions = await peer.conclusions.create([
        { content: 'Likes TypeScript', sessionId: session },
        { content: 'Uses VS Code', sessionId: session.id },
        { content: 'Prefers tabs over spaces', sessionId: session },
      ])

      expect(conclusions.length).toBe(3)
      expect(conclusions[0].content).toBe('Likes TypeScript')
      expect(conclusions[1].content).toBe('Uses VS Code')
      expect(conclusions[2].content).toBe('Prefers tabs over spaces')
    })

    test('create conclusion for target peer', async () => {
      const observer = await client.peer('conclusion-observer')
      const observed = await client.peer('conclusion-observed')
      const session = await client.session('conclusion-target-session')

      const scope = observer.conclusionsOf(observed)
      const conclusions = await scope.create({
        content: 'Observed peer likes coffee',
        sessionId: session,
      })

      expect(conclusions[0].observerId).toBe(observer.id)
      expect(conclusions[0].observedId).toBe(observed.id)
    })
  })

  // ===========================================================================
  // Conclusion Listing (POST /conclusions/list)
  // ===========================================================================

  describe('POST /conclusions/list', () => {
    test('list returns conclusions in scope', async () => {
      const peer = await client.peer('list-conclusion-peer')
      const session = await client.session('list-conclusion-session')

      // Create some conclusions first
      await peer.conclusions.create([
        { content: 'Conclusion A', sessionId: session },
        { content: 'Conclusion B', sessionId: session },
      ])

      const conclusions = await peer.conclusions.list()

      expect(conclusions.length).toBeGreaterThanOrEqual(2)
      // All should be self-conclusions
      for (const c of conclusions) {
        expect(c.observerId).toBe(peer.id)
        expect(c.observedId).toBe(peer.id)
      }
    })

    test('list with pagination', async () => {
      const peer = await client.peer('paginate-conclusion-peer')
      const session = await client.session('paginate-conclusion-session')

      // Create several conclusions
      await peer.conclusions.create(
        Array.from({ length: 5 }, (_, i) => ({
          content: `Paginated conclusion ${i + 1}`,
          sessionId: session,
        }))
      )

      // Get with small page size
      const page1 = await peer.conclusions.list(1, 2)

      expect(page1.length).toBeLessThanOrEqual(2)
    })

    test('list scoped to session', async () => {
      const peer = await client.peer('session-scope-conclusion-peer')
      const session1 = await client.session('conclusion-session-1')
      const session2 = await client.session('conclusion-session-2')

      await peer.conclusions.create([
        { content: 'Session 1 conclusion', sessionId: session1 },
        { content: 'Session 2 conclusion', sessionId: session2 },
      ])

      const conclusions = await peer.conclusions.list(1, 50, session1)

      // All results should be from session1
      for (const c of conclusions) {
        expect(c.sessionId).toBe(session1.id)
      }
    })

    test('list for target peer scope', async () => {
      const observer = await client.peer('list-target-observer')
      const target = await client.peer('list-target-target')
      const session = await client.session('list-target-session')

      await observer.conclusionsOf(target).create({
        content: 'About the target',
        sessionId: session,
      })

      const conclusions = await observer.conclusionsOf(target).list()

      expect(conclusions.length).toBeGreaterThanOrEqual(1)
      for (const c of conclusions) {
        expect(c.observerId).toBe(observer.id)
        expect(c.observedId).toBe(target.id)
      }
    })
  })

  // ===========================================================================
  // Conclusion Query (POST /conclusions/query)
  // ===========================================================================

  describe('POST /conclusions/query', () => {
    test('query returns semantically similar conclusions', async () => {
      const peer = await client.peer('query-conclusion-peer')
      const session = await client.session('query-conclusion-session')

      // Create conclusions with distinct topics
      await peer.conclusions.create([
        { content: 'User enjoys programming in Python', sessionId: session },
        { content: 'User likes hiking in mountains', sessionId: session },
        { content: 'User prefers tea over coffee', sessionId: session },
      ])

      const results = await peer.conclusions.query('programming languages')

      expect(Array.isArray(results)).toBe(true)
      // Results should be semantically relevant to the query
    })

    test('query with topK limit', async () => {
      const peer = await client.peer('topk-conclusion-peer')
      const session = await client.session('topk-conclusion-session')

      await peer.conclusions.create(
        Array.from({ length: 10 }, (_, i) => ({
          content: `Conclusion about topic ${i}`,
          sessionId: session,
        }))
      )

      const results = await peer.conclusions.query('topic', 3)

      expect(results.length).toBeLessThanOrEqual(3)
    })

    test('query with distance threshold', async () => {
      const peer = await client.peer('distance-conclusion-peer')
      const session = await client.session('distance-conclusion-session')

      await peer.conclusions.create({
        content: 'Very specific unique content xyz123',
        sessionId: session,
      })

      const results = await peer.conclusions.query(
        'specific unique xyz123',
        10,
        0.5 // Strict distance threshold
      )

      expect(Array.isArray(results)).toBe(true)
    })

    test('query scoped to target peer', async () => {
      const observer = await client.peer('query-target-observer')
      const target = await client.peer('query-target-target')
      const session = await client.session('query-target-session')

      await observer.conclusionsOf(target).create({
        content: 'Target likes machine learning',
        sessionId: session,
      })

      const results = await observer.conclusionsOf(target).query('ML AI')

      for (const c of results) {
        expect(c.observerId).toBe(observer.id)
        expect(c.observedId).toBe(target.id)
      }
    })
  })

  // ===========================================================================
  // Conclusion Deletion (DELETE /conclusions/:id)
  // ===========================================================================

  describe('DELETE /conclusions/:id', () => {
    test('delete removes conclusion', async () => {
      const peer = await client.peer('delete-conclusion-peer')
      const session = await client.session('delete-conclusion-session')

      const [conclusion] = await peer.conclusions.create({
        content: 'To be deleted',
        sessionId: session,
      })

      // Delete it
      await peer.conclusions.delete(conclusion.id)

      // Should not appear in list
      const remaining = await peer.conclusions.list()
      const ids = remaining.map((c) => c.id)
      expect(ids).not.toContain(conclusion.id)
    })
  })

  // ===========================================================================
  // Representation from Scope
  // ===========================================================================

  describe('getRepresentation from scope', () => {
    test('returns representation for self-scope', async () => {
      const peer = await client.peer('repr-self-scope-peer')
      const session = await client.session('repr-self-scope-session')

      await peer.conclusions.create({
        content: 'User is a software engineer',
        sessionId: session,
      })

      const representation = await peer.conclusions.getRepresentation()

      expect(typeof representation).toBe('string')
    })

    test('returns representation for target scope', async () => {
      const observer = await client.peer('repr-target-scope-observer')
      const target = await client.peer('repr-target-scope-target')
      const session = await client.session('repr-target-scope-session')

      await observer.conclusionsOf(target).create({
        content: 'Target is friendly',
        sessionId: session,
      })

      const representation = await observer
        .conclusionsOf(target)
        .getRepresentation()

      expect(typeof representation).toBe('string')
    })

    test('representation with options', async () => {
      const peer = await client.peer('repr-options-scope-peer')

      const representation = await peer.conclusions.getRepresentation({
        searchQuery: 'preferences',
        searchTopK: 5,
        maxConclusions: 20,
      })

      expect(typeof representation).toBe('string')
    })
  })

  // ===========================================================================
  // Conclusion Class
  // ===========================================================================

  describe('Conclusion class', () => {
    test('fromApiResponse creates instance', () => {
      const response = {
        id: 'test-id',
        content: 'Test content',
        observer_id: 'observer',
        observed_id: 'observed',
        session_id: 'session',
        created_at: '2024-01-15T10:00:00Z',
      }

      const conclusion = Conclusion.fromApiResponse(response)

      expect(conclusion.id).toBe('test-id')
      expect(conclusion.content).toBe('Test content')
      expect(conclusion.observerId).toBe('observer')
      expect(conclusion.observedId).toBe('observed')
      expect(conclusion.sessionId).toBe('session')
      expect(conclusion.createdAt).toBe('2024-01-15T10:00:00Z')
    })

    test('toString returns readable format', async () => {
      const peer = await client.peer('tostring-conclusion-peer')
      const session = await client.session('tostring-conclusion-session')

      const [conclusion] = await peer.conclusions.create({
        content: 'A longer conclusion that should be truncated in toString',
        sessionId: session,
      })

      const str = conclusion.toString()

      expect(str).toContain('Conclusion')
      expect(str).toContain(conclusion.id)
    })

    test('toString truncates long content', () => {
      const conclusion = new Conclusion(
        'id',
        'A'.repeat(100),
        'observer',
        'observed',
        'session',
        '2024-01-01'
      )

      const str = conclusion.toString()

      expect(str).toContain('...')
      expect(str.length).toBeLessThan(100)
    })
  })

  // ===========================================================================
  // ConclusionScope toString
  // ===========================================================================

  describe('ConclusionScope toString', () => {
    test('returns readable format', async () => {
      const peer = await client.peer('scope-tostring-peer')

      const str = peer.conclusions.toString()

      expect(str).toContain('ConclusionScope')
      expect(str).toContain(peer.id)
      expect(str).toContain(client.workspaceId)
    })
  })
})

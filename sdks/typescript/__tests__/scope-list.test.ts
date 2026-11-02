/**
 * Scope List Tests
 *
 * The scope list can filter by scope name and report each scope's member
 * session count.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import type { Honcho } from '../src'
import { createTestClient, generateId, requireServer } from './setup'

describe('scope list', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('scope-list')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // Pages hold one scope each, and the scopes the filter matches alternate
  // with ones it doesn't, so a later page fetched without the filter would
  // differ.
  test('filters and includeSessionCount reach the server on every page', async () => {
    const tag = generateId('t').replace(/[^a-zA-Z0-9]/g, '')
    const [s1, s2] = [generateId('s1'), generateId('s2')]
    for (const id of [s1, s2]) {
      await client.session(id)
    }
    const members: Array<[string, string[]]> = [
      [`keep1${tag}`, [s1]],
      [`drop1${tag}`, []],
      [`keep2${tag}`, [s1, s2]],
      [`drop2${tag}`, []],
    ]
    for (const [scopeId, sessions] of members) {
      const scope = await client.scope(scopeId)
      if (sessions.length) {
        await scope.addSessions(sessions)
      }
    }

    const counted = await (
      await client.scopes({
        filters: {
          AND: [{ id: { icontains: tag } }, { id: { icontains: 'keep' } }],
        },
        size: 1,
        includeSessionCount: true,
      })
    ).toArray()
    expect(counted.map((s) => [s.id, s.sessionCount])).toEqual([
      [`keep1${tag}`, 1],
      [`keep2${tag}`, 2],
    ])

    const plain = await (
      await client.scopes({ filters: { id: { icontains: tag } }, size: 1 })
    ).toArray()
    expect(plain.map((s) => s.id)).toEqual(members.map(([id]) => id))
    expect(plain.every((s) => s.sessionCount === undefined)).toBe(true)
  })
})

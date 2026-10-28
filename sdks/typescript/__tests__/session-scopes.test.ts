/**
 * Session List Scope Tests
 *
 * The workspace session list can report each session's scopes and filter by
 * scope membership.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import type { Honcho } from '../src'
import { createTestClient, generateId, requireServer } from './setup'

describe('session list scopes', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('session-scopes')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  // Pages hold one session each, and each filter's members alternate with
  // non-members in creation order, so a later page fetched without the
  // options would return a different session.
  test('scope, inNoScope and includeScopes reach the server on every page', async () => {
    const [alphaId, betaId] = [generateId('alpha'), generateId('beta')].sort()
    const ids = [1, 2, 3, 4].map((n) => generateId(`s${n}`))
    const [s1, s2, s3, s4] = ids
    for (const id of ids) {
      await client.session(id)
    }
    const alpha = await client.scope(alphaId)
    const beta = await client.scope(betaId)
    await alpha.addSessions([s1, s3])
    await beta.addSessions([s1])
    const onlyOurs = { id: { in: ids } }
    const list = async (options: Record<string, unknown>) =>
      (
        await client.sessions({ filters: onlyOurs, size: 1, ...options })
      ).toArray()

    const withScopes = await list({ includeScopes: true })
    expect(withScopes.map((s) => [s.id, s.scopes])).toEqual([
      [s1, [alphaId, betaId]],
      [s2, []],
      [s3, [alphaId]],
      [s4, []],
    ])

    const inAlpha = await list({ scope: alphaId, includeScopes: true })
    expect(inAlpha.map((s) => [s.id, s.scopes])).toEqual([
      [s1, [alphaId, betaId]],
      [s3, [alphaId]],
    ])

    const inNone = await list({ inNoScope: true })
    expect(inNone.map((s) => s.id)).toEqual([s2, s4])

    const plain = await list({})
    expect(plain.map((s) => s.id)).toEqual(ids)
    expect(plain.every((s) => s.scopes === undefined)).toBe(true)
  })
})

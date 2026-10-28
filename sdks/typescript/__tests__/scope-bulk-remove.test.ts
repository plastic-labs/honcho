/**
 * Scope Bulk Remove Tests
 *
 * `scope.removeSessions` ends many memberships in one request against a real
 * server.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import type { Honcho } from '../src'
import { createTestClient, generateId, requireServer } from './setup'

describe('scope bulk remove', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('scope-bulk-remove')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  test('removes members, ignores non-members, and rejects unknown sessions', async () => {
    const [kept, gone1, gone2, outsider] = [1, 2, 3, 4].map((n) =>
      generateId(`s${n}`)
    )
    for (const id of [kept, gone1, gone2, outsider]) {
      await client.session(id)
    }
    const scope = await client.scope(generateId('scope'))
    await scope.addSessions([kept, gone1, gone2])
    const members = async () =>
      new Set((await (await scope.sessions()).toArray()).map((s) => s.id))

    // An unknown session rejects the whole batch
    await expect(
      scope.removeSessions([gone1, generateId('missing')])
    ).rejects.toThrow()
    expect(await members()).toEqual(new Set([kept, gone1, gone2]))

    await scope.removeSessions([gone1, gone2, outsider])
    expect(await members()).toEqual(new Set([kept]))
  })
})

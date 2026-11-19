/**
 * Iteration against a real server: offset page 1, then cursor mode through
 * `next_cursor`, returns the same rows in the same order as an offset walk.
 */

import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import type { Honcho } from '../src'
import type { Page } from '../src/pagination'
import type { Peer } from '../src/peer'
import type { PeerResponse } from '../src/types/api'
import { createTestClient, requireServer } from './setup'

const SEEDED = 7
const SIZE = 2

async function offsetWalk(first: Page<Peer, PeerResponse>): Promise<string[]> {
  const ids: string[] = []
  let page: Page<Peer, PeerResponse> | null = first
  while (page) {
    ids.push(...page.items.map((peer) => peer.id))
    page = await page.getNextPage()
  }
  return ids
}

describe('Page iteration against the server', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('pagination')
    client = setup.client
    cleanup = setup.cleanup
    for (let i = 0; i < SEEDED; i++) {
      await client.peer(`cursor-peer-${i}`, { metadata: {} })
    }
  })

  afterAll(async () => {
    await cleanup()
  })

  for (const reverse of [false, true]) {
    test(`iteration matches an offset walk (reverse=${reverse})`, async () => {
      const expected = await offsetWalk(
        await client.peers({ size: SIZE, reverse })
      )
      const first = await client.peers({ size: SIZE, reverse })

      const ids: string[] = []
      for await (const peer of first) {
        ids.push(peer.id)
      }

      expect(expected.length).toBeGreaterThanOrEqual(SEEDED)
      expect(first.nextCursor).not.toBeNull()
      expect(first.totalKind).toBe('exact')
      expect(ids).toEqual(expected)
      expect(new Set(ids).size).toBe(ids.length)
    })
  }
})

/**
 * Peer Membership Tests
 *
 * Composable peers: managing a peer's members, and reads that aggregate over
 * them by `depth`.
 *
 * Endpoints covered:
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/members (add member)
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/members/list (list members)
 * - DELETE /v3/workspaces/:workspaceId/peers/:peerId/members/:memberId (remove member)
 * - POST /v3/workspaces/:workspaceId/peers/:peerId/parents/list (list parents)
 * - depth on context, representation, and search
 */

import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { ZodError } from 'zod'
import {
  ConflictError,
  type Honcho,
  Page,
  UnprocessableEntityError,
} from '../src'
import type { HonchoHTTPClient } from '../src/http/client'
import { Peer } from '../src/peer'
import { createTestClient, requireServer } from './setup'

describe('Peer members', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('members')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  test('add, list, and remove members', async () => {
    const team = await client.peer('team')
    const bob = await client.peer('bob')

    const alice = await team.members.add('alice', {
      metadata: { role: 'lead' },
    })
    await team.members.add(bob)

    expect(alice.parentId).toBe(team.id)
    expect(alice.peerId).toBe('alice')
    expect(alice.workspaceId).toBe(client.workspaceId)
    expect(alice.metadata).toEqual({ role: 'lead' })
    expect(typeof alice.createdAt).toBe('string')

    const page = await team.members.list()
    expect(page).toBeInstanceOf(Page)
    // Oldest membership first.
    expect(page.items.map((m) => m.peerId)).toEqual(['alice', 'bob'])

    await team.members.remove('alice')
    // Removing a peer that is not a member is a no-op.
    await team.members.remove('alice')
    const remaining = await team.members.list()
    expect(remaining.items.map((m) => m.peerId)).toEqual(['bob'])
  })

  test('list filters on the member peer', async () => {
    const team = await client.peer('filter-team')
    await client.peer('filter-human', { metadata: { kind: 'human' } })
    await client.peer('filter-bot', { metadata: { kind: 'agent' } })
    await team.members.add('filter-human')
    await team.members.add('filter-bot')

    const page = await team.members.list({
      filters: { metadata: { kind: 'agent' } },
    })

    expect(page.items.map((m) => m.peerId)).toEqual(['filter-bot'])
  })

  test('parents lists the peers a peer belongs to', async () => {
    const alice = await client.peer('parents-alice')
    await (await client.peer('parents-team')).members.add(alice)
    await (await client.peer('parents-guild')).members.add(alice)

    const page = await alice.parents()

    expect(page.items.map((m) => m.parentId)).toEqual([
      'parents-team',
      'parents-guild',
    ])
    expect(page.items.every((m) => m.peerId === alice.id)).toBe(true)
  })

  test('re-adding a member returns the existing membership', async () => {
    const team = await client.peer('readd-team')

    const first = await team.members.add('readd-alice', {
      metadata: { role: 'ic' },
    })
    const again = await team.members.add('readd-alice')
    const replaced = await team.members.add('readd-alice', {
      metadata: { role: 'lead' },
    })

    expect(again.createdAt).toBe(first.createdAt)
    expect(again.metadata).toEqual({ role: 'ic' })
    expect(replaced.metadata).toEqual({ role: 'lead' })
    expect((await team.members.list()).items).toHaveLength(1)
  })

  test('a cycle rejects with ConflictError', async () => {
    const team = await client.peer('cycle-team')
    const alice = await client.peer('cycle-alice')
    await team.members.add(alice)

    await expect(alice.members.add(team)).rejects.toBeInstanceOf(ConflictError)
  })

  test('a peer cannot be its own member', async () => {
    const team = await client.peer('self-team')

    await expect(team.members.add(team)).rejects.toBeInstanceOf(
      UnprocessableEntityError
    )
  })

  test('context surfaces memberCards only when aggregated', async () => {
    const team = await client.peer('cards-team')
    const alice = await client.peer('cards-alice')
    await team.setCard(['The team ships on Fridays'])
    await alice.setCard(['Alice leads the rewrite'])
    await team.members.add(alice)

    const aggregated = await team.context()
    const alone = await team.context({ depth: 0 })

    expect(aggregated.peerCard).toEqual(['The team ships on Fridays'])
    expect(aggregated.memberCards).toEqual({
      [alice.id]: ['Alice leads the rewrite'],
    })
    expect(alone.memberCards).toBeNull()
    // The card getter returns the target's own card; members' come via context.
    expect(await team.getCard()).toEqual(['The team ships on Fridays'])
  })

  test('representation aggregates member conclusions by depth', async () => {
    const team = await client.peer('rep-team')
    const alice = await client.peer('rep-alice')
    await alice.conclusions.create([{ content: 'Alice leads the rewrite' }])
    await team.members.add(alice)

    expect(await team.representation()).toContain('Alice leads the rewrite')
    expect(await team.representation({ depth: 0 })).not.toContain(
      'Alice leads the rewrite'
    )
  })

  test('search matches member messages by depth', async () => {
    const team = await client.peer('search-team')
    const alice = await client.peer('search-alice')
    const session = await client.session('search-members-session')
    const content = 'the rewrite will use a unique storage engine'
    await session.addMessages([alice.message(content)])
    await team.members.add(alice)

    const aggregated = await team.search(content)
    const alone = await team.search(content, { depth: 0 })

    expect(aggregated.map((m) => m.peerId)).toEqual([alice.id])
    expect(alone).toEqual([])
  })
})

/**
 * Capture the body and query of the last request, so tests can assert the
 * wire shape without a server.
 */
function capturingHttp(response: unknown): {
  http: HonchoHTTPClient
  body: () => Record<string, unknown> | undefined
  query: () => Record<string, unknown> | undefined
} {
  let capturedBody: Record<string, unknown> | undefined
  let capturedQuery: Record<string, unknown> | undefined
  const http = {
    post: async (
      _path: string,
      options?: { body?: Record<string, unknown> }
    ) => {
      capturedBody = options?.body
      return response
    },
    get: async (
      _path: string,
      options?: { query?: Record<string, unknown> }
    ) => {
      capturedQuery = options?.query
      return response
    },
    stream: async (
      _method: string,
      _path: string,
      options?: { body?: Record<string, unknown> }
    ) => {
      capturedBody = options?.body
      return new Response('data: {"done": true}\n\n')
    },
  } as unknown as HonchoHTTPClient
  return { http, body: () => capturedBody, query: () => capturedQuery }
}

describe('depth option', () => {
  test('chat sends depth only when given', async () => {
    const { http, body } = capturingHttp({ content: 'ok' })
    const peer = new Peer('team', 'workspace-1', http)

    await peer.chat('q')
    expect(body()?.depth).toBeUndefined()

    await peer.chat('q', { depth: 0 })
    expect(body()).toMatchObject({ depth: 0 })
  })

  test('chatStream sends depth', async () => {
    const { http, body } = capturingHttp(undefined)
    const peer = new Peer('team', 'workspace-1', http)

    await peer.chatStream('q', { depth: 2 })

    expect(body()).toMatchObject({ depth: 2, stream: true })
  })

  test('context sends depth as a query parameter', async () => {
    const { http, query } = capturingHttp({
      peer_id: 'team',
      target_id: 'team',
      representation: '',
      peer_card: null,
    })
    const peer = new Peer('team', 'workspace-1', http)

    await peer.context({ depth: 2 })
    expect(query()).toMatchObject({ depth: 2 })
  })

  test('a negative depth is rejected before the request', async () => {
    const { http } = capturingHttp({ representation: '' })
    const peer = new Peer('team', 'workspace-1', http)

    await expect(peer.representation({ depth: -1 })).rejects.toBeInstanceOf(
      ZodError
    )
  })

  test('remove rejects a malformed member ID', async () => {
    const { http } = capturingHttp(undefined)
    const peer = new Peer('team', 'workspace-1', http)

    // The ID lands in the request path, so `alice?typo` would remove `alice`.
    await expect(peer.members.remove('alice?typo')).rejects.toBeInstanceOf(
      ZodError
    )
  })
})

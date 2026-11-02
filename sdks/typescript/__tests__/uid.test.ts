/**
 * UID Tests
 *
 * Workspaces, peers and sessions expose the server-assigned `uid` alongside the
 * client-chosen `id`, and keep the same `uid` across fetch paths.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho, Peer } from '../src'
import {
  createTestClient,
  generateId,
  requireServer,
  TEST_CONFIG,
} from './setup'

describe('uid', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('uid')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  test('client learns its workspace uid from its first API call', async () => {
    const fresh = new Honcho({
      baseURL: TEST_CONFIG.baseURL,
      apiKey: TEST_CONFIG.apiKey,
      workspaceId: client.workspaceId,
      timeout: TEST_CONFIG.timeout,
    })
    expect(fresh.uid).toBeUndefined()

    await fresh.getMetadata()
    expect(typeof fresh.uid).toBe('string')
    expect(fresh.uid).not.toBe(fresh.workspaceId)
    expect(client.uid).toBe(fresh.uid)
  })

  test('peer uid is stable across get-or-create, list and refresh', async () => {
    const peerId = generateId('uid-peer')
    const peer = await client.peer(peerId)
    expect(typeof peer.uid).toBe('string')
    expect(peer.uid).not.toBe(peer.id)

    const listed = await (await client.peers()).toArray()
    expect(listed.find((p) => p.id === peerId)?.uid).toBe(peer.uid)

    const unfetched = new Peer(peerId, client.workspaceId, client.http)
    expect(unfetched.uid).toBeUndefined()
    await unfetched.refresh()
    expect(unfetched.uid).toBe(peer.uid)
  })

  test('session uid is stable; session peers carry theirs; clones get a new one', async () => {
    const peer = await client.peer(generateId('uid-session-peer'))
    const session = await client.session(generateId('uid-session'))
    expect(typeof session.uid).toBe('string')
    expect(session.uid).not.toBe(session.id)

    await session.addPeers(peer)
    const sessionPeers = await session.peers()
    expect(sessionPeers.map((p) => p.uid)).toEqual([peer.uid])

    const listed = await (await client.sessions()).toArray()
    expect(listed.find((s) => s.id === session.id)?.uid).toBe(session.uid)

    const clone = await session.clone()
    expect(typeof clone.uid).toBe('string')
    expect(clone.uid).not.toBe(session.uid)
  })
})

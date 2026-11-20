/**
 * Ingest Tests
 *
 * Auto workspaces, batched ingest with indexes counted across batches, and
 * releasing records ingested with `derive: 'later'`.
 */

import { beforeAll, describe, expect, test } from 'bun:test'
import { ConflictError, Honcho, type IngestRecord, NotFoundError } from '../src'
import { generateWorkspaceId, requireServer, TEST_CONFIG } from './setup'

function sdk(workspaceId: string, auto?: boolean): Honcho {
  return new Honcho({
    baseURL: TEST_CONFIG.baseURL,
    apiKey: TEST_CONFIG.apiKey,
    workspaceId,
    timeout: TEST_CONFIG.timeout,
    ...(auto === undefined ? {} : { auto }),
  })
}

function* records(count: number): Generator<IngestRecord> {
  for (let n = 0; n < count; n++) {
    yield {
      source: 'discord',
      sourceInstance: 'guild1',
      sourceId: `msg-${n}`,
      createdAt: new Date(Date.UTC(2026, 10, 4, 10, 0, n)),
      container: { id: `chan-${n % 3}`, name: `channel ${n % 3}` },
      author: { id: `user-${n % 4}`, name: `Person ${n % 4}` },
      content: `message ${n}`,
    }
  }
}

async function workspaceAuto(workspaceId: string): Promise<boolean | null> {
  const page = await sdk('default').http.post<{
    items: { id: string; auto?: boolean }[]
  }>('/v3/workspaces/list', { body: { filters: { id: workspaceId } } })
  const found = page.items.find((w) => w.id === workspaceId)
  return found ? (found.auto ?? false) : null
}

describe('ingest', () => {
  beforeAll(async () => {
    await requireServer()
  })

  test('an auto client creates an auto workspace', async () => {
    const id = generateWorkspaceId('ingest-auto')
    await sdk(id, true).getMetadata()
    expect(await workspaceAuto(id)).toBe(true)
  })

  test('an auto client refuses an existing regular workspace', async () => {
    const id = generateWorkspaceId('ingest-regular')
    await sdk(id).getMetadata()
    await expect(sdk(id, true).getMetadata()).rejects.toBeInstanceOf(
      ConflictError
    )
  })

  test('ingest sends every record in batches and numbers them in order', async () => {
    const client = sdk(generateWorkspaceId('ingest-batches'), true)

    const result = await client.ingest(records(250), { derive: 'later' })

    expect(result.records.map((r) => r.index)).toEqual(
      Array.from({ length: 250 }, (_, i) => i)
    )
    expect(new Set(result.records.map((r) => r.outcome))).toEqual(
      new Set(['created'])
    )
    expect(new Set(result.records.map((r) => r.derivation))).toEqual(
      new Set(['deferred'])
    )
    expect(new Set(result.records.map((r) => r.messageId)).size).toBe(250)

    const again = await client.ingest(records(250), { derive: 'later' })
    expect(new Set(again.records.map((r) => r.outcome))).toEqual(
      new Set(['duplicate'])
    )
    expect(again.records.map((r) => r.messageId)).toEqual(
      result.records.map((r) => r.messageId)
    )
  })

  test('ingestDerive releases every deferred record', async () => {
    const client = sdk(generateWorkspaceId('ingest-derive'), true)
    await client.ingest(records(120), { derive: 'later' })

    expect(await client.ingestDerive()).toEqual({ pending: 120, remaining: 0 })
    expect(await client.ingestDerive()).toEqual({ pending: 0, remaining: 0 })
  })

  test('ingest without auto does not create the workspace', async () => {
    const id = generateWorkspaceId('ingest-missing')
    await expect(sdk(id).ingest(records(1))).rejects.toBeInstanceOf(
      NotFoundError
    )
    expect(await workspaceAuto(id)).toBeNull()
  })

  test('ingest into a regular workspace is refused', async () => {
    const client = sdk(generateWorkspaceId('ingest-refused'))
    await client.getMetadata()
    await expect(client.ingest(records(1))).rejects.toBeInstanceOf(
      ConflictError
    )
  })

  test('null fields and long unicode names are sent as the server takes them', async () => {
    const client = sdk(generateWorkspaceId('ingest-nullable'), true)
    const result = await client.ingest([
      {
        source: 'discord',
        sourceInstance: 'guild1',
        sourceId: null,
        revisedAt: null,
        createdAt: '2026-11-04T10:00:00Z',
        container: { id: 'chan-1', name: null, parentId: null },
        author: { id: null, name: '\u{1F600}'.repeat(300), bot: null },
        participants: null,
        metadata: null,
        content: 'hello',
      },
    ])
    expect(result.records.map((r) => r.outcome)).toEqual(['created'])
  })

  test('a bad record throws before its batch is sent', async () => {
    const client = sdk(generateWorkspaceId('ingest-invalid'), true)
    const bad = [...records(3)]
    bad[2] = { ...bad[2], sourceInstance: '' }

    await expect(client.ingest(bad)).rejects.toThrow()
    const sessions = await client.sessions()
    expect(sessions.total).toBe(0)
  })

  test('review an import before deriving it', async () => {
    const client = sdk(generateWorkspaceId('ingest-review'), true)
    await client.ingest(records(12), { derive: 'later' })

    const status = await client.ingestStatus()
    expect(status.records.deferred).toBe(12)
    expect([status.people, status.sessions]).toEqual([4, 3])
    expect(status.deferredTokens).toBeGreaterThan(0)

    const people = await (await client.ingestPeople({ size: 2 })).toArray()
    expect(people.map((p) => p.messages)).toEqual([3, 3, 3, 3])
    expect(new Set(people.map((p) => p.displayName))).toEqual(
      new Set(['Person 0', 'Person 1', 'Person 2', 'Person 3'])
    )
    expect(people[0].accounts[0].sourceInstance).toBe('guild1')

    const sessions = await (await client.ingestSessions()).toArray()
    expect(new Set(sessions.map((s) => s.displayName))).toEqual(
      new Set(['channel 0', 'channel 1', 'channel 2'])
    )

    const excluded = people[0].peerId
    await (await client.peer(excluded)).setConfiguration({ observeMe: false })
    expect((await client.ingestStatus()).peopleNotModeled).toBe(1)
    const after = await (await client.ingestPeople()).toArray()
    expect(after.filter((p) => !p.modeled).map((p) => p.peerId)).toEqual([
      excluded,
    ])
  })

  test('link and unlink wait for a reset; reset shows in status at once', async () => {
    const client = sdk(generateWorkspaceId('ingest-link'), true)
    await client.ingest(records(8), { derive: 'later' })
    const people = await (await client.ingestPeople()).toArray()
    const [keep, merge] = people.map((p) => p.peerId)

    // Already imported: refused until a reset (the deriver runs the deletion).
    await expect(client.ingestLink([keep, merge])).rejects.toBeInstanceOf(
      ConflictError
    )
    await expect(
      client.ingestUnlink({
        peer: keep,
        account: { source: 'discord', sourceInstance: 'guild1', id: 'user-0' },
      })
    ).rejects.toBeInstanceOf(ConflictError)

    const reset = await client.ingestReset()
    expect(reset.sessions.length).toBe(3)
    const status = await client.ingestStatus()
    expect(status.records).toEqual({ deferred: 0, pending: 0, queued: 0 })
    expect(status.sessions).toBe(0)
  })

  test('a suggestions refresh is queued once and shows in status', async () => {
    const client = sdk(generateWorkspaceId('ingest-suggest'), true)
    await client.ingest(records(4), { derive: 'later' })

    // The deriver runs the refresh; this test server has none, so it waits.
    expect(await client.ingestRefreshSuggestions()).toBe(true)
    expect(await client.ingestRefreshSuggestions()).toBe(false)
    const status = await client.ingestStatus()
    expect(status.suggestions).toEqual({
      rules: { high: 0, low: 0 },
      refreshing: true,
    })
    expect((await client.ingestSuggestions()).items).toEqual([])
    await expect(
      client.ingestAcceptSuggestion('missing')
    ).rejects.toBeInstanceOf(NotFoundError)
    await expect(
      client.ingestDismissSuggestion('missing')
    ).rejects.toBeInstanceOf(NotFoundError)
  })

  test('a delete in the dry run removes the record for good', async () => {
    const client = sdk(generateWorkspaceId('ingest-delete'), true)
    await client.ingest(records(2), { derive: 'later' })

    const deleted = await client.ingest([
      {
        op: 'delete',
        source: 'discord',
        sourceInstance: 'guild1',
        sourceId: 'msg-0',
      },
    ])
    expect(deleted.records[0].outcome).toBe('deleted')
    const again = await client.ingest(
      (function* () {
        yield* records(1)
      })(),
      { derive: 'auto' }
    )
    expect(again.records[0].reason).toBe('deleted')
  })

  test('a sync state is saved, guarded by its revision, and forgotten', async () => {
    const client = sdk(generateWorkspaceId('ingest-sync'), true)
    await client.ingest(records(1), { derive: 'later' })
    expect(await client.ingestSyncState('discord', 'guild1')).toBeNull()

    const first = await client.ingestSaveSyncState('discord', 'guild1', {
      'chan-0': 'msg-0',
    })
    const second = await client.ingestSaveSyncState(
      'discord',
      'guild1',
      { 'chan-0': 'msg-1' },
      {
        revision: first.revision,
        lastRecordAt: new Date(Date.UTC(2026, 10, 4)),
      }
    )
    await expect(
      client.ingestSaveSyncState(
        'discord',
        'guild1',
        {},
        { revision: first.revision }
      )
    ).rejects.toBeInstanceOf(ConflictError)

    const state = await client.ingestSyncState('discord', 'guild1')
    expect(state?.position).toEqual({ 'chan-0': 'msg-1' })
    expect(state?.revision).toBe(second.revision)
    const status = await client.ingestStatus()
    expect(status.sources.map((s) => s.sourceInstance)).toEqual(['guild1'])
    expect(await client.ingestForgetSyncState('discord', 'guild1')).toBe(true)
    expect(await client.ingestForgetSyncState('discord', 'guild1')).toBe(false)
  })
})

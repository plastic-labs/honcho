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
})

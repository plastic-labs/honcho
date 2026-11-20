import { describe, expect, test } from 'bun:test'
import type { HonchoHTTPClient } from '../src/http/client'
import { Honcho } from '../src'
import { Page } from '../src/pagination'
import { Session } from '../src/session'
import type {
  CursorPageResponse,
  MessageResponse,
  PageResponse,
  TotalKind,
} from '../src/types/api'

interface Item {
  id: string
}

function offsetPage(
  page: number,
  pages: number,
  extra: Partial<PageResponse<Item>> = {}
): PageResponse<Item> {
  return {
    items: [{ id: `p${page}` }],
    page,
    size: 1,
    total: pages,
    pages,
    ...extra,
  }
}

async function collect<T>(page: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = []
  for await (const item of page) {
    items.push(item)
  }
  return items
}

describe('Page iteration', () => {
  test('follows next_cursor, then each next_page, keeping the transform', async () => {
    const cursorPages: Record<string, CursorPageResponse<Item>> = {
      c1: { items: [{ id: 'b' }], next_page: 'c2' },
      c2: { items: [{ id: 'c' }], next_page: null },
    }
    const cursorCalls: [string, number][] = []
    const page = new Page<string, Item>(
      { ...offsetPage(1, 9, { next_cursor: 'c1' }), items: [{ id: 'a' }] },
      (item) => item.id.toUpperCase(),
      async () => {
        throw new Error('walked by page number past a page with next_cursor')
      },
      async (cursor, size) => {
        cursorCalls.push([cursor, size])
        return cursorPages[cursor]
      }
    )

    expect(await collect(page)).toEqual(['A', 'B', 'C'])
    expect(cursorCalls).toEqual([
      ['c1', 1],
      ['c2', 1],
    ])
  })

  test('a page that ends the list makes no further request', async () => {
    const page = new Page<Item>(
      offsetPage(1, 1, { next_cursor: null }),
      undefined,
      async () => {
        throw new Error('requested past the end')
      },
      async () => {
        throw new Error('requested past the end')
      }
    )

    expect((await collect(page)).map((item) => item.id)).toEqual(['p1'])
  })

  test('a server without next_cursor is walked by page number', async () => {
    const page = new Page<Item>(
      offsetPage(1, 3),
      undefined,
      async (next) => offsetPage(next, 3),
      async () => {
        throw new Error('followed a cursor the server never sent')
      }
    )

    expect((await collect(page)).map((item) => item.id)).toEqual([
      'p1',
      'p2',
      'p3',
    ])
    expect(page.nextCursor).toBeNull()
  })

  test('getNextPage stays in offset mode', async () => {
    const requested: number[] = []
    const page = new Page<Item>(
      offsetPage(1, 3, { next_cursor: 'c1' }),
      undefined,
      async (next) => {
        requested.push(next)
        return offsetPage(next, 3, { next_cursor: 'c2' })
      },
      async () => {
        throw new Error('getNextPage followed a cursor')
      }
    )

    const second = await page.getNextPage()

    expect(second?.page).toBe(2)
    expect(requested).toEqual([2])
  })

  test('an offset answer to a cursor request is an error', async () => {
    const page = new Page<Item>(
      offsetPage(1, 2, { next_cursor: 'c1' }),
      undefined,
      undefined,
      async () => offsetPage(1, 2)
    )

    await expect(collect(page)).rejects.toThrow('cursor-mode')
  })
})

describe('Page accessors', () => {
  test.each<[Partial<PageResponse<Item>>, TotalKind]>([
    [{ total: 5 }, 'exact'],
    [{ total: 100_000 }, 'exact'],
    [{ total: 100_001 }, 'at_least'],
    [{ total: 250_000, total_kind: 'estimate' }, 'estimate'],
    [{ total: 4, total_kind: 'at_most' }, 'at_most'],
    [{ total: 7, total_kind: 'at_least' }, 'at_least'],
    [{ total: 250_000, total_kind: 'exact' }, 'exact'],
  ])('totalKind for %p is %p', (fields, expected) => {
    const page = new Page<Item>({ ...offsetPage(1, 1), ...fields })
    expect(page.totalKind).toBe(expected)
  })

  test('nextCursor returns the server value', () => {
    expect(
      new Page<Item>(offsetPage(1, 2, { next_cursor: 'c1' })).nextCursor
    ).toBe('c1')
  })
})

describe('list methods read on in cursor mode', () => {
  test('session.messages repeats its filters and ordering with the cursor', async () => {
    const message = (id: string): MessageResponse => ({
      id,
      content: id,
      peer_id: 'peer-1',
      session_id: 'session-1',
      workspace_id: 'workspace-1',
      metadata: {},
      created_at: '2024-01-01T00:00:00Z',
      token_count: 1,
    })
    const queries: Record<string, unknown>[] = []
    const bodies: unknown[] = []
    const http = {
      post: async (
        _path: string,
        options?: { body?: unknown; query?: Record<string, unknown> }
      ) => {
        queries.push(options?.query ?? {})
        bodies.push(options?.body)
        if (queries.length === 1) {
          return {
            items: [message('m1'), message('m2')],
            page: 1,
            size: 2,
            total: 3,
            pages: 2,
            total_kind: 'exact',
            next_cursor: 'c1',
          }
        }
        return { items: [message('m3')], next_page: null }
      },
    } as unknown as HonchoHTTPClient
    const session = new Session(
      'session-1',
      'workspace-1',
      http,
      undefined,
      undefined,
      async () => {}
    )

    const page = await session.messages({
      filters: { peer_id: 'peer-1' },
      size: 2,
      reverse: true,
    })
    const ids = (await collect(page)).map((m) => m.id)

    expect(ids).toEqual(['m1', 'm2', 'm3'])
    expect(page.totalKind).toBe('exact')
    expect(queries[1]).toEqual({
      page: undefined,
      size: 2,
      cursor: 'c1',
      reverse: 'true',
    })
    expect(bodies[1]).toEqual(bodies[0])
  })
})

describe('ingest lists read on in cursor mode', () => {
  test('ingestPeople repeats its GET with the cursor', async () => {
    const person = (peerId: string) => ({
      peer_id: peerId,
      accounts: [],
      messages: 1,
      modeled: true,
    })
    const queries: Record<string, unknown>[] = []
    const http = {
      get: async (
        path: string,
        options?: { query?: Record<string, unknown> }
      ) => {
        expect(path.endsWith('/ingest/people')).toBe(true)
        queries.push(options?.query ?? {})
        if (queries.length === 1) {
          return {
            items: [person('a'), person('b')],
            page: 1,
            size: 2,
            total: 3,
            pages: 2,
            total_kind: 'exact',
            next_cursor: 'c1',
          }
        }
        return { items: [person('c')], next_page: null }
      },
    } as unknown as HonchoHTTPClient
    const client = new Honcho({ workspaceId: 'workspace-1', apiKey: 'test' })
    ;(client as unknown as { _http: HonchoHTTPClient })._http = http

    const page = await client.ingestPeople({ size: 2 })
    const ids = (await collect(page)).map((p) => p.peerId)

    expect(ids).toEqual(['a', 'b', 'c'])
    expect(queries).toEqual([
      { page: undefined, size: 2 },
      { cursor: 'c1', size: 2 },
    ])
  })
})

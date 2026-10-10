import { describe, expect, spyOn, test } from 'bun:test'
import { Honcho } from '../src/client'
import { HonchoHTTPClient } from '../src/http/client'
import { CursorPage } from '../src/pagination'

const item = (id: string) => ({
  id,
  task_type: 'dream',
  workspace_id: 'progress',
  session_id: 'session',
  observer: 'alice',
  observers: [],
  observed: 'bob',
  started_at: null,
  stage: 'deduction',
  last_activity_at: null,
})

describe('active queue work cursor contract', () => {
  test('preserves filters and transforms every page, including an empty tail', async () => {
    const create = spyOn(HonchoHTTPClient.prototype, 'post').mockResolvedValue({
      id: 'progress',
    })
    const get = spyOn(HonchoHTTPClient.prototype, 'get')
      .mockResolvedValueOnce({ items: [item('a')], next_page: 'cursor-a' })
      .mockResolvedValueOnce({ items: [item('b')], next_page: 'cursor-b' })
      .mockResolvedValueOnce({ items: [], next_page: null })
    try {
      const client = new Honcho({
        workspaceId: 'progress',
        apiKey: 'test',
        baseURL: 'http://localhost',
      })
      const page = await client.queueWorkUnits({
        taskType: 'dream',
        observer: 'alice',
        sender: 'bob',
        session: 'session',
        size: 1,
      })
      expect(page).toBeInstanceOf(CursorPage)
      expect(page.items[0].startedAt).toBeNull()
      const ids: string[] = []
      for await (const work of page) {
        ids.push(work.id)
        expect(work.taskType).toBe('dream')
        expect(work.workspaceId).toBe('progress')
        expect(work.sessionId).toBe('session')
      }
      expect(ids).toEqual(['a', 'b'])
      expect(get.mock.calls.map((call) => call[1]?.query?.cursor)).toEqual([
        '',
        'cursor-a',
        'cursor-b',
      ])
      for (const call of get.mock.calls) {
        expect(call[1]?.query).toMatchObject({
          size: 1,
          task_type: 'dream',
          observer_id: 'alice',
          sender_id: 'bob',
          session_id: 'session',
        })
      }
    } finally {
      create.mockRestore()
      get.mockRestore()
    }
  })

  test('rejects an unbounded page size before making any request', async () => {
    const client = new Honcho({
      workspaceId: 'progress',
      apiKey: 'test',
      baseURL: 'http://localhost',
    })
    for (const size of [0, 101, 1.5]) {
      await expect(client.queueWorkUnits({ size })).rejects.toThrow(RangeError)
    }
  })
})

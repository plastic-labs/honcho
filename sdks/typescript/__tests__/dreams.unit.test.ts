/**
 * Representation History Unit Tests
 *
 * `dreams()` and `getDream()` against a mocked HTTP client, covering the wire
 * shape: the list filter body, the paths, and the pass-through response types.
 */

import { describe, expect, test } from 'bun:test'
import { Honcho } from '../src'
import type { HonchoHTTPClient } from '../src/http/client'
import type { DreamRun, DreamRunDetail, PageResponse } from '../src/types/api'

const RUN: DreamRun = {
  id: 'run-1',
  observer_id: 'maya',
  observed_id: 'maya',
  dream_type: 'omni',
  trigger_reason: 'manual',
  status: 'completed',
  created_count: 1,
  removed_count: 1,
  peer_card_updated: false,
  started_at: '2026-11-03T00:00:00Z',
  finished_at: '2026-11-03T00:01:00Z',
  pruned_at: null,
}

function honchoWith(response: unknown) {
  const calls: { method: string; path: string; body?: unknown }[] = []
  const http = {
    post: async (path: string, options?: { body?: unknown }) => {
      calls.push({ method: 'POST', path, body: options?.body })
      return response
    },
    get: async (path: string) => {
      calls.push({ method: 'GET', path })
      return response
    },
  } as unknown as HonchoHTTPClient
  const honcho = new Honcho({ apiKey: 'test', workspaceId: 'ws' })
  ;(honcho as unknown as { _http: HonchoHTTPClient })._http = http
  ;(
    honcho as unknown as { _ensureWorkspace: () => Promise<void> }
  )._ensureWorkspace = async () => undefined
  return { honcho, calls }
}

describe('dreams', () => {
  test('lists runs filtered by observed and observer', async () => {
    const page: PageResponse<DreamRun> = {
      items: [RUN],
      total: 1,
      page: 1,
      size: 50,
      pages: 1,
    }
    const { honcho, calls } = honchoWith(page)

    const result = await honcho.dreams({ observed: 'maya', observer: 'maya' })

    expect(calls).toEqual([
      {
        method: 'POST',
        path: '/v3/workspaces/ws/dreams/list',
        body: { observed_id: 'maya', observer_id: 'maya' },
      },
    ])
    expect(result.items).toEqual([RUN])
  })

  test('getDream reads one run by id', async () => {
    const detail: DreamRunDetail = {
      ...RUN,
      created: [
        {
          conclusion: {
            id: 'c-new',
            content: 'Maya relies on caffeine at night',
            level: 'deductive',
            status: 'current',
            removed_in_run: null,
          },
          sources: [],
          specialist: 'deduction',
        },
      ],
      removed: [],
      peer_card: null,
    }
    const { honcho, calls } = honchoWith(detail)

    const result = await honcho.getDream('run-1')

    expect(calls).toEqual([
      { method: 'GET', path: '/v3/workspaces/ws/dreams/run-1' },
    ])
    expect(result).toEqual(detail)
  })
})

import { describe, expect, test } from 'bun:test'
import { ZodError } from 'zod'
import type { HonchoHTTPClient } from '../src/http/client'
import { Peer } from '../src/peer'
import { Scope } from '../src/scope'
import { Session } from '../src/session'
import type { ScopeStatusResponse } from '../src/types/api'

/**
 * Capture the body of the single request a call makes, so the wire shape the
 * server actually receives is asserted rather than the SDK's own options.
 */
function capturingHttp(response: unknown): {
  http: HonchoHTTPClient
  body: () => Record<string, unknown> | undefined
  query: () => Record<string, unknown> | undefined
  path: () => string | undefined
} {
  let capturedBody: Record<string, unknown> | undefined
  let capturedQuery: Record<string, unknown> | undefined
  let capturedPath: string | undefined
  const http = {
    post: async (
      path: string,
      options?: { body?: Record<string, unknown> }
    ) => {
      capturedPath = path
      capturedBody = options?.body
      return response
    },
    get: async (
      path: string,
      options?: { query?: Record<string, unknown> }
    ) => {
      capturedPath = path
      capturedQuery = options?.query
      return response
    },
    delete: async (path: string) => {
      capturedPath = path
      return undefined
    },
  } as unknown as HonchoHTTPClient
  return {
    http,
    body: () => capturedBody,
    query: () => capturedQuery,
    path: () => capturedPath,
  }
}

describe('sessions allowlist sugar', () => {
  test('chat sends `sessions` as a session_id filter, not a bare field', async () => {
    const { http, body } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await peer.chat('what happened?', {
      sessions: ['session-a', new Session('session-b', 'workspace-1', http)],
    })

    expect(body()).toMatchObject({
      filters: { session_id: ['session-a', 'session-b'] },
    })
    // The sugar must not leak through as its own wire field — the server would
    // reject an unknown key.
    expect(body()).not.toHaveProperty('sessions')
  })

  test('representation sends `sessions` as a session_id filter', async () => {
    const { http, body } = capturingHttp({ representation: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await peer.representation({ sessions: ['session-a'] })

    expect(body()).toMatchObject({ filters: { session_id: ['session-a'] } })
  })

  test('an empty allowlist is rejected rather than silently recalling nothing', async () => {
    const { http } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await expect(peer.chat('q', { sessions: [] })).rejects.toBeInstanceOf(
      ZodError
    )
  })
})

describe('scope read option', () => {
  test('a single scope passes through as `scope`', async () => {
    const { http, body } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await peer.chat('q', { scope: 'therapy' })

    expect(body()).toMatchObject({ scope: 'therapy' })
  })

  test('a Scope object resolves to its id', async () => {
    const { http, body } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await peer.chat('q', { scope: new Scope('therapy', 'workspace-1', http) })

    expect(body()).toMatchObject({ scope: 'therapy' })
  })

  test('a list of scopes passes through as a list', async () => {
    const { http, body } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await peer.chat('q', { scope: ['therapy', 'work'] })

    expect(body()).toMatchObject({ scope: ['therapy', 'work'] })
  })

  test('scope and sessions are rejected together', async () => {
    const { http } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await expect(
      peer.chat('q', { scope: 'therapy', sessions: ['session-a'] })
    ).rejects.toBeInstanceOf(ZodError)
  })

  test('scope and a single session are rejected together', async () => {
    const { http } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await expect(
      peer.chat('q', { scope: 'therapy', session: 'session-a' })
    ).rejects.toBeInstanceOf(ZodError)
  })
})

describe('scope id validation', () => {
  test('a prefixed name reports the reserved prefix, not the charset', async () => {
    const { http } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    // 'scope.therapy' fails both rules; the prefix message is the useful one,
    // so it must be the only one raised.
    const error = await peer
      .chat('q', { scope: 'scope.therapy' })
      .then(() => undefined)
      .catch((err: unknown) => err as ZodError)

    expect(error).toBeInstanceOf(ZodError)
    const messages = (error as ZodError).issues.map((issue) => issue.message)
    expect(messages.some((m) => m.includes('reserved prefix'))).toBe(true)
    expect(messages.some((m) => m.includes('may only contain'))).toBe(false)
  })

  test('a name with illegal characters is rejected', async () => {
    const { http } = capturingHttp({ content: 'ok' })
    const peer = new Peer('alice', 'workspace-1', http)

    await expect(peer.chat('q', { scope: 'my scope' })).rejects.toBeInstanceOf(
      ZodError
    )
  })
})

describe('session.context scoping', () => {
  const contextResponse = {
    id: 'session-a',
    messages: [],
    summary: null,
    peer_representation: null,
    peer_card: null,
  }

  test('scope and sessions are sent as their own query params', async () => {
    const { http, query } = capturingHttp(contextResponse)
    const session = new Session('session-a', 'workspace-1', http)

    await session.context({
      peerTarget: 'user',
      sessions: ['session-a', 'session-b'],
    })

    // An array here relies on the HTTP client emitting repeated params; see
    // http-client.test.ts.
    expect(query()).toMatchObject({ sessions: ['session-a', 'session-b'] })
  })

  test('sessions without peerTarget is rejected, not silently ignored', async () => {
    const { http } = capturingHttp(contextResponse)
    const session = new Session('session-a', 'workspace-1', http)

    await expect(
      session.context({ sessions: ['session-a'] })
    ).rejects.toBeInstanceOf(ZodError)
  })

  test('sessions and scope are rejected together', async () => {
    const { http } = capturingHttp(contextResponse)
    const session = new Session('session-a', 'workspace-1', http)

    await expect(
      session.context({
        peerTarget: 'user',
        scope: 'therapy',
        sessions: ['session-a'],
      })
    ).rejects.toBeInstanceOf(ZodError)
  })

  test('sessions and limitToSession are rejected together', async () => {
    const { http } = capturingHttp(contextResponse)
    const session = new Session('session-a', 'workspace-1', http)

    await expect(
      session.context({
        peerTarget: 'user',
        limitToSession: true,
        sessions: ['session-a'],
      })
    ).rejects.toBeInstanceOf(ZodError)
  })

  test('scope and peerPerspective are rejected together', async () => {
    const { http } = capturingHttp(contextResponse)
    const session = new Session('session-a', 'workspace-1', http)

    await expect(
      session.context({
        peerTarget: 'user',
        peerPerspective: 'assistant',
        scope: 'therapy',
      })
    ).rejects.toBeInstanceOf(ZodError)
  })
})

describe('Scope membership and status', () => {
  test('addSessions posts session_ids and resolves Session objects', async () => {
    const { http, body, path } = capturingHttp(undefined)
    const scope = new Scope('therapy', 'workspace-1', http)

    await scope.addSessions([
      'session-a',
      new Session('session-b', 'workspace-1', http),
    ])

    expect(path()).toBe('/v3/workspaces/workspace-1/scopes/therapy/sessions')
    expect(body()).toEqual({ session_ids: ['session-a', 'session-b'] })
  })

  test('addSessions rejects a batch over the server limit instead of chunking', async () => {
    const { http } = capturingHttp(undefined)
    const scope = new Scope('therapy', 'workspace-1', http)

    const tooMany = Array.from({ length: 101 }, (_, i) => `session-${i}`)

    await expect(scope.addSessions(tooMany)).rejects.toBeInstanceOf(ZodError)
  })

  test('removeSession targets the session subpath', async () => {
    const { http, path } = capturingHttp(undefined)
    const scope = new Scope('therapy', 'workspace-1', http)

    await scope.removeSession(new Session('session-b', 'workspace-1', http))

    expect(path()).toBe(
      '/v3/workspaces/workspace-1/scopes/therapy/sessions/session-b'
    )
  })

  test('status maps snake_case job fields to camelCase', async () => {
    const response: ScopeStatusResponse = {
      backfill_status: {
        'session-a': {
          state: 'completed',
          updated_at: '2024-01-01T00:00:00Z',
          docs_copied: 12,
        },
        'session-b': { state: 'pending', updated_at: '2024-01-02T00:00:00Z' },
      },
    }
    const { http } = capturingHttp(response)
    const scope = new Scope('therapy', 'workspace-1', http)

    const status = await scope.status()

    expect(status.backfillStatus['session-a']).toEqual({
      state: 'completed',
      updatedAt: '2024-01-01T00:00:00Z',
      docsCopied: 12,
    })
    expect(status.backfillStatus['session-b']?.docsCopied).toBeUndefined()
  })

  test('status on a scope with no backfill is an empty map, not a throw', async () => {
    // The server omits the key entirely when nothing was ever enqueued.
    const { http } = capturingHttp({} as ScopeStatusResponse)
    const scope = new Scope('therapy', 'workspace-1', http)

    const status = await scope.status()

    expect(status.backfillStatus).toEqual({})
  })
})

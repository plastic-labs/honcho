import { describe, expect, test } from 'bun:test'
import { ZodError } from 'zod'
import type { HonchoHTTPClient } from '../src/http/client'
import { Session } from '../src/session'
import type { SessionContextResponse, SessionResponse } from '../src/types/api'

function createSessionResponse(
  overrides: Partial<SessionResponse> = {}
): SessionResponse {
  return {
    id: 'session-1',
    workspace_id: 'workspace-1',
    is_active: true,
    metadata: {},
    configuration: {},
    created_at: '2024-01-01T00:00:00Z',
    ...overrides,
  }
}

function createSessionContextResponse(): SessionContextResponse {
  return {
    id: 'session-1',
    messages: [],
    summary: null,
    peer_representation: null,
    peer_card: null,
  }
}

describe('Session unit behavior', () => {
  test('getMetadata populates createdAt and isActive from fetched session data', async () => {
    const http = {
      post: async () =>
        createSessionResponse({
          metadata: { topic: 'testing' },
          created_at: '2024-02-01T12:00:00Z',
          is_active: true,
        }),
    } as unknown as HonchoHTTPClient

    const session = new Session('session-1', 'workspace-1', http)

    expect(session.createdAt).toBeUndefined()
    expect(session.isActive).toBeUndefined()

    const metadata = await session.getMetadata()

    expect(metadata).toEqual({ topic: 'testing' })
    expect(session.createdAt).toBe('2024-02-01T12:00:00Z')
    expect(session.isActive).toBe(true)
  })

  test('refresh updates cached createdAt and isActive from the latest fetch', async () => {
    const responses = [
      createSessionResponse({
        metadata: { version: 1 },
        configuration: { reasoning: { enabled: true } },
        created_at: '2024-01-01T00:00:00Z',
        is_active: true,
      }),
      createSessionResponse({
        metadata: { version: 2 },
        configuration: { reasoning: { enabled: false } },
        created_at: '2024-01-02T00:00:00Z',
        is_active: false,
      }),
    ]

    const http = {
      post: async () => responses.shift() ?? createSessionResponse(),
    } as unknown as HonchoHTTPClient

    const session = new Session('session-1', 'workspace-1', http)

    await session.getMetadata()
    await session.refresh()

    expect(session.metadata).toEqual({ version: 2 })
    expect(session.configuration).toMatchObject({
      reasoning: { enabled: false },
    })
    expect(session.createdAt).toBe('2024-01-02T00:00:00Z')
    expect(session.isActive).toBe(false)
  })

  test('uploadFile converts configuration to API field names', async () => {
    let capturedFormData: FormData | undefined

    const http = {
      upload: async (_path: string, formData: FormData) => {
        capturedFormData = formData
        return []
      },
    } as unknown as HonchoHTTPClient

    const session = new Session('session-1', 'workspace-1', http)

    await session.uploadFile(new Blob(['hello'], { type: 'text/plain' }), 'peer-1', {
      configuration: {
        reasoning: {
          enabled: true,
          customInstructions: 'Keep headings intact',
        },
      },
    })

    expect(capturedFormData).toBeDefined()

    const rawConfiguration = capturedFormData?.get('configuration')
    expect(typeof rawConfiguration).toBe('string')
    expect(JSON.parse(rawConfiguration as string)).toEqual({
      reasoning: {
        enabled: true,
        custom_instructions: 'Keep headings intact',
      },
    })
  })

  test('context rejects invalid content-like searchQuery inputs', async () => {
    const http = {
      get: async () => createSessionContextResponse(),
    } as unknown as HonchoHTTPClient

    const session = new Session('session-1', 'workspace-1', http)

    await expect(
      session.context({
        peerTarget: 'peer-1',
        representationOptions: {
          searchQuery: { foo: 'bar' } as never,
        },
      })
    ).rejects.toBeInstanceOf(ZodError)
  })

  test('representation rejects whitespace-only content-like searchQuery inputs', async () => {
    const http = {
      post: async () => ({ representation: 'ok' }),
    } as unknown as HonchoHTTPClient

    const session = new Session('session-1', 'workspace-1', http)

    await expect(
      session.representation('peer-1', {
        searchQuery: { content: '   ' } as never,
      })
    ).rejects.toBeInstanceOf(ZodError)
  })
})

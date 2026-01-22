/**
 * Error Handling Tests
 *
 * Tests for SDK error handling and error types.
 *
 * This file is split into:
 * - Unit tests: Test error classes directly (no server required)
 * - Integration tests: Test error scenarios against live server
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { Honcho, Peer } from '../src'
import {
  HonchoError,
  BadRequestError,
  AuthenticationError,
  NotFoundError,
  PermissionDeniedError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ConnectionError,
  createErrorFromResponse,
} from '../src/http/errors'
import {
  PeerIdSchema,
  SessionIdSchema,
  SearchQuerySchema,
  LimitSchema,
} from '../src/validation'
import { createTestClient, requireServer, TEST_CONFIG } from './setup'

// =============================================================================
// Unit Tests (no server required)
// =============================================================================

describe('Error types (unit)', () => {
  test('HonchoError is base class', () => {
    const error = new HonchoError('Test error', 500)

    expect(error).toBeInstanceOf(Error)
    expect(error).toBeInstanceOf(HonchoError)
    expect(error.message).toBe('Test error')
    expect(error.status).toBe(500)
    expect(error.name).toBe('HonchoError')
  })

  test('BadRequestError has status 400', () => {
    const error = new BadRequestError('Invalid input', { field: 'name' })

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(400)
    expect(error.code).toBe('bad_request')
    expect(error.body).toEqual({ field: 'name' })
  })

  test('AuthenticationError has status 401', () => {
    const error = new AuthenticationError('Invalid token')

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(401)
    expect(error.code).toBe('authentication_error')
  })

  test('PermissionDeniedError has status 403', () => {
    const error = new PermissionDeniedError('Access denied')

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(403)
    expect(error.code).toBe('permission_denied')
  })

  test('NotFoundError has status 404', () => {
    const error = new NotFoundError('Workspace not found')

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(404)
    expect(error.code).toBe('not_found')
  })

  test('RateLimitError has status 429', () => {
    const error = new RateLimitError('Too many requests', 5000)

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(429)
    expect(error.code).toBe('rate_limit_exceeded')
    expect(error.retryAfter).toBe(5000)
  })

  test('ServerError has status 5xx', () => {
    const error = new ServerError('Internal error', 503)

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(503)
    expect(error.code).toBe('server_error')
  })

  test('TimeoutError has status 0', () => {
    const error = new TimeoutError('Request timed out')

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(0)
    expect(error.code).toBe('timeout')
  })

  test('ConnectionError has status 0', () => {
    const error = new ConnectionError('Network error')

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(0)
    expect(error.code).toBe('connection_error')
  })
})

describe('createErrorFromResponse (unit)', () => {
  test('400 creates BadRequestError', () => {
    const error = createErrorFromResponse(400, 'Bad request', { field: 'x' })

    expect(error).toBeInstanceOf(BadRequestError)
    expect(error.body).toEqual({ field: 'x' })
  })

  test('401 creates AuthenticationError', () => {
    const error = createErrorFromResponse(401, 'Unauthorized')

    expect(error).toBeInstanceOf(AuthenticationError)
  })

  test('403 creates PermissionDeniedError', () => {
    const error = createErrorFromResponse(403, 'Forbidden')

    expect(error).toBeInstanceOf(PermissionDeniedError)
  })

  test('404 creates NotFoundError', () => {
    const error = createErrorFromResponse(404, 'Not found')

    expect(error).toBeInstanceOf(NotFoundError)
  })

  test('429 creates RateLimitError with retryAfter', () => {
    const error = createErrorFromResponse(429, 'Rate limited', {}, 3000)

    expect(error).toBeInstanceOf(RateLimitError)
    expect((error as RateLimitError).retryAfter).toBe(3000)
  })

  test('500 creates ServerError', () => {
    const error = createErrorFromResponse(500, 'Server error')

    expect(error).toBeInstanceOf(ServerError)
  })

  test('503 creates ServerError', () => {
    const error = createErrorFromResponse(503, 'Service unavailable')

    expect(error).toBeInstanceOf(ServerError)
    expect(error.status).toBe(503)
  })

  test('unknown status creates HonchoError', () => {
    const error = createErrorFromResponse(418, "I'm a teapot")

    expect(error).toBeInstanceOf(HonchoError)
    expect(error.status).toBe(418)
  })
})

describe('Error messages (unit)', () => {
  test('HonchoError includes status in output', () => {
    const error = new HonchoError('Something went wrong', 500)

    expect(error.message).toBe('Something went wrong')
    expect(error.status).toBe(500)
  })

  test('BadRequestError preserves body', () => {
    const body = {
      detail: [
        { loc: ['body', 'name'], msg: 'field required', type: 'value_error' },
      ],
    }
    const error = new BadRequestError('Validation failed', body)

    expect(error.body).toEqual(body)
  })

  test('default error messages are sensible', () => {
    expect(new AuthenticationError().message).toBe('Authentication failed')
    expect(new PermissionDeniedError().message).toBe('Permission denied')
    expect(new NotFoundError().message).toBe('Resource not found')
    expect(new RateLimitError().message).toBe('Rate limit exceeded')
    expect(new ServerError().message).toBe('Server error')
    expect(new TimeoutError().message).toBe('Request timed out')
    expect(new ConnectionError().message).toBe('Connection failed')
  })

  test('instanceof checks work correctly', () => {
    const validation = new BadRequestError('test')
    const auth = new AuthenticationError('test')
    const notFound = new NotFoundError('test')

    // All are HonchoError
    expect(validation instanceof HonchoError).toBe(true)
    expect(auth instanceof HonchoError).toBe(true)
    expect(notFound instanceof HonchoError).toBe(true)

    // But distinct types
    expect(validation instanceof AuthenticationError).toBe(false)
    expect(auth instanceof NotFoundError).toBe(false)
  })
})

describe('Client-side validation (unit)', () => {
  // Test Zod schemas directly to avoid Honcho constructor side effects

  test('empty peer ID throws', () => {
    expect(() => PeerIdSchema.parse('')).toThrow()
  })

  test('valid peer ID passes', () => {
    expect(PeerIdSchema.parse('valid-peer')).toBe('valid-peer')
  })

  test('empty session ID throws', () => {
    expect(() => SessionIdSchema.parse('')).toThrow()
  })

  test('valid session ID passes', () => {
    expect(SessionIdSchema.parse('valid-session')).toBe('valid-session')
  })

  test('empty search query throws', () => {
    expect(() => SearchQuerySchema.parse('')).toThrow()
  })

  test('valid search query passes', () => {
    expect(SearchQuerySchema.parse('find this')).toBe('find this')
  })

  test('limit of 0 throws', () => {
    expect(() => LimitSchema.parse(0)).toThrow()
  })

  test('limit over 100 throws', () => {
    expect(() => LimitSchema.parse(101)).toThrow()
  })

  test('valid limit passes', () => {
    expect(LimitSchema.parse(50)).toBe(50)
  })

  test('peer card with empty string throws', async () => {
    // Create a peer directly without HTTP client to test validation
    const peer = new Peer('test-peer', 'workspace', {} as never)

    // Zod validation requires non-empty string (PeerIdSchema)
    await expect(peer.card('')).rejects.toThrow()
  })

  test('peer card with invalid target type throws', async () => {
    const peer = new Peer('test-peer', 'workspace', {} as never)

    // Zod throws on invalid type
    await expect(peer.card(123 as never)).rejects.toThrow()
  })
})

describe('Connection scenarios (unit)', () => {
  test('unreachable server throws error', async () => {
    // Note: Creating Honcho fires a background request, so we test
    // the HTTP client directly to avoid side effects
    const { HonchoHTTPClient } = await import('../src/http/client')

    const httpClient = new HonchoHTTPClient({
      baseURL: 'http://localhost:59999',
      timeout: 1000,
      maxRetries: 0,
    })

    await expect(httpClient.get('/test')).rejects.toThrow()
  })
})

// =============================================================================
// Integration Tests (require server)
// =============================================================================

describe('Error Handling (integration)', () => {
  let client: Honcho
  let cleanup: () => Promise<void>

  beforeAll(async () => {
    await requireServer()
    const setup = await createTestClient('errors')
    client = setup.client
    cleanup = setup.cleanup
  })

  afterAll(async () => {
    await cleanup()
  })

  test('invalid API key returns 401 (if auth enabled)', async () => {
    const badClient = new Honcho({
      baseURL: TEST_CONFIG.baseURL,
      apiKey: 'invalid-key-12345',
      workspaceId: 'test-bad-auth',
    })

    try {
      await badClient.getMetadata()
      // If we get here, auth is disabled on server - that's OK
    } catch (error) {
      if (error instanceof AuthenticationError) {
        expect(error.status).toBe(401)
      }
    }
  })

  test('operations work with valid client', async () => {
    // Basic sanity check that the client works
    const metadata = await client.getMetadata()
    expect(typeof metadata).toBe('object')
  })
})

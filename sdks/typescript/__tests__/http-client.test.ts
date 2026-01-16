/**
 * HTTP Client Unit Tests
 *
 * Tests for the standalone HTTP client with:
 * - Constructor configuration
 * - URL and header building
 * - Retry logic with exponential backoff
 * - Timeout handling
 * - Error response parsing
 * - Streaming support
 *
 * These tests mock fetch to verify client behavior without a server.
 */

import { describe, test, expect, beforeEach, afterEach, mock, spyOn } from 'bun:test'
import { HonchoHTTPClient } from '../src/http/client'
import {
  HonchoError,
  BadRequestError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ConnectionError,
} from '../src/http/errors'

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a mock Response object
 */
function mockResponse(
  body: unknown,
  options: {
    status?: number
    ok?: boolean
    headers?: Record<string, string>
  } = {}
): Response {
  const { status = 200, ok = status >= 200 && status < 300, headers = {} } = options

  const responseHeaders = new Headers(headers)
  const bodyString = typeof body === 'string' ? body : JSON.stringify(body)

  return new Response(bodyString, {
    status,
    headers: responseHeaders,
  })
}

/**
 * Create a mock Response that throws on text()/json()
 */
function mockErrorResponse(
  status: number,
  errorBody: unknown,
  headers: Record<string, string> = {}
): Response {
  return mockResponse(errorBody, { status, headers })
}

// =============================================================================
// Constructor Tests
// =============================================================================

describe('HonchoHTTPClient constructor', () => {
  test('normalizes baseURL by removing trailing slash', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com/',
    })

    expect(client.baseURL).toBe('https://api.example.com')
  })

  test('preserves baseURL without trailing slash', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
    })

    expect(client.baseURL).toBe('https://api.example.com')
  })

  test('uses default timeout of 60000ms', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
    })

    expect(client.timeout).toBe(60000)
  })

  test('uses custom timeout when provided', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      timeout: 30000,
    })

    expect(client.timeout).toBe(30000)
  })

  test('uses default maxRetries of 2', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
    })

    expect(client.maxRetries).toBe(2)
  })

  test('uses custom maxRetries when provided', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 5,
    })

    expect(client.maxRetries).toBe(5)
  })

  test('sets default Content-Type header', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
    })

    expect(client.defaultHeaders['Content-Type']).toBe('application/json')
  })

  test('merges custom default headers', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      defaultHeaders: {
        'X-Custom-Header': 'custom-value',
      },
    })

    expect(client.defaultHeaders['Content-Type']).toBe('application/json')
    expect(client.defaultHeaders['X-Custom-Header']).toBe('custom-value')
  })

  test('custom headers can override default Content-Type', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      defaultHeaders: {
        'Content-Type': 'text/plain',
      },
    })

    expect(client.defaultHeaders['Content-Type']).toBe('text/plain')
  })

  test('stores apiKey', () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      apiKey: 'test-api-key',
    })

    expect(client.apiKey).toBe('test-api-key')
  })
})

// =============================================================================
// URL Building Tests
// =============================================================================

describe('URL building', () => {
  let client: HonchoHTTPClient
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0, // Disable retries for URL tests
    })
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('builds URL from path', async () => {
    let capturedURL = ''
    globalThis.fetch = async (url) => {
      capturedURL = url.toString()
      return mockResponse({ ok: true })
    }

    await client.get('/v1/test')

    expect(capturedURL).toBe('https://api.example.com/v1/test')
  })

  test('adds query parameters', async () => {
    let capturedURL = ''
    globalThis.fetch = async (url) => {
      capturedURL = url.toString()
      return mockResponse({ ok: true })
    }

    await client.get('/v1/test', {
      query: { page: 1, limit: 10 },
    })

    const url = new URL(capturedURL)
    expect(url.searchParams.get('page')).toBe('1')
    expect(url.searchParams.get('limit')).toBe('10')
  })

  test('handles boolean query parameters', async () => {
    let capturedURL = ''
    globalThis.fetch = async (url) => {
      capturedURL = url.toString()
      return mockResponse({ ok: true })
    }

    await client.get('/v1/test', {
      query: { active: true, deleted: false },
    })

    const url = new URL(capturedURL)
    expect(url.searchParams.get('active')).toBe('true')
    expect(url.searchParams.get('deleted')).toBe('false')
  })

  test('omits undefined query parameters', async () => {
    let capturedURL = ''
    globalThis.fetch = async (url) => {
      capturedURL = url.toString()
      return mockResponse({ ok: true })
    }

    await client.get('/v1/test', {
      query: { present: 'value', missing: undefined },
    })

    const url = new URL(capturedURL)
    expect(url.searchParams.get('present')).toBe('value')
    expect(url.searchParams.has('missing')).toBe(false)
  })
})

// =============================================================================
// Header Building Tests
// =============================================================================

describe('Header building', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('includes default headers', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    let capturedHeaders: Headers | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = new Headers(init?.headers as HeadersInit)
      return mockResponse({ ok: true })
    }

    await client.get('/test')

    expect(capturedHeaders?.get('Content-Type')).toBe('application/json')
  })

  test('includes Authorization header when apiKey provided', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      apiKey: 'secret-key',
      maxRetries: 0,
    })

    let capturedHeaders: Headers | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = new Headers(init?.headers as HeadersInit)
      return mockResponse({ ok: true })
    }

    await client.get('/test')

    expect(capturedHeaders?.get('Authorization')).toBe('Bearer secret-key')
  })

  test('merges request-specific headers', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    let capturedHeaders: Headers | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = new Headers(init?.headers as HeadersInit)
      return mockResponse({ ok: true })
    }

    await client.get('/test', {
      headers: { 'X-Request-ID': 'abc123' },
    })

    expect(capturedHeaders?.get('Content-Type')).toBe('application/json')
    expect(capturedHeaders?.get('X-Request-ID')).toBe('abc123')
  })

  test('request headers override default headers', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      defaultHeaders: { 'X-Custom': 'default' },
      maxRetries: 0,
    })

    let capturedHeaders: Headers | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = new Headers(init?.headers as HeadersInit)
      return mockResponse({ ok: true })
    }

    await client.get('/test', {
      headers: { 'X-Custom': 'override' },
    })

    expect(capturedHeaders?.get('X-Custom')).toBe('override')
  })
})

// =============================================================================
// HTTP Methods Tests
// =============================================================================

describe('HTTP methods', () => {
  let client: HonchoHTTPClient
  let originalFetch: typeof globalThis.fetch
  let capturedMethod: string

  beforeEach(() => {
    client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })
    originalFetch = globalThis.fetch
    capturedMethod = ''
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('get() uses GET method', async () => {
    globalThis.fetch = async (_, init) => {
      capturedMethod = init?.method || ''
      return mockResponse({ data: 'test' })
    }

    await client.get('/test')

    expect(capturedMethod).toBe('GET')
  })

  test('post() uses POST method', async () => {
    globalThis.fetch = async (_, init) => {
      capturedMethod = init?.method || ''
      return mockResponse({ data: 'test' })
    }

    await client.post('/test', { body: { key: 'value' } })

    expect(capturedMethod).toBe('POST')
  })

  test('put() uses PUT method', async () => {
    globalThis.fetch = async (_, init) => {
      capturedMethod = init?.method || ''
      return mockResponse({ data: 'test' })
    }

    await client.put('/test', { body: { key: 'value' } })

    expect(capturedMethod).toBe('PUT')
  })

  test('patch() uses PATCH method', async () => {
    globalThis.fetch = async (_, init) => {
      capturedMethod = init?.method || ''
      return mockResponse({ data: 'test' })
    }

    await client.patch('/test', { body: { key: 'value' } })

    expect(capturedMethod).toBe('PATCH')
  })

  test('delete() uses DELETE method', async () => {
    globalThis.fetch = async (_, init) => {
      capturedMethod = init?.method || ''
      return mockResponse({ data: 'test' })
    }

    await client.delete('/test')

    expect(capturedMethod).toBe('DELETE')
  })

  test('post() serializes body as JSON', async () => {
    let capturedBody: string | undefined
    globalThis.fetch = async (_, init) => {
      capturedBody = init?.body as string
      return mockResponse({ ok: true })
    }

    await client.post('/test', { body: { name: 'test', count: 42 } })

    expect(capturedBody).toBe('{"name":"test","count":42}')
  })
})

// =============================================================================
// Response Handling Tests
// =============================================================================

describe('Response handling', () => {
  let client: HonchoHTTPClient
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('parses JSON response body', async () => {
    globalThis.fetch = async () => mockResponse({ result: 'success', count: 42 })

    const data = await client.get<{ result: string; count: number }>('/test')

    expect(data.result).toBe('success')
    expect(data.count).toBe(42)
  })

  test('handles empty response body', async () => {
    globalThis.fetch = async () => mockResponse('')

    const data = await client.delete('/test')

    expect(data).toBeUndefined()
  })

  test('handles null response body', async () => {
    globalThis.fetch = async () => new Response(null, { status: 204 })

    const data = await client.delete('/test')

    expect(data).toBeUndefined()
  })
})

// =============================================================================
// Error Response Tests
// =============================================================================

describe('Error responses', () => {
  let client: HonchoHTTPClient
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0, // Disable retries to test error handling directly
    })
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('400 throws BadRequestError', async () => {
    globalThis.fetch = async () =>
      mockErrorResponse(400, { detail: 'Invalid input' })

    await expect(client.get('/test')).rejects.toBeInstanceOf(BadRequestError)
  })

  test('401 throws AuthenticationError', async () => {
    globalThis.fetch = async () =>
      mockErrorResponse(401, { detail: 'Invalid token' })

    await expect(client.get('/test')).rejects.toBeInstanceOf(AuthenticationError)
  })

  test('404 throws NotFoundError', async () => {
    globalThis.fetch = async () =>
      mockErrorResponse(404, { detail: 'Resource not found' })

    await expect(client.get('/test')).rejects.toBeInstanceOf(NotFoundError)
  })

  test('429 throws RateLimitError', async () => {
    globalThis.fetch = async () =>
      mockErrorResponse(429, { detail: 'Too many requests' })

    await expect(client.get('/test')).rejects.toBeInstanceOf(RateLimitError)
  })

  test('500 throws ServerError', async () => {
    globalThis.fetch = async () =>
      mockErrorResponse(500, { detail: 'Internal server error' })

    await expect(client.get('/test')).rejects.toBeInstanceOf(ServerError)
  })

  test('error includes message from response body', async () => {
    globalThis.fetch = async () =>
      mockErrorResponse(400, { detail: 'Name is required' })

    try {
      await client.get('/test')
      throw new Error('Should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(BadRequestError)
      expect((error as BadRequestError).message).toBe('Name is required')
    }
  })

  test('handles non-JSON error response', async () => {
    globalThis.fetch = async () =>
      new Response('Internal Server Error', {
        status: 500,
        headers: { 'Content-Type': 'text/plain' },
      })

    try {
      await client.get('/test')
      throw new Error('Should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(ServerError)
      // Falls back to HTTP status message
      expect((error as ServerError).message).toBe('HTTP 500')
    }
  })
})

// =============================================================================
// Retry Logic Tests
// =============================================================================

describe('Retry logic', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('retries on 429 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      if (attempts < 3) {
        return mockErrorResponse(429, { detail: 'Rate limited' })
      }
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(3) // Initial + 2 retries
    expect(result.success).toBe(true)
  })

  test('retries on 500 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      if (attempts < 3) {
        return mockErrorResponse(500, { detail: 'Server error' })
      }
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(3)
    expect(result.success).toBe(true)
  })

  test('retries on 502 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 1,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      if (attempts < 2) {
        return mockErrorResponse(502, { detail: 'Bad gateway' })
      }
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(2)
    expect(result.success).toBe(true)
  })

  test('retries on 503 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 1,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      if (attempts < 2) {
        return mockErrorResponse(503, { detail: 'Service unavailable' })
      }
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(2)
    expect(result.success).toBe(true)
  })

  test('retries on 504 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 1,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      if (attempts < 2) {
        return mockErrorResponse(504, { detail: 'Gateway timeout' })
      }
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(2)
    expect(result.success).toBe(true)
  })

  test('does not retry on 400 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      return mockErrorResponse(400, { detail: 'Bad request' })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(BadRequestError)
    expect(attempts).toBe(1) // No retries
  })

  test('does not retry on 401 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      return mockErrorResponse(401, { detail: 'Unauthorized' })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(AuthenticationError)
    expect(attempts).toBe(1)
  })

  test('does not retry on 404 status code', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      return mockErrorResponse(404, { detail: 'Not found' })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(NotFoundError)
    expect(attempts).toBe(1)
  })

  test('throws after exhausting retries', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      return mockErrorResponse(500, { detail: 'Server error' })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(ServerError)
    expect(attempts).toBe(3) // Initial + 2 retries
  })

  test('respects maxRetries = 0', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      return mockErrorResponse(500, { detail: 'Server error' })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(ServerError)
    expect(attempts).toBe(1) // No retries
  })
})

// =============================================================================
// Retry-After Header Tests
// =============================================================================

describe('Retry-After header parsing', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('parses Retry-After as seconds', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () =>
      mockErrorResponse(429, { detail: 'Rate limited' }, { 'Retry-After': '5' })

    try {
      await client.get('/test')
    } catch (error) {
      expect(error).toBeInstanceOf(RateLimitError)
      expect((error as RateLimitError).retryAfter).toBe(5000) // Converted to ms
    }
  })

  test('includes retryAfter in RateLimitError', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () =>
      mockErrorResponse(429, { detail: 'Rate limited' }, { 'Retry-After': '10' })

    try {
      await client.get('/test')
    } catch (error) {
      expect(error).toBeInstanceOf(RateLimitError)
      expect((error as RateLimitError).retryAfter).toBe(10000)
    }
  })
})

// =============================================================================
// Exponential Backoff Tests
// =============================================================================

describe('Exponential backoff', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('uses exponential backoff delays', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 3,
    })

    const delays: number[] = []
    const startTimes: number[] = []

    globalThis.fetch = async () => {
      const now = Date.now()
      if (startTimes.length > 0) {
        delays.push(now - startTimes[startTimes.length - 1])
      }
      startTimes.push(now)
      return mockErrorResponse(500, { detail: 'Server error' })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(ServerError)

    // Should have 3 delays (between 4 attempts)
    expect(delays.length).toBe(3)

    // Delays should be approximately: 500ms, 1000ms, 2000ms
    // Allow some tolerance for timing
    expect(delays[0]).toBeGreaterThanOrEqual(400)
    expect(delays[0]).toBeLessThan(700)
    expect(delays[1]).toBeGreaterThanOrEqual(900)
    expect(delays[1]).toBeLessThan(1200)
    expect(delays[2]).toBeGreaterThanOrEqual(1800)
    expect(delays[2]).toBeLessThan(2500)
  })
})

// =============================================================================
// Timeout Tests
// =============================================================================

describe('Timeout handling', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('throws TimeoutError when request exceeds timeout', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      timeout: 100, // 100ms timeout
      maxRetries: 0,
    })

    globalThis.fetch = async (_, init) => {
      // Create a promise that respects the abort signal
      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal

        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }

        const timeoutId = setTimeout(() => {
          resolve(mockResponse({ success: true }))
        }, 200)

        signal?.addEventListener('abort', () => {
          clearTimeout(timeoutId)
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(TimeoutError)
  })

  test('TimeoutError message includes timeout value', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      timeout: 50,
      maxRetries: 0,
    })

    globalThis.fetch = async (_, init) => {
      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal

        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }

        const timeoutId = setTimeout(() => {
          resolve(mockResponse({ success: true }))
        }, 100)

        signal?.addEventListener('abort', () => {
          clearTimeout(timeoutId)
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
    }

    try {
      await client.get('/test')
    } catch (error) {
      expect(error).toBeInstanceOf(TimeoutError)
      expect((error as TimeoutError).message).toContain('50ms')
    }
  })

  test('respects per-request timeout override', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      timeout: 1000, // Default 1s
      maxRetries: 0,
    })

    globalThis.fetch = async (_, init) => {
      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal

        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }

        const timeoutId = setTimeout(() => {
          resolve(mockResponse({ success: true }))
        }, 150)

        signal?.addEventListener('abort', () => {
          clearTimeout(timeoutId)
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
    }

    // Should timeout with 100ms override
    await expect(client.get('/test', { timeout: 100 })).rejects.toBeInstanceOf(
      TimeoutError
    )
  })

  test('successful request within timeout', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      timeout: 1000,
      maxRetries: 0,
    })

    globalThis.fetch = async () => {
      await new Promise((resolve) => setTimeout(resolve, 10))
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')
    expect(result.success).toBe(true)
  })

  test('retries on timeout when retries available', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      timeout: 50,
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async (_, init) => {
      attempts++

      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal

        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }

        if (attempts < 3) {
          // First two attempts timeout
          const timeoutId = setTimeout(() => {
            resolve(mockResponse({ success: false }))
          }, 100)

          signal?.addEventListener('abort', () => {
            clearTimeout(timeoutId)
            reject(new DOMException('Aborted', 'AbortError'))
          })
        } else {
          // Third attempt succeeds quickly
          resolve(mockResponse({ success: true }))
        }
      })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(3)
    expect(result.success).toBe(true)
  })
})

// =============================================================================
// AbortSignal Tests
// =============================================================================

describe('AbortSignal support', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('respects external AbortSignal', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    const controller = new AbortController()

    globalThis.fetch = async (_, init) => {
      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal

        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }

        const timeoutId = setTimeout(() => {
          resolve(mockResponse({ success: true }))
        }, 100)

        signal?.addEventListener('abort', () => {
          clearTimeout(timeoutId)
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
    }

    // Abort after 10ms
    setTimeout(() => controller.abort(), 10)

    await expect(
      client.get('/test', { signal: controller.signal })
    ).rejects.toBeInstanceOf(TimeoutError)
  })

  test('aborted requests trigger abort', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    const controller = new AbortController()
    let abortEventFired = false

    globalThis.fetch = async (_, init) => {
      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal

        // Check if already aborted
        if (signal?.aborted) {
          abortEventFired = true
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }

        // Listen for abort
        signal?.addEventListener('abort', () => {
          abortEventFired = true
          reject(new DOMException('Aborted', 'AbortError'))
        })

        // Slow request that will be aborted
        setTimeout(() => {
          resolve(mockResponse({ success: true }))
        }, 200)
      })
    }

    // Abort after 20ms
    setTimeout(() => controller.abort(), 20)

    await expect(
      client.get('/test', { signal: controller.signal })
    ).rejects.toBeInstanceOf(TimeoutError)

    // Verify abort was triggered
    expect(abortEventFired).toBe(true)
  })
})

// =============================================================================
// Streaming Tests
// =============================================================================

describe('stream() method', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('returns Response object for successful stream', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () => mockResponse('data: {"chunk": 1}\n')

    const response = await client.stream('POST', '/stream')

    expect(response).toBeInstanceOf(Response)
    expect(response.ok).toBe(true)
  })

  test('stream sets Accept header to text/event-stream', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    let capturedHeaders: Headers | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = new Headers(init?.headers as HeadersInit)
      return mockResponse('data: {"chunk": 1}\n')
    }

    await client.stream('POST', '/stream')

    expect(capturedHeaders?.get('Accept')).toBe('text/event-stream')
  })

  test('stream throws on error response', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () =>
      mockErrorResponse(500, { detail: 'Stream error' })

    await expect(client.stream('POST', '/stream')).rejects.toBeInstanceOf(
      ServerError
    )
  })

  test('stream does not retry on errors', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 3, // Retries are set but shouldn't apply to stream
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      return mockErrorResponse(500, { detail: 'Stream error' })
    }

    await expect(client.stream('POST', '/stream')).rejects.toBeInstanceOf(
      ServerError
    )

    // stream() doesn't implement retry logic
    expect(attempts).toBe(1)
  })
})

// =============================================================================
// Upload Tests
// =============================================================================

describe('upload() method', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('sends FormData body', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    let capturedBody: FormData | undefined
    globalThis.fetch = async (_, init) => {
      capturedBody = init?.body as FormData
      return mockResponse({ uploaded: true })
    }

    const formData = new FormData()
    formData.append('file', new Blob(['test content']), 'test.txt')

    const result = await client.upload<{ uploaded: boolean }>('/upload', formData)

    expect(result.uploaded).toBe(true)
    expect(capturedBody).toBeInstanceOf(FormData)
  })

  test('does not set Content-Type header (browser sets with boundary)', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    let capturedHeaders: Record<string, string> | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = init?.headers as Record<string, string>
      return mockResponse({ uploaded: true })
    }

    const formData = new FormData()
    formData.append('file', new Blob(['test']), 'test.txt')

    await client.upload('/upload', formData)

    // Content-Type should NOT be set (browser handles multipart boundary)
    expect(capturedHeaders?.['Content-Type']).toBeUndefined()
  })

  test('includes Authorization header when apiKey provided', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      apiKey: 'upload-key',
      maxRetries: 0,
    })

    let capturedHeaders: Record<string, string> | undefined
    globalThis.fetch = async (_, init) => {
      capturedHeaders = init?.headers as Record<string, string>
      return mockResponse({ uploaded: true })
    }

    const formData = new FormData()
    formData.append('file', new Blob(['test']), 'test.txt')

    await client.upload('/upload', formData)

    expect(capturedHeaders?.['Authorization']).toBe('Bearer upload-key')
  })

  test('upload handles empty response', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () => mockResponse('')

    const formData = new FormData()
    formData.append('file', new Blob(['test']), 'test.txt')

    const result = await client.upload('/upload', formData)

    expect(result).toBeUndefined()
  })

  test('upload throws on error response', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () =>
      mockErrorResponse(400, { detail: 'Invalid file' })

    const formData = new FormData()
    formData.append('file', new Blob(['test']), 'test.txt')

    await expect(client.upload('/upload', formData)).rejects.toBeInstanceOf(
      BadRequestError
    )
  })
})

// =============================================================================
// Connection Error Tests
// =============================================================================

describe('Connection errors', () => {
  let originalFetch: typeof globalThis.fetch

  beforeEach(() => {
    originalFetch = globalThis.fetch
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('network failure throws ConnectionError', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 0,
    })

    globalThis.fetch = async () => {
      throw new TypeError('fetch failed: Connection refused')
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(ConnectionError)
  })

  test('retries on connection error', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      if (attempts < 3) {
        throw new TypeError('fetch failed: Connection refused')
      }
      return mockResponse({ success: true })
    }

    const result = await client.get<{ success: boolean }>('/test')

    expect(attempts).toBe(3)
    expect(result.success).toBe(true)
  })

  test('throws ConnectionError after exhausting retries', async () => {
    const client = new HonchoHTTPClient({
      baseURL: 'https://api.example.com',
      maxRetries: 2,
    })

    let attempts = 0
    globalThis.fetch = async () => {
      attempts++
      throw new TypeError('fetch failed: Network error')
    }

    await expect(client.get('/test')).rejects.toBeInstanceOf(ConnectionError)
    expect(attempts).toBe(3)
  })
})

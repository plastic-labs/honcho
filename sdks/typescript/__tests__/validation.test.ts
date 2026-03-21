/**
 * Validation Schema Tests
 *
 * Unit tests for Zod validation schemas. No server required.
 */

import { describe, test, expect } from 'bun:test'
import { ZodError } from 'zod'
import {
  ChatQuerySchema,
  HonchoConfigSchema,
  MessageInputSchema,
  FileUploadSchema,
} from '../src/validation'

// =============================================================================
// ChatQuerySchema
// =============================================================================

describe('ChatQuerySchema', () => {
  // --- Valid inputs ---

  test('minimal valid query', () => {
    const result = ChatQuerySchema.parse({ query: 'hello' })
    expect(result.query).toBe('hello')
    expect(result.target).toBeUndefined()
    expect(result.session).toBeUndefined()
    expect(result.reasoningLevel).toBeUndefined()
  })

  test('query with all optional fields', () => {
    const result = ChatQuerySchema.parse({
      query: 'hello',
      target: 'peer-1',
      session: 'session-1',
      reasoningLevel: 'high',
    })
    expect(result.query).toBe('hello')
    expect(result.target).toBe('peer-1')
    expect(result.session).toBe('session-1')
    expect(result.reasoningLevel).toBe('high')
  })

  test('target as object with id is transformed to string', () => {
    const result = ChatQuerySchema.parse({
      query: 'hello',
      target: { id: 'peer-1' },
    })
    expect(result.target).toBe('peer-1')
  })

  test('session as object with id is transformed to string', () => {
    const result = ChatQuerySchema.parse({
      query: 'hello',
      session: { id: 'session-1' },
    })
    expect(result.session).toBe('session-1')
  })

  test.each(['minimal', 'low', 'medium', 'high', 'max'] as const)(
    'reasoning level "%s" is valid',
    (level) => {
      const result = ChatQuerySchema.parse({ query: 'hello', reasoningLevel: level })
      expect(result.reasoningLevel).toBe(level)
    }
  )

  // --- Missing required fields ---

  test('missing query throws', () => {
    expect(() => ChatQuerySchema.parse({})).toThrow(ZodError)
  })

  test('empty query throws', () => {
    expect(() => ChatQuerySchema.parse({ query: '' })).toThrow(ZodError)
  })

  test('whitespace-only query throws', () => {
    expect(() => ChatQuerySchema.parse({ query: '   ' })).toThrow(ZodError)
  })

  // --- Invalid field values ---

  test('invalid reasoning level throws', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', reasoningLevel: 'ultra' })
    ).toThrow(ZodError)
  })

  test('empty target string throws', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', target: '' })
    ).toThrow(ZodError)
  })

  test('empty session string throws', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', session: '' })
    ).toThrow(ZodError)
  })

  test('target with special characters throws', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', target: 'peer with spaces' })
    ).toThrow(ZodError)
  })

  test('session with special characters throws', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', session: 'session/bad' })
    ).toThrow(ZodError)
  })

  // --- Strict mode: unknown fields ---

  test('unknown field throws (strict)', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', typo: 'oops' })
    ).toThrow(ZodError)
  })

  test('misspelled field throws (strict)', () => {
    expect(() =>
      ChatQuerySchema.parse({ query: 'hello', reasoning_level: 'high' })
    ).toThrow(ZodError)
  })

  test('extra field alongside valid fields throws (strict)', () => {
    expect(() =>
      ChatQuerySchema.parse({
        query: 'hello',
        target: 'peer-1',
        session: 'session-1',
        reasoningLevel: 'low',
        extra: true,
      })
    ).toThrow(ZodError)
  })
})

// =============================================================================
// HonchoConfigSchema (strict validation)
// =============================================================================

describe('HonchoConfigSchema (strict)', () => {
  test('baseUrl (wrong casing) throws', () => {
    expect(() =>
      HonchoConfigSchema.parse({
        baseUrl: 'http://localhost:8000',
        workspaceId: 'test',
      })
    ).toThrow(ZodError)
  })

  test('baseURL (correct casing) passes', () => {
    const result = HonchoConfigSchema.parse({
      baseURL: 'http://localhost:8000',
      workspaceId: 'test',
    })
    expect(result.baseURL).toBe('http://localhost:8000')
  })

  test('unknown option throws', () => {
    expect(() =>
      HonchoConfigSchema.parse({
        workspaceId: 'test',
        retries: 3,
      })
    ).toThrow(ZodError)
  })
})

// =============================================================================
// MessageInputSchema (strict validation)
// =============================================================================

describe('MessageInputSchema (strict)', () => {
  test('valid message input', () => {
    const result = MessageInputSchema.parse({
      peerId: 'peer-1',
      content: 'hello',
      metadata: { key: 'value' },
    })
    expect(result.peerId).toBe('peer-1')
    expect(result.content).toBe('hello')
  })

  test('snake_case peerId alias throws (strict)', () => {
    expect(() =>
      MessageInputSchema.parse({
        peer_id: 'peer-1',
        content: 'hello',
      })
    ).toThrow(ZodError)
  })

  test('unknown field throws (strict)', () => {
    expect(() =>
      MessageInputSchema.parse({
        peerId: 'peer-1',
        content: 'hello',
        role: 'user',
      })
    ).toThrow(ZodError)
  })
})

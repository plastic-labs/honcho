/**
 * Test Helpers
 *
 * Shared utilities for test assertions and common patterns.
 */

import { expect } from 'bun:test'
import type {
  MessageResponse,
  PeerResponse,
  SessionResponse,
  WorkspaceResponse,
  ConclusionResponse,
  PageResponse,
} from '../src/types'

// =============================================================================
// Response Shape Assertions
// =============================================================================

/**
 * Assert that a response matches the WorkspaceResponse schema.
 */
export function assertWorkspaceShape(workspace: WorkspaceResponse): void {
  expect(workspace).toBeDefined()
  expect(typeof workspace.id).toBe('string')
  expect(workspace.id.length).toBeGreaterThan(0)
  expect(typeof workspace.metadata).toBe('object')
  expect(typeof workspace.configuration).toBe('object')
  expect(typeof workspace.created_at).toBe('string')
  // Validate ISO 8601 date format
  expect(() => new Date(workspace.created_at)).not.toThrow()
}

/**
 * Assert that a response matches the PeerResponse schema.
 */
export function assertPeerShape(peer: PeerResponse): void {
  expect(peer).toBeDefined()
  expect(typeof peer.id).toBe('string')
  expect(peer.id.length).toBeGreaterThan(0)
  expect(typeof peer.workspace_id).toBe('string')
  expect(typeof peer.metadata).toBe('object')
  expect(typeof peer.configuration).toBe('object')
  expect(typeof peer.created_at).toBe('string')
  expect(() => new Date(peer.created_at)).not.toThrow()
}

/**
 * Assert that a response matches the SessionResponse schema.
 */
export function assertSessionShape(session: SessionResponse): void {
  expect(session).toBeDefined()
  expect(typeof session.id).toBe('string')
  expect(session.id.length).toBeGreaterThan(0)
  expect(typeof session.workspace_id).toBe('string')
  expect(typeof session.is_active).toBe('boolean')
  expect(typeof session.metadata).toBe('object')
  expect(typeof session.configuration).toBe('object')
  expect(typeof session.created_at).toBe('string')
  expect(() => new Date(session.created_at)).not.toThrow()
}

/**
 * Assert that a response matches the MessageResponse schema.
 */
export function assertMessageShape(message: MessageResponse): void {
  expect(message).toBeDefined()
  expect(typeof message.id).toBe('string')
  expect(message.id.length).toBeGreaterThan(0)
  expect(typeof message.content).toBe('string')
  expect(typeof message.peer_id).toBe('string')
  expect(typeof message.session_id).toBe('string')
  expect(typeof message.workspace_id).toBe('string')
  expect(typeof message.metadata).toBe('object')
  expect(typeof message.created_at).toBe('string')
  expect(typeof message.token_count).toBe('number')
  expect(message.token_count).toBeGreaterThanOrEqual(0)
  expect(() => new Date(message.created_at)).not.toThrow()
}

/**
 * Assert that a response matches the ConclusionResponse schema.
 */
export function assertConclusionShape(conclusion: ConclusionResponse): void {
  expect(conclusion).toBeDefined()
  expect(typeof conclusion.id).toBe('string')
  expect(conclusion.id.length).toBeGreaterThan(0)
  expect(typeof conclusion.content).toBe('string')
  expect(typeof conclusion.observer_id).toBe('string')
  expect(typeof conclusion.observed_id).toBe('string')
  expect(typeof conclusion.session_id).toBe('string')
  expect(typeof conclusion.created_at).toBe('string')
  expect(() => new Date(conclusion.created_at)).not.toThrow()
}

/**
 * Assert that a response matches the PageResponse schema.
 */
export function assertPageShape<T>(
  page: PageResponse<T>,
  itemAssertion?: (item: T) => void
): void {
  expect(page).toBeDefined()
  expect(Array.isArray(page.items)).toBe(true)
  expect(typeof page.page).toBe('number')
  expect(page.page).toBeGreaterThanOrEqual(1)
  expect(typeof page.size).toBe('number')
  expect(page.size).toBeGreaterThan(0)
  expect(typeof page.total).toBe('number')
  expect(page.total).toBeGreaterThanOrEqual(0)
  expect(typeof page.pages).toBe('number')
  expect(page.pages).toBeGreaterThanOrEqual(0)

  if (itemAssertion) {
    for (const item of page.items) {
      itemAssertion(item)
    }
  }
}

// =============================================================================
// Test Data Generators
// =============================================================================

/**
 * Generate test message content with optional index.
 */
export function testMessage(index?: number): string {
  const suffix = index !== undefined ? ` #${index}` : ''
  return `Test message content${suffix} - ${Date.now()}`
}

/**
 * Generate test metadata.
 */
export function testMetadata(extra?: Record<string, unknown>): Record<string, unknown> {
  return {
    test: true,
    timestamp: Date.now(),
    ...extra,
  }
}

// =============================================================================
// Async Helpers
// =============================================================================

/**
 * Wait for a condition to be true, with timeout.
 */
export async function waitFor(
  condition: () => boolean | Promise<boolean>,
  options: { timeout?: number; interval?: number } = {}
): Promise<void> {
  const { timeout = 10000, interval = 100 } = options
  const start = Date.now()

  while (Date.now() - start < timeout) {
    if (await condition()) {
      return
    }
    await new Promise((resolve) => setTimeout(resolve, interval))
  }

  throw new Error(`waitFor timed out after ${timeout}ms`)
}

/**
 * Collect all items from an async iterable (for pagination testing).
 */
export async function collectAll<T>(iterable: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = []
  for await (const item of iterable) {
    items.push(item)
  }
  return items
}

/**
 * Collect streaming chunks into a single string.
 */
export async function collectStream(
  stream: AsyncIterable<string>
): Promise<string> {
  const chunks: string[] = []
  for await (const chunk of stream) {
    chunks.push(chunk)
  }
  return chunks.join('')
}

// =============================================================================
// Error Assertion Helpers
// =============================================================================

/**
 * Assert that an async function throws a specific error type.
 */
export async function expectError<E extends Error>(
  fn: () => Promise<unknown>,
  errorType: new (...args: unknown[]) => E,
  messageMatch?: string | RegExp
): Promise<E> {
  try {
    await fn()
    throw new Error(`Expected ${errorType.name} to be thrown`)
  } catch (error) {
    if (!(error instanceof errorType)) {
      throw new Error(
        `Expected ${errorType.name} but got ${error instanceof Error ? error.constructor.name : typeof error}`
      )
    }
    if (messageMatch) {
      if (typeof messageMatch === 'string') {
        expect(error.message).toContain(messageMatch)
      } else {
        expect(error.message).toMatch(messageMatch)
      }
    }
    return error
  }
}

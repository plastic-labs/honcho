/**
 * Test Setup
 *
 * Provides utilities for running integration tests against a live Honcho server.
 *
 * ============================================================================
 * Run with `bun run test` (from sdks/typescript), not bare `bun test`.
 *
 * `bun run test` runs scripts/run_ts_sdk_tests.py, which starts a throwaway
 * Honcho server (fresh database, API, mock LLM provider) and runs `bun test`
 * against it.
 * ============================================================================
 *
 * Environment variables (set automatically by `bun run test`):
 *   - HONCHO_TEST_URL: Base URL of the test server
 *   - HONCHO_TEST_API_KEY: API key for authentication
 */

import { Honcho } from '../src'

/**
 * Configuration for test runs.
 */
export const TEST_CONFIG = {
  baseURL: process.env.HONCHO_TEST_URL || 'http://localhost:8000',
  apiKey: process.env.HONCHO_TEST_API_KEY,
  timeout: 30000, // 30 seconds for integration tests
} as const

/**
 * Generate a unique workspace ID for test isolation.
 * Each test file should use its own workspace to avoid interference.
 */
export function generateWorkspaceId(prefix: string): string {
  const timestamp = Date.now()
  const random = Math.random().toString(36).substring(2, 8)
  return `test-${prefix}-${timestamp}-${random}`
}

/**
 * Generate a unique ID for test entities (peers, sessions, etc.)
 */
export function generateId(prefix: string): string {
  const random = Math.random().toString(36).substring(2, 10)
  return `${prefix}-${random}`
}

/**
 * Create a test client with an isolated workspace.
 * Returns the client and a cleanup function.
 */
export async function createTestClient(
  prefix: string
): Promise<{ client: Honcho; cleanup: () => Promise<void> }> {
  const workspaceId = generateWorkspaceId(prefix)

  const client = new Honcho({
    baseURL: TEST_CONFIG.baseURL,
    apiKey: TEST_CONFIG.apiKey,
    workspaceId,
    timeout: TEST_CONFIG.timeout,
  })

  // Ensure workspace exists by fetching metadata
  await client.getMetadata()

  const cleanup = async () => {
    try {
      await client.deleteWorkspace(workspaceId)
    } catch {
      // Workspace may already be deleted or not exist
    }
  }

  return { client, cleanup }
}

/**
 * Wait for the server to be ready.
 * Checks the /docs endpoint since there's no /health endpoint.
 */
export async function waitForServer(
  maxAttempts = 10,
  delayMs = 1000
): Promise<boolean> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      // Try the docs endpoint - it's always available
      const response = await fetch(`${TEST_CONFIG.baseURL}/docs`)
      if (response.ok) {
        // Also verify the API is actually working by creating a test workspace
        const apiResponse = await fetch(`${TEST_CONFIG.baseURL}/v3/workspaces`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: `health-check-${Date.now()}` }),
        })
        // 200/201 means working, 401 means auth required but server works
        if (apiResponse.ok || apiResponse.status === 401) {
          return true
        }
      }
    } catch {
      // Server not ready yet
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs))
  }
  return false
}

/**
 * Skip test if server is not available.
 * Use this in beforeAll to gracefully skip integration tests.
 */
export async function requireServer(): Promise<void> {
  // Wait up to 15 seconds (30 attempts × 500ms) for the server to be ready
  const ready = await waitForServer(30, 500)
  if (!ready) {
    throw new Error(
      '\n\n' +
        '╔══════════════════════════════════════════════════════════════════╗\n' +
        `║  ERROR: No Honcho server answering at ${TEST_CONFIG.baseURL}\n` +
        '║                                                                  ║\n' +
        '║  Run the suite with `bun run test` from sdks/typescript, which  ║\n' +
        '║  starts a throwaway server for it.                              ║\n' +
        '╚══════════════════════════════════════════════════════════════════╝\n'
    )
  }
}

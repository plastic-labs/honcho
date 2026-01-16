/**
 * Test Setup
 *
 * Provides utilities for running integration tests against a live Honcho server.
 *
 * ============================================================================
 * 🚨 DO NOT RUN `bun test` DIRECTLY - IT WILL FAIL 🚨
 *
 * These tests require a running server with database and Redis.
 * The test infrastructure is orchestrated via pytest from the monorepo root.
 *
 * To run these tests:
 *   cd /path/to/honcho  # monorepo root, NOT sdks/typescript
 *   uv run pytest tests/ -k typescript
 * ============================================================================
 *
 * Environment variables (set automatically by pytest):
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
  // This accounts for server startup time in pytest fixtures
  const ready = await waitForServer(30, 500)
  if (!ready) {
    throw new Error(
      '\n\n' +
        '╔══════════════════════════════════════════════════════════════════╗\n' +
        '║  ERROR: Cannot run tests - no server available                   ║\n' +
        '║                                                                  ║\n' +
        '║  You probably ran `bun test` directly. DON\'T DO THAT.           ║\n' +
        '║                                                                  ║\n' +
        '║  These tests MUST be run via pytest from the monorepo root:     ║\n' +
        '║                                                                  ║\n' +
        '║    cd /path/to/honcho                                           ║\n' +
        '║    uv run pytest tests/ -k typescript                           ║\n' +
        '║                                                                  ║\n' +
        '╚══════════════════════════════════════════════════════════════════╝\n'
    )
  }
}

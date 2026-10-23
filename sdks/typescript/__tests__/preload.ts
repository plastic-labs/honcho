/**
 * Test preload script - runs before any tests
 *
 * The suite needs a running Honcho server, which `bun run test` starts and
 * points HONCHO_TEST_URL at. Fail fast with a helpful message if it's missing.
 */

if (!process.env.HONCHO_TEST_URL) {
  console.error(`
╔══════════════════════════════════════════════════════════════════╗
║  ERROR: No test server. Use \`bun run test\`, not \`bun test\`.     ║
║                                                                  ║
║  \`bun run test\` starts a throwaway Honcho server (fresh database, ║
║  API, mock LLM provider) and runs \`bun test\` against it. It      ║
║  needs Postgres at DB_CONNECTION_URI.                            ║
║                                                                  ║
║  To use a server you already have running instead:              ║
║                                                                  ║
║    HONCHO_TEST_URL=http://localhost:8000 bun test               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
`)
  process.exit(1)
}

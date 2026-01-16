/**
 * Test preload script - runs before any tests
 *
 * This script checks if tests are being run via pytest (which sets HONCHO_TEST_URL)
 * and fails fast with a helpful message if not.
 */

if (!process.env.HONCHO_TEST_URL) {
  console.error(`
╔══════════════════════════════════════════════════════════════════╗
║  ERROR: Do not run \`bun test\` directly!                         ║
║                                                                  ║
║  These tests require a running server with database and Redis.  ║
║  The infrastructure is set up automatically by pytest.          ║
║                                                                  ║
║  Run tests from the monorepo root:                              ║
║                                                                  ║
║    cd /path/to/honcho                                           ║
║    uv run pytest tests/ -k typescript                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
`)
  process.exit(1)
}

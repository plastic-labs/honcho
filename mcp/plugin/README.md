# Honcho plugin (ChatGPT, Codex, dots)

The package we submit to OpenAI's plugin directory. One listing covers ChatGPT, Codex and dots.
It is an [Agent Plugins](https://agent-plugins.org) package: `plugin.json` holds the listing and
review fields, and `mcp.json` points at the hosted server, `https://mcp.honcho.dev`. Clients
discover OAuth from the server, so the package holds no credentials.

It lives here, not at the repo root, so the honcho repo itself never reads as a plugin and
contributors don't pick up a project MCP config.

## Submitting a new version

1. Bump `version` in `plugin.json`. Tool changes on the server don't need this: OpenAI rescans
   the server and picks them up on its own.
2. Zip the folder contents: `cd mcp/plugin && zip -r /tmp/honcho-plugin.zip . -x README.md`.
3. Upload the ZIP in the OpenAI plugin dashboard and fix any automated findings.

The reviewer test-account credentials never go in the package; enter them in the dashboard's
secure form. Add `review.demo_recording_url` once the walkthrough video is recorded. The review
test cases run against the `claude-review` test workspace. Domain verification is served by the Worker at
`/.well-known/openai-apps-challenge` from the `OPENAI_APPS_CHALLENGE` secret.

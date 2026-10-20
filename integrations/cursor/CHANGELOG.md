# Changelog

All notable changes to the Honcho plugin for Cursor and Grok Bot.

## [0.2.0] - 2026-10-15

First version in the Honcho monorepo, rewritten for the Cursor Marketplace.

- One plugin for Cursor and Grok Bot: a Cursor Marketplace listing is mirrored
  as a Grok Bot plugin page, so the same install serves both.
- Connects to the hosted Honcho MCP server at `https://mcp.honcho.dev` over
  HTTP with OAuth sign-in. No API key to paste, no local server, no `bun`.
- Skills drive the memory loop (`honcho-memory`), the Grok Bot conventions
  (`honcho-grok-bot`) and first-run setup (`honcho-setup`). Skills and the MCP
  server are the only components Grok Bot loads, so the plugin depends on
  nothing else.
- Cursor-only extras: an always-on rule, the `/recall` command, and three
  lifecycle hooks (`sessionStart`, `beforeSubmitPrompt`, `afterAgentResponse`)
  bundled into a committed `dist/hook.js`. The hooks
  read `~/.honcho/config.json` and exit silently without it, so an OAuth-only
  install keeps the skill-driven loop.
- Removed from the previous standalone repo: the stdio `mcp-server.ts`, the
  other six hooks (`stop`, `sessionEnd`, `postToolUse`, `preCompact`,
  `subagentStop`, `afterAgentThought`), the `/remember` command (the
  `honcho-memory` skill saves what the user asks it to remember) and the
  `honcho-dev` plugin.

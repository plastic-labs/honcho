---
name: honcho-cli
version: 0.1.0
description: A terminal for Honcho — memory that reasons.
---

# Honcho CLI — Agent Interface

## Overview

`honcho` is a CLI for administering and debugging Honcho workspaces. It wraps the Honcho Python SDK with agent-friendly defaults: JSON output, structured errors, input validation.

## Output Modes

- **TTY**: Human-readable tables (default when interactive)
- **Piped/scripted**: JSON automatically
- `--json`: Force JSON output

## Exit Codes

- 0: Success
- 1: Client error (bad input, not found)
- 2: Server error
- 3: Auth error

## Config

Shared with other Honcho tools at `~/.honcho/config.json`. The CLI owns only
`apiKey` and `environmentUrl` at the top level. Host-specific
entries under `hosts` are untouched.

Run `honcho init` to confirm or set those two values. Workspace / peer /
session are per-command — pass them via flags or env vars:

```bash
honcho peer card -w my-workspace -p my-peer
# or
export HONCHO_WORKSPACE_ID=my-workspace
export HONCHO_PEER_ID=my-peer
honcho peer card
```

## Command Groups

- `honcho config` — Manage CLI configuration
- `honcho workspace` — Inspect, delete, search workspaces
- `honcho peer` — Inspect, card, chat, search peers
- `honcho session` — Inspect, messages, context, summaries
- `honcho message` — List and get messages
- `honcho conclusion` — List, search, create, delete conclusions

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
- **Piped/scripted**: JSON/NDJSON automatically
- `--json`: Force JSON output
- `--quiet`: Suppress status messages

## Exit Codes

- 0: Success
- 1: Client error (bad input, not found)
- 2: Server error
- 3: Auth error

## Config

Stored at `~/.honcho/config.toml`. Set defaults to avoid repeating IDs:

```bash
honcho config set workspace_id my-workspace
honcho config set peer_id my-peer
```

## Command Groups

- `honcho config` — Manage CLI configuration
- `honcho workspace` — Inspect, delete, search workspaces
- `honcho peer` — Inspect, card, chat, search peers
- `honcho session` — Inspect, messages, context, summaries
- `honcho message` — List and get messages
- `honcho conclusion` — List, search, create, delete conclusions

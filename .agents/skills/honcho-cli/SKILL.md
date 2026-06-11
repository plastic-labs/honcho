---
name: honcho-cli
description: Inspect and debug Honcho workspaces via the `honcho` CLI. Use when investigating peer representations, memory state, session context, queue status, or dialectic quality — any task that requires introspection of a Honcho deployment.
allowed-tools: Bash(honcho:*), Bash(jq:*), Read, Grep
---

# Honcho CLI

`honcho` wraps the Honcho Python SDK with agent-friendly defaults: JSON output, structured errors, input validation. Use it to inspect workspace state, debug peer memory, and diagnose the dialectic.

## Output & config

- **TTY**: human-readable tables (default when interactive)
- **Piped / `--json`**: JSON — collection commands emit arrays, single-resource commands emit objects
- **Exit codes**: `0` success · `1` client error (bad input, not found) · `2` server error · `3` auth error
- **Config**: `~/.honcho/config.json` (shared with other Honcho tools). The CLI owns `apiKey` and `environmentUrl` at the top level; run `honcho init` to confirm or set them. Per-command scope (workspace / peer / session) is via `-w` / `-p` / `-s` flags or `HONCHO_*` env vars.

## Command groups

- `honcho config` — CLI configuration
- `honcho workspace` — inspect, delete, search
- `honcho peer` — inspect, card, chat, search
- `honcho session` — inspect, messages, context, summaries
- `honcho message` — list and get
- `honcho conclusion` — list, search, create, delete

## Rules

- Always pass `--json` when processing output programmatically.
- Run `honcho peer inspect` before `honcho peer chat` to understand context.
- Use `honcho session context` to see exactly what an agent receives.
- Never run `honcho workspace delete` without `honcho workspace inspect` first.
- Check queue status when derivation seems stalled.
- Compare peer card with conclusions to understand memory state.

## Inspection tour

When orienting to a Honcho deployment, walk outside-in:

### 1. Understand the workspace

```bash
honcho workspace inspect --json
```

### 2. Find the peer

```bash
honcho peer list --json
honcho peer inspect <peer_id> --json
```

### 3. Check peer's memory

```bash
honcho peer card <peer_id> --json
honcho conclusion list --observer <peer_id> --json
honcho conclusion search "topic" --observer <peer_id> --json
```

### 4. Debug a session

```bash
honcho session inspect <session_id> --json
honcho message list <session_id> --last 20 --json
honcho session context <session_id> --json
honcho session summaries <session_id> --json
```

### 5. Search across workspace

```bash
honcho workspace search "query" --json
honcho peer search <peer_id> "query" --json
```

## Debugging playbook

### Peer not learning?

```bash
# Is observation enabled?
honcho peer inspect <peer_id> --json | jq '.configuration'

# Is the deriver queue processing messages?
honcho workspace queue-status --json

# What conclusions exist?
honcho conclusion list --observer <peer_id> --json
honcho conclusion search "expected topic" --observer <peer_id> --json
```

### Session context looks wrong?

```bash
# Raw context an agent would receive
honcho session context <session_id> --json

# Summaries feeding the context
honcho session summaries <session_id> --json

# Recent message history
honcho message list <session_id> --last 50 --json
```

### Dialectic giving bad answers?

```bash
# What the peer card says
honcho peer card <peer_id> --json

# Conclusions on the specific topic
honcho conclusion search "topic" --observer <peer_id> --json

# Exercise the dialectic directly
honcho peer chat <peer_id> "what do you know about X?" --json
```

---
name: honcho-cli-inspect
version: 0.1.0
description: Inspect Honcho workspace state for debugging
---

# Honcho CLI — Inspection Skills

## Rules

- Always use `--json` when processing output programmatically
- Run `honcho peer inspect` before `honcho peer chat` to understand context
- Use `honcho session context` to see exactly what an agent receives
- Never run `honcho workspace delete` without `honcho workspace inspect` first

## Inspection Workflow

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
honcho session messages <session_id> --last 20 --json
honcho session context <session_id> --json
honcho session summaries <session_id> --json
```

### 5. Search across workspace

```bash
honcho workspace search "query" --json
honcho peer search <peer_id> "query" --json
```

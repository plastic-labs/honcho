---
name: honcho-cli-admin
version: 0.1.0
description: Admin operations for Honcho workspaces
---

# Honcho CLI — Admin Skills

## Rules

- Always inspect before deleting: `honcho workspace inspect` then `honcho workspace delete`
- Use `--dry-run` for destructive operations to preview impact
- Generated keys default to 90-day expiry; use `--no-expire` only when intentional
- Admin JWT required for key generation and workspace deletion

## Key Generation

```bash
# Workspace-scoped key (90-day default)
honcho key generate --workspace my-ws --json

# Peer-scoped key with custom expiry
honcho key generate --peer <id> --expires 30d --json

# No-expiry key (use with caution)
honcho key generate --workspace my-ws --no-expire --json
```

## Workspace Lifecycle

```bash
# Inspect first
honcho workspace inspect <workspace_id> --json

# Check queue status
honcho workspace queue-status --json

# Delete (requires --yes for non-interactive)
honcho workspace delete <workspace_id> --yes
```

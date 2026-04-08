# honcho-cli

Agent-first admin & debugging CLI for [Honcho](https://honcho.dev).

## Install

```bash
pip install honcho-cli
# or
uv pip install honcho-cli
```

## Quick Start

```bash
# Configure
honcho config init

# Or set values directly
honcho config set base_url https://api.honcho.dev
honcho config set api_key <your-admin-jwt>
honcho config set workspace_id <your-workspace>

# Inspect
honcho workspace inspect
honcho peer list
honcho peer inspect <peer_id>
honcho session messages <session_id> --last 20

# Peer management
honcho peer create <peer_id>
honcho peer create <peer_id> --observe-me --metadata '{"role": "user"}'
honcho peer get-metadata <peer_id>
honcho peer set-metadata <peer_id> --metadata '{"role": "user"}'
honcho peer representation <peer_id>
honcho peer representation <peer_id> --search-query "preferences" --max-conclusions 20

# Debug
honcho peer card <peer_id>
honcho conclusion search "topic" --observer <peer_id>
honcho workspace queue-status

# Admin
honcho key generate --workspace my-ws --expires 30d
honcho workspace delete <workspace_id> --yes
```

## Agent Usage

All commands output JSON when stdout isn't a TTY, or when `--json` is forced:

```bash
honcho peer list --json
honcho workspace inspect --json | jq '.peers'
```

Errors are structured:

```json
{
  "error": {
    "code": "PEER_NOT_FOUND",
    "message": "Peer 'abc' not found in workspace 'my-ws'",
    "details": {"workspace_id": "my-ws", "peer_id": "abc"}
  }
}
```

## Context Threading

Set defaults to avoid repeating IDs:

```bash
honcho config set peer_id peer_abc123
honcho peer inspect          # uses default
honcho peer card             # uses default
honcho peer inspect other_id # positional arg overrides
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HONCHO_BASE_URL` | API base URL |
| `HONCHO_API_KEY` | Admin JWT |
| `HONCHO_WORKSPACE_ID` | Default workspace |
| `HONCHO_PEER_ID` | Default peer |
| `HONCHO_SESSION_ID` | Default session |

## Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Force JSON output |
| `--quiet` / `-q` | Suppress status messages |
| `--workspace` / `-w` | Override workspace ID |
| `--peer` / `-p` | Override peer ID |
| `--session` / `-s` | Override session ID |
